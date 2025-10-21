from __future__ import annotations
import os
import time
import threading
import configparser
import atexit
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


@dataclass
class _Turn:
    """Internal representation of a conversation turn."""
    user: str
    assistant: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Session:
    """Internal session state for multi-turn conversations."""
    session_id: str
    system_prompt: str
    turns: List[_Turn] = field(default_factory=list)
    
    def add_turn(self, user: str, assistant: str, meta: Optional[Dict[str, Any]] = None):
        self.turns.append(_Turn(user=user, assistant=assistant, meta=meta or {}))
    
    def build_messages(self, new_user_message: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        messages.append({"role": "user", "content": new_user_message})
        return messages


class Model:
    """Persistent vLLM model wrapper with session memory.
    
    Args:
        model_id: HuggingFace model ID or local path
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0, default 0.8)
        max_model_len: Maximum model sequence length (limits KV cache memory)
        system_prompt: Default system prompt for all sessions
        temperature: Default sampling temperature (0.0-2.0)
        top_p: Default nucleus sampling threshold
        top_k: Default top-k sampling limit
        max_tokens: Default max tokens to generate
        **kwargs: Additional vLLM.LLM initialization parameters
    
    Example:
        model = Model("meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.8)
        result = model.infer("Hello!", session_id="chat1")
        print(result["text"])
    """
    
    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: Optional[int] = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        max_tokens: int = 1024,
        **kwargs
    ):
        self.model_id = model_id
        self.default_system_prompt = system_prompt
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_top_k = top_k
        self.default_max_tokens = max_tokens
        self._closed = False
        
        # Initialize model and tokenizer
        print(f"[Model] Loading '{model_id}' (tp={tensor_parallel_size})...")
        
        # Configure distributed backend for multi-GPU
        vllm_kwargs = {
            "model": model_id,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": "auto",  # Avoids torch_dtype deprecation warning
            "gpu_memory_utilization": gpu_memory_utilization,
            "disable_log_stats": True,  # Reduce logging verbosity
        }
        
        # Add max_model_len if specified
        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len
        
        # Use ray backend for multi-GPU (SLURM clusters with A100/H200)
        if tensor_parallel_size > 1:
            vllm_kwargs["distributed_executor_backend"] = "ray"
        
        # Merge with user-provided kwargs (user kwargs take precedence)
        vllm_kwargs.update(kwargs)
        
        self._llm = LLM(**vllm_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._sessions: Dict[str, _Session] = {}
        self._lock = threading.Lock()
        print(f"[Model] Loaded.")

        # Ensure resources are cleaned up on interpreter shutdown
        atexit.register(self.close)

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        print("[Model] Cleaning up resources...")
        with self._lock:
            if getattr(self, "_closed", False):
                return
            # Shutdown vLLM if supported
            try:
                if hasattr(self, "_llm"):
                    self._llm = None  # Dereference the vLLM instance
            except Exception as e:
                print(f"[Model] Warning: vLLM cleanup failed: {e}")
            # Destroy torch.distributed process group if initialized
            try:
                import torch.distributed as dist  # type: ignore
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                print(f"[Model] Warning: destroy_process_group failed: {e}")
            self._closed = True
    
    def _get_session(self, session_id: str, system_prompt: str) -> _Session:
        """Get or create a session."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = _Session(
                    session_id=session_id,
                    system_prompt=system_prompt,
                )
            return self._sessions[session_id]
    
    def infer(
        self,
        prompt: str,
        session_id: str = "default",
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference with optional session memory.
        
        Args:
            prompt: User message to send
            session_id: Session identifier for conversation memory
            system_prompt: Override default system prompt for this session
            temperature: Override default temperature
            top_p: Override default top_p
            top_k: Override default top_k
            max_tokens: Override default max_tokens
            enable_thinking: Enable thinking tokens (model-dependent)
            **kwargs: Additional metadata to store with result
        
        Returns:
            dict: {
                "text": cleaned output text,
                "raw_text": raw model output,
                "tokens": number of tokens generated,
                "latency_s": generation time in seconds,
                "session_id": session identifier,
                "thinking": whether thinking was enabled,
                **kwargs
            }
        """
        # Use defaults if not overridden
        system_prompt = system_prompt or self.default_system_prompt
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p
        top_k = top_k if top_k is not None else self.default_top_k
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        # Build conversation history
        session = self._get_session(session_id, system_prompt)
        messages = session.build_messages(prompt)
        
        # Format with chat template
        input_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        
        # Generate
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        
        start = time.time()
        outputs = self._llm.generate([input_text], sampling)
        latency = time.time() - start
        
        # Extract and clean output
        raw = outputs[0].outputs[0].text
        # Handle multiple thinking delimiter formats:
        # 1. Standard: <think>...</think>
        # 2. gpt-oss: analysis...assistantfinal (case-insensitive)
        clean = raw
        if "</think>" in raw:
            clean = raw.split("</think>", 1)[1].strip()
        elif "assistantfinal" in raw.lower():
            # gpt-oss format: split after assistantfinal marker (case-insensitive)
            marker_pos = raw.lower().find("assistantfinal")
            clean = raw[marker_pos + len("assistantfinal"):].strip()
        else:
            clean = raw.strip()
        
        # Update session memory
        session.add_turn(user=prompt, assistant=clean, meta={"raw": raw})
        
        return {
            "text": clean,
            "raw_text": raw,
            "tokens": len(outputs[0].outputs[0].token_ids),
            "latency_s": latency,
            "session_id": session_id,
            "thinking": enable_thinking,
            **kwargs,
        }
    
    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
    
    def list_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        with self._lock:
            return list(self._sessions.keys())


def _find_config() -> Optional[Path]:
    """Find config.ini in current dir or parent dirs."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        config_path = parent / "config.ini"
        if config_path.exists():
            return config_path
    return None


def _load_config() -> Dict[str, Any]:
    """Load configuration from config.ini."""
    config_path = _find_config()
    if not config_path:
        return {}
    
    parser = configparser.ConfigParser()
    parser.read(config_path)
    
    config = {}
    if parser.has_section("model"):
        config["model_id"] = parser.get("model", "model_id", fallback=None)
        config["toy_model_id"] = parser.get("model", "toy_model_id", fallback=None)
    
    if parser.has_section("inference"):
        config["temperature"] = parser.getfloat("inference", "temperature", fallback=0.7)
        config["top_p"] = parser.getfloat("inference", "top_p", fallback=0.95)
        config["top_k"] = parser.getint("inference", "top_k", fallback=50)
        config["max_tokens"] = parser.getint("inference", "max_tokens", fallback=1024)
    
    if parser.has_section("system"):
        config["system_prompt"] = parser.get("system", "system_prompt", fallback="You are a helpful AI assistant.")
    
    if parser.has_section("hardware"):
        config["tensor_parallel_size"] = parser.getint("hardware", "tensor_parallel_size", fallback=1)
        config["gpu_memory_utilization"] = parser.getfloat("hardware", "gpu_memory_utilization", fallback=0.8)
        config["max_model_len"] = parser.getint("hardware", "max_model_len", fallback=None)
    
    return config


def load_from_config(toy: bool = False, **kwargs) -> Model:
    """Load model using settings from config.ini.
    
    Args:
        toy: If True, use toy_model_id instead of model_id
        **kwargs: Override any config parameters
    """
    config = _load_config()
    
    if toy:
        config["model_id"] = config.get("toy_model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Remove toy_model_id from config - it's not a Model parameter
    config.pop("toy_model_id", None)
    
    # Override with kwargs
    config.update(kwargs)
    
    if "model_id" not in config:
        raise ValueError("model_id must be specified in config.ini or as argument")
    
    return Model(**config)
