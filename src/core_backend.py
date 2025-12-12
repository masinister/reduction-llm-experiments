"""Core LLM interaction layer using instructor + vLLM server.

Thin wrapper around instructor for structured LLM outputs via
an OpenAI-compatible vLLM server endpoint. Automatically starts
and stops the vLLM server as needed.
"""

import atexit
import socket
import subprocess
import time
from typing import Type, TypeVar

import instructor
from openai import OpenAI
from pydantic import BaseModel

from src import config
from src import debug_printer

T = TypeVar('T', bound=BaseModel)

# Global server process for cleanup
_server_process: subprocess.Popen | None = None


def _cleanup_server():
    """Cleanup function to terminate server on exit."""
    global _server_process
    if _server_process is not None:
        _server_process.terminate()
        _server_process.wait(timeout=10)
        _server_process = None


atexit.register(_cleanup_server)


def _is_port_open(host: str, port: int) -> bool:
    """Check if a port is open."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (OSError, socket.timeout):
        return False


def _parse_url(url: str) -> tuple[str, int]:
    """Extract host and port from URL."""
    # Remove protocol and path
    url = url.replace("http://", "").replace("https://", "")
    url = url.split("/")[0]
    if ":" in url:
        host, port = url.split(":")
        return host, int(port)
    return url, 8000


class Backend:
    """Instructor-based LLM client for structured outputs.
    
    Automatically starts a vLLM server if one isn't running.
    """
    
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        auto_start: bool = True,
    ):
        """Initialize the instructor client.
        
        Args:
            base_url: vLLM server URL (defaults to config.VLLM_URL)
            model: Model name (defaults to config.MODEL)
            auto_start: Whether to auto-start vLLM server if not running
        """
        self.base_url = base_url or config.VLLM_URL
        self.model = model or config.MODEL
        
        host, port = _parse_url(self.base_url)
        
        if auto_start and not _is_port_open(host, port):
            self._start_server(port)
        
        self.client = instructor.from_openai(
            OpenAI(base_url=self.base_url, api_key="not-needed"),
            mode=instructor.Mode.JSON_SCHEMA,
        )
    
    def _start_server(self, port: int):
        """Start vLLM server and wait for it to be ready."""
        global _server_process
        
        if _server_process is not None:
            return  # Already started
        
        print(f"Starting vLLM server with model {self.model}...")
        
        cmd = [
            "vllm", "serve", self.model,
            "--port", str(port),
            "--dtype", config.DTYPE,
            "--gpu-memory-utilization", str(config.GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", str(config.TENSOR_PARALLEL_SIZE),
            "--max-model-len", str(config.MAX_CONTEXT),
        ]
        
        _server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        
        # Wait for server to be ready
        host, _ = _parse_url(self.base_url)
        start_time = time.time()
        timeout = config.STARTUP_TIMEOUT
        
        while time.time() - start_time < timeout:
            if _is_port_open(host, port):
                # Give it a moment to fully initialize
                time.sleep(2)
                print(f"vLLM server ready on port {port}")
                return
            time.sleep(1)
        
        # Timeout - kill the process
        _server_process.terminate()
        _server_process = None
        raise RuntimeError(f"vLLM server failed to start within {timeout}s")
    
    def create(
        self,
        prompt: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Generate structured output from prompt.
        
        Args:
            prompt: User prompt
            response_model: Pydantic model for response
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Parsed Pydantic model instance
        """
        if config.DEBUG:
            debug_printer.print_prompt(prompt)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model,
            temperature=temperature if temperature is not None else config.TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS,
        )
        
        if config.DEBUG:
            debug_printer.print_response(response)
        
        return response
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count (chars / 4)."""
        return len(text) // 4
