"""Core LLM interaction layer using instructor + vLLM server.

Thin wrapper around instructor for structured LLM outputs via
an OpenAI-compatible vLLM server endpoint. Automatically starts
and stops the vLLM server as needed.
"""

import atexit
import gc
import os
import socket
import subprocess
import sys
import time
from typing import Type, TypeVar

import instructor
import requests
from openai import OpenAI
from pydantic import BaseModel

from src import config
from src import debug_printer

# Request tracking for performance debugging
_request_count = 0
_total_time = 0.0

T = TypeVar('T', bound=BaseModel)

# Global server process for cleanup
_server_process: subprocess.Popen | None = None


def _cleanup_server():
    """Cleanup function to terminate server on exit."""
    global _server_process
    if _server_process is not None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
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


def _get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback for systems without psutil
        return -1.0


def _get_gc_stats() -> dict:
    """Get garbage collector statistics."""
    counts = gc.get_count()
    return {
        "gen0": counts[0],
        "gen1": counts[1],
        "gen2": counts[2],
        "tracked_objects": len(gc.get_objects()),
    }


def _fetch_vllm_metrics(base_url: str) -> dict | None:
    """Fetch metrics from vLLM server's /metrics endpoint.
    
    Returns parsed metrics dict or None if unavailable.
    """
    try:
        # vLLM metrics endpoint is on the same host but root path
        metrics_url = base_url.replace("/v1", "") + "/metrics"
        resp = requests.get(metrics_url, timeout=2)
        if resp.status_code != 200:
            return None
        
        # Parse Prometheus-style metrics
        metrics = {}
        for line in resp.text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            # Parse "metric_name{labels} value" or "metric_name value"
            if " " in line:
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    name_part, value = parts
                    # Extract metric name (before any {)
                    metric_name = name_part.split("{")[0]
                    try:
                        metrics[metric_name] = float(value)
                    except ValueError:
                        pass
        return metrics
    except Exception:
        return None


def _print_diagnostics(base_url: str, request_num: int) -> None:
    """Print diagnostic statistics for debugging resource leaks."""
    print(f"\n{'─'*60}")
    print(f"[Diagnostics for Request #{request_num}]")
    
    # Memory usage
    mem_mb = _get_memory_usage_mb()
    if mem_mb > 0:
        print(f"  Python process memory: {mem_mb:.1f} MB")
    
    # GC stats
    gc_stats = _get_gc_stats()
    print(f"  GC generations: gen0={gc_stats['gen0']}, gen1={gc_stats['gen1']}, gen2={gc_stats['gen2']}")
    print(f"  Tracked objects: {gc_stats['tracked_objects']:,}")
    
    # vLLM server metrics
    vllm_metrics = _fetch_vllm_metrics(base_url)
    if vllm_metrics:
        # Key metrics to watch for resource leaks
        kv_cache_usage = vllm_metrics.get("vllm:gpu_cache_usage_perc", -1)
        cpu_cache_usage = vllm_metrics.get("vllm:cpu_cache_usage_perc", -1)
        num_requests_running = vllm_metrics.get("vllm:num_requests_running", -1)
        num_requests_waiting = vllm_metrics.get("vllm:num_requests_waiting", -1)
        num_preemptions = vllm_metrics.get("vllm:num_preemptions_total", -1)
        
        print(f"  vLLM GPU KV cache usage: {kv_cache_usage*100:.1f}%" if kv_cache_usage >= 0 else "  vLLM GPU KV cache: N/A")
        print(f"  vLLM CPU cache usage: {cpu_cache_usage*100:.1f}%" if cpu_cache_usage >= 0 else "  vLLM CPU cache: N/A")
        print(f"  vLLM requests running: {int(num_requests_running)}" if num_requests_running >= 0 else "")
        print(f"  vLLM requests waiting: {int(num_requests_waiting)}" if num_requests_waiting >= 0 else "")
        if num_preemptions > 0:
            print(f"  ⚠️  vLLM preemptions (OOM indicator): {int(num_preemptions)}")
    else:
        print("  vLLM metrics: unavailable")
    
    print(f"{'─'*60}")


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
        self._request_count_since_reset = 0
        self._reset_interval = 10  # Reset client every N requests
        
        host, port = _parse_url(self.base_url)
        
        if auto_start and not _is_port_open(host, port):
            self._start_server(port)
        
        self._create_client()
    
    def _create_client(self):
        """Create or recreate the instructor client."""
        # Close any existing underlying HTTP client to avoid socket/FD buildup
        old_openai_client = getattr(self, "_openai_client", None)
        if old_openai_client is not None:
            close = getattr(old_openai_client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        self._openai_client = OpenAI(base_url=self.base_url, api_key="not-needed")
        self.client = instructor.from_openai(
            self._openai_client,
            mode=instructor.Mode.JSON_SCHEMA,
        )
    
    def _maybe_reset_client(self):
        """Reset client periodically to prevent connection pool buildup."""
        self._request_count_since_reset += 1
        if self._request_count_since_reset >= self._reset_interval:
            self._create_client()
            self._request_count_since_reset = 0
            if config.DEBUG:
                print("[Backend] Client reset to prevent resource buildup")
    
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
        
        # IMPORTANT: Do NOT pipe stdout without draining it.
        # If vLLM logs fill the pipe buffer, the server can deadlock and
        # requests will appear to hang/time out after some number of calls.
        _server_process = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
            text=True,
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
        global _request_count, _total_time
        _request_count += 1
        request_num = _request_count
        
        # Periodically reset client to prevent connection pool buildup
        self._maybe_reset_client()
        
        # Calculate prompt size
        prompt_chars = len(prompt)
        prompt_tokens_approx = prompt_chars // 4
        
        if config.DEBUG:
            print(f"\n{'='*60}")
            print(f"[Request #{request_num}] Starting...")
            print(f"  Prompt size: {prompt_chars:,} chars (~{prompt_tokens_approx:,} tokens)")
            print(f"  Max tokens: {max_tokens if max_tokens is not None else config.MAX_TOKENS}")
            print(f"  Response model: {response_model.__name__}")
            debug_printer.print_prompt(prompt)
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                temperature=temperature if temperature is not None else config.TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else config.MAX_TOKENS,
                max_retries=2,
                extra_body={"repetition_penalty": config.REPETITION_PENALTY},
            )
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[Request #{request_num}] FAILED after {elapsed:.2f}s")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error: {e}")
            raise
        
        elapsed = time.time() - start_time
        _total_time += elapsed
        avg_time = _total_time / _request_count
        
        if config.DEBUG:
            print(f"\n[Request #{request_num}] Completed in {elapsed:.2f}s")
            print(f"  Running avg: {avg_time:.2f}s/request")
            print(f"  Total requests: {_request_count}, Total time: {_total_time:.1f}s")
            
            # Print raw response for debugging
            raw_response = getattr(response, '_raw_response', None)
            debug_printer.print_raw_response(raw_response)
            
            debug_printer.print_response(response)
            
            _print_diagnostics(self.base_url, request_num)
        elif elapsed > 10:  # Always warn on slow requests
            print(f"\n[SLOW] Request #{request_num} took {elapsed:.2f}s (avg: {avg_time:.2f}s)")
            # Print diagnostics on slow requests even in non-debug mode
            _print_diagnostics(self.base_url, request_num)
        
        return response
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count (chars / 4)."""
        return len(text) // 4
