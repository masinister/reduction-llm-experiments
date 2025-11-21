from __future__ import annotations

import os
from pathlib import Path
from configparser import ConfigParser
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.ini"

def _get(parser: ConfigParser, section: str, key: str, type_: type, fallback: Any):
	try:
		if type_ is bool:
			return parser.getboolean(section, key, fallback=fallback)
		if type_ is int:
			return parser.getint(section, key, fallback=fallback)
		if type_ is float:
			return parser.getfloat(section, key, fallback=fallback)
		# default to string
		return parser.get(section, key, fallback=fallback)
	except Exception:
		# Do not raise on parse error here — return fallback
		return fallback


def load(path: str | Path | None = None) -> None:
	if path:
		cfg_path = Path(path)
	elif (Path.cwd() / "config.ini").exists():
		cfg_path = Path.cwd() / "config.ini"
	else:
		cfg_path = _CONFIG_PATH

	parser = ConfigParser()
	if cfg_path.exists():
		parser.read(cfg_path)
	else:
		parser.read_dict({})

	# Model settings
	global USE_TOY_MODEL, MODEL_ID, DTYPE
	DTYPE = _get(parser, "model", "dtype", str, "half")
	USE_TOY_MODEL = _get(parser, "model", "use_toy_model", bool, True)
	BIG_MODEL_ID = _get(parser, "model", "model_id", str, "openai/gpt-oss-20b")
	TOY_MODEL_ID = _get(parser, "model", "toy_model_id", str, "Qwen/Qwen3-0.6B")
	MODEL_ID = TOY_MODEL_ID if USE_TOY_MODEL else BIG_MODEL_ID


	# Inference defaults
	global TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS
	TEMPERATURE = _get(parser, "inference", "temperature", float, 0.7)
	TOP_P = _get(parser, "inference", "top_p", float, 0.95)
	TOP_K = _get(parser, "inference", "top_k", int, 50)
	MAX_TOKENS = _get(parser, "inference", "max_tokens", int, 2048)

	# System prompt
	global SYSTEM_PROMPT
	SYSTEM_PROMPT = _get(parser, "system", "system_prompt", str, "You are a helpful AI assistant.")

	# Hardware
	global TENSOR_PARALLEL_SIZE, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN
	TENSOR_PARALLEL_SIZE = _get(parser, "hardware", "tensor_parallel_size", int, 1)
	GPU_MEMORY_UTILIZATION = _get(parser, "hardware", "gpu_memory_utilization", float, 0.8)
	MAX_MODEL_LEN = _get(parser, "hardware", "max_model_len", int, 8192)

	# Compilation
	global CUDAGRAPH_MODE
	# CUDAGraphMode for vLLM: NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE
	CUDAGRAPH_MODE = _get(parser, "compilation", "cudagraph_mode", str, "PIECEWISE")


# Expose names in __all__
__all__ = [
	"_CONFIG_PATH",
	"USE_TOY_MODEL",
	"MODEL_ID",
	"TEMPERATURE",
	"TOP_P",
	"TOP_K",
	"MAX_TOKENS",
	"SYSTEM_PROMPT",
	"TENSOR_PARALLEL_SIZE",
	"GPU_MEMORY_UTILIZATION",
	"MAX_MODEL_LEN",
	"CUDAGRAPH_MODE",
	"load",
]
