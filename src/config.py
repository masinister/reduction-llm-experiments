"""Configuration loader for reduction-llm."""

from pathlib import Path
from configparser import ConfigParser

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.ini"


class ConfigError(Exception):
    """Raised when a required config value is missing."""
    pass


def _get(parser: ConfigParser, section: str, key: str, type_: type):
    """Get a config value, raising ConfigError if missing."""
    if not parser.has_option(section, key):
        raise ConfigError(f"Missing required config: [{section}] {key}")
    try:
        if type_ is bool:
            return parser.getboolean(section, key)
        if type_ is int:
            return parser.getint(section, key)
        if type_ is float:
            return parser.getfloat(section, key)
        return parser.get(section, key)
    except ValueError as e:
        raise ConfigError(f"Invalid value for [{section}] {key}: {e}")


def load(path: str | Path | None = None) -> None:
    """Load configuration from INI file."""
    if path:
        cfg_path = Path(path)
    elif (Path.cwd() / "config.ini").exists():
        cfg_path = Path.cwd() / "config.ini"
    else:
        cfg_path = _CONFIG_PATH

    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")

    parser = ConfigParser()
    parser.read(cfg_path)

    # Server settings
    global VLLM_URL, MODEL, DTYPE, GPU_MEMORY_UTILIZATION, STARTUP_TIMEOUT
    VLLM_URL = _get(parser, "server", "vllm_url", str)
    MODEL = _get(parser, "server", "model", str)
    DTYPE = _get(parser, "server", "dtype", str)
    GPU_MEMORY_UTILIZATION = _get(parser, "server", "gpu_memory_utilization", float)
    STARTUP_TIMEOUT = _get(parser, "server", "startup_timeout", int)

    # Inference settings
    global TEMPERATURE, MAX_TOKENS, MAX_CONTEXT
    TEMPERATURE = _get(parser, "inference", "temperature", float)
    MAX_TOKENS = _get(parser, "inference", "max_tokens", int)
    MAX_CONTEXT = _get(parser, "inference", "max_context", int)

    # Chunking settings
    global CHUNK_SIZE, CHUNK_OVERLAP
    CHUNK_SIZE = _get(parser, "chunking", "chunk_size", int)
    CHUNK_OVERLAP = _get(parser, "chunking", "overlap", int)


__all__ = [
    "ConfigError",
    "VLLM_URL",
    "MODEL",
    "DTYPE",
    "GPU_MEMORY_UTILIZATION",
    "STARTUP_TIMEOUT",
    "TEMPERATURE",
    "MAX_TOKENS",
    "MAX_CONTEXT",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "load",
]
