#!/usr/bin/env python3
"""
Comprehensive smoke test for this repository.

What it verifies:
1) Package import and versions
2) Core dependencies present
3) GPU/CUDA availability and details (non-fatal if missing)
4) config.ini discoverable and readable (warn if missing)
5) Toy-model inference (single turn)
6) Multi-turn conversation memory
7) Parameter overrides (temperature/max_tokens)
"""

import sys
from pathlib import Path


def find_config() -> Path | None:
    """Search for config.ini in CWD or its parents."""
    for p in [Path.cwd(), *Path.cwd().parents]:
        candidate = p / "config.ini"
        if candidate.exists():
            return candidate
    return None


def main():
    print("=" * 80)
    print("SMOKE TEST")
    print("=" * 80)

    # 1) Package import and versions
    print("\n[1/7] Package import and versions...")
    try:
        from src.inference import Model, load_from_config
        print("✓ src.inference imported")
    except ImportError as e:
        print(f"✗ Failed to import package: {e}")
        print("Hint: run 'pip install -e .' in the repo root.")
        sys.exit(1)

    # 2) Dependencies
    print("\n[2/7] Checking core dependencies...")
    deps = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "vllm": "vLLM",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "pydantic": "Pydantic",
    }
    missing = []
    for module, name in deps.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "?")
            print(f"✓ {name} ({module}) v{ver}")
        except ImportError:
            print(f"✗ {name} ({module}) MISSING")
            missing.append(name)
    if missing:
        print(f"Missing: {', '.join(missing)}")
        sys.exit(1)

    # 3) GPU/CUDA details (non-fatal)
    print("\n[3/7] GPU / CUDA details...")
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"  - CUDA available: {cuda}")
        if cuda:
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - Devices: {torch.cuda.device_count()}")
            try:
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                mem_gb = props.total_memory / 1024 ** 3
                print(f"  - GPU 0: {name} ({mem_gb:.1f} GB)")
            except Exception:
                pass
        else:
            print("  ⚠ No GPU detected; running in CPU-only mode if supported")
    except Exception as e:
        print(f"⚠ torch/CUDA check failed: {e}")

    # 4) config.ini
    print("\n[4/7] Looking for config.ini...")
    try:
        import configparser

        cfg_path = find_config()
        if cfg_path:
            parser = configparser.ConfigParser()
            parser.read(cfg_path)
            print(f"✓ Found: {cfg_path}")
            if parser.has_section("model"):
                model_id = parser.get("model", "model_id", fallback="N/A")
                toy_id = parser.get("model", "toy_model_id", fallback="N/A")
                print(f"  - model_id: {model_id}")
                print(f"  - toy_model_id: {toy_id}")
        else:
            print("⚠ config.ini not found; load_from_config will require defaults/overrides")
    except Exception as e:
        print(f"⚠ config.ini check failed: {e}")

    # 5) Toy model single-turn inference
    print("\n[5/7] Toy model - single inference...")
    try:
        print("  Loading toy model (this may take a minute)...")
        model = load_from_config(toy=True, max_tokens=64, gpu_memory_utilization=0.5)
        result = model.infer("Say 'smoke test passed' if you can read this.")
        print("✓ Inference ok")
        print(f"  - Response: {result['text'][:120]}...")
        print(f"  - Tokens: {result['tokens']}")
        print(f"  - Latency: {result['latency_s']:.2f}s")
    except Exception as e:
        print(f"✗ Toy model inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6) Multi-turn memory
    print("\n[6/7] Multi-turn conversation memory...")
    try:
        r1 = model.infer("What is computational complexity?", session_id="smoke")
        print(f"  - Turn 1: {r1['text'][:100]}...")
        r2 = model.infer("Give a simple example.", session_id="smoke")
        print(f"  - Turn 2: {r2['text'][:100]}...")
        print("✓ Conversation ok")
    except Exception as e:
        print(f"✗ Multi-turn failed: {e}")
        sys.exit(1)

    # 7) Parameter override
    print("\n[7/7] Parameter overrides...")
    try:
        r = model.infer("Count to 5.", temperature=0.1, max_tokens=32)
        print(f"  - Response: {r['text'][:120]}...")
        print("✓ Overrides ok")
    except Exception as e:
        print(f"✗ Override test failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ ALL SMOKE TESTS PASSED")
    print("=" * 80)
    print("\nNext steps (optional):")
    print("  python examples/interactive.py --toy --prompt_file examples/example_prompts.txt")
    print("  # or")
    print("  python examples/reduction_batch.py")


if __name__ == "__main__":
    main()
