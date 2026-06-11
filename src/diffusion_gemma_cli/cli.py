from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_REPO_ID = "unsloth/diffusiongemma-26B-A4B-it-GGUF"
DEFAULT_QUANT = "Q4_K_M"
DEFAULT_LOCAL_DIR = "models/unsloth/diffusiongemma-26B-A4B-it-GGUF"
DEFAULT_LLAMA_BINARY = "llama-diffusion-cli"
THINK_TOKEN = "<|think|>"


@dataclass(frozen=True)
class GgufOptions:
    repo_id: str
    quant: str
    model_path: Path | None
    local_dir: Path
    llama_binary: str
    gpu_layers: int
    max_new_tokens: int
    stream: bool
    raw: bool
    no_download: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffusion-gemma",
        description="Run the quantized Unsloth DiffusionGemma GGUF with llama-diffusion-cli.",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text. Omit for interactive chat mode.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hugging Face GGUF repo. Default: {DEFAULT_REPO_ID}")
    parser.add_argument("--quant", default=DEFAULT_QUANT, help=f"GGUF quantization to download. Default: {DEFAULT_QUANT}")
    parser.add_argument("--model-path", type=Path, help="Path to an existing .gguf file. Skips Hugging Face download.")
    parser.add_argument("--local-dir", type=Path, default=Path(DEFAULT_LOCAL_DIR), help=f"Download directory. Default: {DEFAULT_LOCAL_DIR}")
    parser.add_argument("--llama-binary", default=DEFAULT_LLAMA_BINARY, help=f"Path/name of llama-diffusion-cli. Default: {DEFAULT_LLAMA_BINARY}")
    parser.add_argument("--gpu-layers", type=int, default=99, help="Layers to offload to GPU via -ngl. Use 0 for CPU-only. Default: 99")
    parser.add_argument("--system", default="", help="Optional system prompt.")
    parser.add_argument("--think", action="store_true", help=f"Enable thinking mode by prefixing the system prompt with {THINK_TOKEN}.")
    parser.add_argument("--max-new-tokens", type=positive_int, default=512, help="Target generated tokens passed as -n. Default: 512")
    parser.add_argument("--stream", action="store_true", help="Show the live diffusion canvas with --diffusion-visual.")
    parser.add_argument("--raw", action="store_true", help="Do not post-process llama.cpp output.")
    parser.add_argument("--no-download", action="store_true", help="Require --model-path or an already downloaded matching GGUF.")
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = GgufOptions(
        repo_id=args.repo_id,
        quant=args.quant,
        model_path=args.model_path,
        local_dir=args.local_dir,
        llama_binary=args.llama_binary,
        gpu_layers=args.gpu_layers,
        max_new_tokens=args.max_new_tokens,
        stream=args.stream,
        raw=args.raw,
        no_download=args.no_download,
    )

    try:
        runner = DiffusionGemmaGgufRunner(options)
    except RuntimeError as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    system_prompt = make_system_prompt(args.system, args.think)
    if args.prompt:
        prompt = build_prompt(" ".join(args.prompt), system_prompt)
        return runner.run_once(prompt)

    return runner.run_chat()


class DiffusionGemmaGgufRunner:
    def __init__(self, options: GgufOptions) -> None:
        self.options = options
        self.binary = resolve_binary(options.llama_binary)
        self.model_path = resolve_model_path(options)

    def run_once(self, prompt: str) -> int:
        command = self.base_command()
        command.extend(["-p", prompt])
        return run_command(command, raw=self.options.raw)

    def run_chat(self) -> int:
        command = self.base_command()
        command.append("-cnv")
        return run_command(command, raw=True)

    def base_command(self) -> list[str]:
        command = [
            str(self.binary),
            "-m",
            str(self.model_path),
            "-ngl",
            str(self.options.gpu_layers),
            "-n",
            str(self.options.max_new_tokens),
        ]
        if self.options.stream:
            command.append("--diffusion-visual")
        return command


def resolve_binary(binary: str) -> Path:
    binary_path = Path(binary)
    if binary_path.exists():
        return binary_path

    found = shutil.which(binary)
    if found:
        return Path(found)

    raise RuntimeError(
        "llama-diffusion-cli was not found. Build the DiffusionGemma llama.cpp branch "
        "and pass its path with --llama-binary."
    )


def resolve_model_path(options: GgufOptions) -> Path:
    if options.model_path:
        path = options.model_path.expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"GGUF file does not exist: {path}")
        return path

    existing = sorted(options.local_dir.glob(f"*{options.quant}*.gguf"))
    if existing:
        return existing[0].resolve()

    if options.no_download:
        raise RuntimeError(f"No *{options.quant}*.gguf found in {options.local_dir}.")

    return download_gguf(options.repo_id, options.quant, options.local_dir)


def download_gguf(repo_id: str, quant: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `pip install -e .` to enable GGUF downloads.") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[f"*{quant}*.gguf"],
    )

    matches = sorted(local_dir.glob(f"*{quant}*.gguf"))
    if not matches:
        raise RuntimeError(f"Downloaded repo did not contain a *{quant}*.gguf file.")
    return matches[0].resolve()


def run_command(command: list[str], raw: bool) -> int:
    if raw:
        return subprocess.run(command, check=False).returncode

    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=None)
    print(clean_llama_output(completed.stdout))
    return completed.returncode


def make_system_prompt(system_prompt: str, think: bool) -> str:
    system_prompt = system_prompt.strip()
    if think and not system_prompt.startswith(THINK_TOKEN):
        return f"{THINK_TOKEN}\n{system_prompt}" if system_prompt else THINK_TOKEN
    return system_prompt


def build_prompt(user_prompt: str, system_prompt: str) -> str:
    if not system_prompt:
        return user_prompt
    return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def clean_llama_output(output: str) -> str:
    lines = []
    for line in output.splitlines():
        if line.startswith("llama_") or line.startswith("ggml_") or line.startswith("build:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


if __name__ == "__main__":
    raise SystemExit(main())
