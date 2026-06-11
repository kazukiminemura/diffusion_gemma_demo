from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Iterable


DEFAULT_MODEL_ID = "google/diffusiongemma-26B-A4B-it"
THINK_TOKEN = "<|think|>"


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int
    stream: bool
    raw: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diffusion-gemma",
        description="Run google/diffusiongemma-26B-A4B-it with Hugging Face Transformers.",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text. Omit for interactive chat mode.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help=f"Model ID to load. Default: {DEFAULT_MODEL_ID}")
    parser.add_argument("--system", default="", help="Optional system prompt.")
    parser.add_argument("--think", action="store_true", help=f"Enable thinking mode by prefixing the system prompt with {THINK_TOKEN}.")
    parser.add_argument("--max-new-tokens", type=positive_int, default=512, help="Maximum tokens to generate. Default: 512")
    parser.add_argument("--dtype", default="auto", help='dtype passed to from_pretrained. Default: "auto"')
    parser.add_argument("--device-map", default="auto", help='device_map passed to from_pretrained. Default: "auto"')
    parser.add_argument("--stream", action="store_true", help="Show diffusion intermediate text while generating.")
    parser.add_argument("--raw", action="store_true", help="Print the raw decoded sequence, including chat tags and special tokens.")
    parser.add_argument("--no-gpu-check", action="store_true", help="Skip the CUDA availability warning.")
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        runner = DiffusionGemmaRunner(
            model_id=args.model_id,
            dtype=args.dtype,
            device_map=args.device_map,
            check_gpu=not args.no_gpu_check,
        )
    except Exception as exc:  # pragma: no cover - depends on local GPU/model availability
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 1

    options = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        stream=args.stream,
        raw=args.raw,
    )
    system_prompt = make_system_prompt(args.system, args.think)

    if args.prompt:
        prompt = " ".join(args.prompt)
        response = runner.generate(
            messages=make_messages(prompt, system_prompt=system_prompt),
            options=options,
        )
        print(response)
        return 0

    return run_repl(runner, options=options, system_prompt=system_prompt)


class DiffusionGemmaRunner:
    def __init__(self, model_id: str, dtype: str, device_map: str, check_gpu: bool) -> None:
        self._warn_about_gpu(check_gpu)

        try:
            import torch
            from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -e .` before running the CLI.") from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    def generate(self, messages: list[dict[str, str]], options: GenerationOptions) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_kwargs = {
            **inputs,
            "max_new_tokens": options.max_new_tokens,
        }

        if options.stream:
            from transformers import TextDiffusionStreamer

            generate_kwargs["streamer"] = TextDiffusionStreamer(tokenizer=self.processor.tokenizer)

        with self.torch.inference_mode():
            output = self.model.generate(**generate_kwargs)

        decoded = self.processor.decode(output[0], skip_special_tokens=False)
        if options.raw:
            return decoded

        return extract_final_answer(decoded)

    def _warn_about_gpu(self, check_gpu: bool) -> None:
        if not check_gpu:
            return
        try:
            import torch
        except ImportError:
            return

        if not torch.cuda.is_available():
            print(
                "Warning: CUDA is not available. DiffusionGemma 26B usually requires a large GPU; loading may fail or be very slow.",
                file=sys.stderr,
            )
            return

        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if memory_gb < 60:
            print(
                f"Warning: detected CUDA device has {memory_gb:.1f}GB VRAM. Google recommends a GPU with more than 60GB for this model.",
                file=sys.stderr,
            )


def make_system_prompt(system_prompt: str, think: bool) -> str:
    system_prompt = system_prompt.strip()
    if think and not system_prompt.startswith(THINK_TOKEN):
        return f"{THINK_TOKEN}\n{system_prompt}" if system_prompt else THINK_TOKEN
    return system_prompt


def make_messages(prompt: str, system_prompt: str = "", history: Iterable[dict[str, str]] = ()) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def run_repl(runner: DiffusionGemmaRunner, options: GenerationOptions, system_prompt: str) -> int:
    print("DiffusionGemma chat. Type /exit or press Ctrl+C to quit.")
    history: list[dict[str, str]] = []

    while True:
        try:
            prompt = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0

        response = runner.generate(
            messages=make_messages(prompt, system_prompt=system_prompt, history=history),
            options=options,
        )
        print(f"\nmodel> {response}")

        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response})


def extract_final_answer(decoded: str) -> str:
    text = strip_after_first(decoded, "<turn|>")
    text = remove_prompt_turns(text)
    text = remove_thought_channel(text)
    text = remove_special_tokens(text)
    return collapse_padding(text).strip()


def strip_after_first(text: str, marker: str) -> str:
    if marker in text:
        return text.rsplit(marker, maxsplit=1)[0]
    return text


def remove_prompt_turns(text: str) -> str:
    model_marker = "<|turn>model"
    if model_marker in text:
        return text.split(model_marker, maxsplit=1)[1]
    return text


def remove_thought_channel(text: str) -> str:
    return re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", text, flags=re.DOTALL)


def remove_special_tokens(text: str) -> str:
    text = re.sub(r"<\|channel>[^<]*<channel\|>", "", text)
    text = re.sub(r"<[^>\n]+>", "", text)
    return text


def collapse_padding(text: str) -> str:
    text = text.replace("<pad>", "")
    text = text.replace("<eos>", "")
    text = text.replace("<bos>", "")
    return re.sub(r"\n{3,}", "\n\n", text)


if __name__ == "__main__":
    raise SystemExit(main())
