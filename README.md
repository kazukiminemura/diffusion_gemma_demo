# diffusion_gemma_demo

`unsloth/diffusiongemma-26B-A4B-it-GGUF` の量子化 GGUF を使って推論する CLI です。

DiffusionGemma は block-diffusion アーキテクチャなので、この GGUF は標準の `llama-cli` ではなく、DiffusionGemma 対応 build の `llama-diffusion-cli` で実行します。デフォルトは小さめの `Q4_K_M` です。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

Hugging Face の認証が必要な場合:

```powershell
huggingface-cli login
```

別途、DiffusionGemma 対応の `llama-diffusion-cli` を用意してください。

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release --target llama-diffusion-cli
```

## 使い方

初回実行時に `Q4_K_M` の GGUF を `models/unsloth/diffusiongemma-26B-A4B-it-GGUF` にダウンロードします。

```powershell
diffusion-gemma --llama-binary C:\path\to\llama-diffusion-cli.exe "日本語でDiffusionGemmaの特徴を3つ説明して"
```

既に GGUF を持っている場合:

```powershell
diffusion-gemma --llama-binary C:\path\to\llama-diffusion-cli.exe --model-path C:\models\diffusiongemma-26B-A4B-it-Q4_K_M.gguf "量子化とは？"
```

対話モード:

```powershell
diffusion-gemma --llama-binary C:\path\to\llama-diffusion-cli.exe
```

主なオプション:

```powershell
diffusion-gemma --quant Q8_0 "Write a short Python function"
diffusion-gemma --gpu-layers 0 "CPUだけで実行して"
diffusion-gemma --stream "Show the denoising canvas"
diffusion-gemma --system "You are a concise Japanese assistant." "要約して"
diffusion-gemma --think "難しい問題を段階的に解いて"
```

## メモ

- デフォルト repo は `unsloth/diffusiongemma-26B-A4B-it-GGUF` です。
- デフォルト quant は `Q4_K_M` です。
- `--stream` は `llama-diffusion-cli` に `--diffusion-visual` を渡します。
- `--max-new-tokens` は `llama-diffusion-cli` の `-n` に渡します。
- `--gpu-layers 99` がデフォルトです。CPU のみなら `--gpu-layers 0` を使います。
