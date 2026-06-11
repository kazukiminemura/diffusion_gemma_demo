# diffusion_gemma_demo

`google/diffusiongemma-26B-A4B-it` を Hugging Face Transformers で実行する CLI サンプルです。

Google の公式手順に合わせて `DiffusionGemmaForBlockDiffusion` と `AutoProcessor` を使います。DiffusionGemma 26B は大きいモデルなので、公式ドキュメントでは 60GB 超の GPU メモリを持つ GPU が必要とされています。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

Hugging Face のモデル利用条件への同意や認証が必要な場合は、事前にログインしてください。

```powershell
huggingface-cli login
```

## 使い方

単発プロンプト:

```powershell
diffusion-gemma "日本語でDiffusionGemmaの特徴を3つ説明して"
```

対話モード:

```powershell
diffusion-gemma
```

主なオプション:

```powershell
diffusion-gemma --system "You are a concise Japanese assistant." --max-new-tokens 256 "量子化とは？"
diffusion-gemma --think "難しい問題を段階的に解いて"
diffusion-gemma --stream "Write a short Python function"
diffusion-gemma --raw "Why is the sky blue?"
```

デフォルトではチャットタグ、特殊トークン、thinking チャンネルを取り除いて最終回答だけを表示します。`--raw` を付けるとデコード結果をそのまま表示します。

## 実装メモ

- モデル ID のデフォルトは `google/diffusiongemma-26B-A4B-it` です。
- `dtype="auto"` と `device_map="auto"` を `from_pretrained` に渡します。
- 対話モードでは、過去の assistant 応答には最終回答だけを保存します。
- CUDA がない場合や VRAM が 60GB 未満の場合は警告を出します。
