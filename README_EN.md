# RTX 3060 12GB Running Qwen3.6-35B-A3B: 27 tok/s on a Budget GPU

> [中文](README.md) | **English**

`Qwen3.6-35B-A3B` is one of my favorite local models. It's knowledgeable, capable, and works great for everything — daily chat, technical collaboration, data work, even as an agent backend.

The catch: a 35B MoE model, even quantized, sounds like it needs at least **24 GB VRAM**. That's a high bar for most people.

This repo answers one question: **Can a humble RTX 3060 12GB actually run it?**

The answer is **yes** — and not just barely. It's genuinely usable.

📖 **[Web Version](https://castlen3.github.io/rtx3060-qwen3.6-35b-guide/)** — clean visual summary, 5-minute read

## TL;DR

| Model Variant | Best Decode | Recommended Settings |
|:--------------|:-----------:|:---------------------|
| Standard | **27.73 tok/s** | `ngl=99, n_cpu_moe=20, threads=8, ctx=32768, q8_0/q8_0` |
| MTP | **25.99 tok/s** | `ngl=40, n_cpu_moe=24, draft=3, ctx=4096, q4_0/q4_0` |

### Bottom Line

**RTX 3060 12GB handles Qwen3.6-35B-A3B comfortably at practical speeds.**

- The standard version is the clear winner
- MTP doesn't beat it at this VRAM budget
- Many early results were actually CPU-only — I used a binary that silently fell back to CPU

## Key Findings

### 1. Real GPU vs. Fake GPU is a huge deal

Early tests used `llama-b9444-bin-win-cuda-13.3-x64.zip`, which **lacks CUDA runtime DLLs**. Despite specifying `-ngl 99`, everything ran on CPU.

Switching to `cudart-llama-bin-win-cuda-13.3-x64.zip` unlocked true GPU acceleration — about **2.5× faster**.

### 2. Standard beats MTP at 12 GB VRAM

MTP's speculative decoding sounds good on paper, but the draft head costs **~2.9 GB of extra VRAM**. At the 12 GB boundary, that overhead pushes out experts that could have been on GPU, and the standard model ends up faster and more stable.

### 3. `--n-cpu-moe` is the most impactful knob

For this MoE model, `--n-cpu-moe` matters more than `threads` and even `-ngl`. Dial this one first.

## Just Give Me the Command

### Standard (recommended)

```bash
llama-server.exe ^
  -m "Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  -t 8 ^
  --port 8080 ^
  -ngl 99 ^
  --n-cpu-moe 20 ^
  --no-mmap ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  -c 32768 ^
  -np 1 ^
  --reasoning off
```

> ✏️ **Update (2026-06-03):** Same params, keep `q8_0` KV cache, bump context to **64K** — still runs smoothly with negligible speed loss (~200-300 MB extra VRAM).

### MTP (alternative)

```bash
llama-server.exe ^
  -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" ^
  -t 8 ^
  --port 8080 ^
  -ngl 40 ^
  --n-cpu-moe 24 ^
  --no-mmap ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  -c 4096 ^
  -np 1 ^
  --reasoning off ^
  --spec-type draft-mtp ^
  --spec-draft-n-max 3 ^
  -fit off
```

## Two Checks Before You Start

### Check 1: Are you actually on GPU?

```bash
llama-server.exe --list-devices
```

You should see:
```
Available devices:
  CUDA0: NVIDIA GeForce RTX 3060 (12287 MiB, 11255 MiB free)
```

If this is empty, your benchmark results are meaningless.

### Check 2: Is VRAM actually going up?

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

- CPU-only: ~932 MiB (just desktop)
- Real GPU: ~11600 MiB (model loaded)

## Repo Structure

```text
rtx3060-qwen3.6-35b-guide/
├── README.md               # 中文
├── README_EN.md            # English (you are here)
├── index.html              # Web version
├── QUICKSTART.md
├── ASK_AGENT.md            # Prompts for your AI agent
├── notes/
│   ├── 01_environment.md
│   ├── 02_software.md
│   ├── 03_models.md
│   ├── 04_commands.md
│   ├── 05_analysis.md
│   └── 06_conclusion.md
└── data/
    ├── results.csv
    └── raw_log_excerpt.md
```

## Suggested Reading Order

1. Start here (`README.md` or `README_EN.md`) for conclusions and recommended settings
2. `QUICKSTART.md` for minimal reproduction steps
3. `notes/05_analysis.md` for deep-dive analysis
4. `data/results.csv` for raw data

## What This Repo Answers

- Can RTX 3060 12GB run Qwen3.6-35B-A3B?
- Standard vs. MTP: which one at 12 GB?
- How to tune `--n-cpu-moe`, `-ngl`, and `threads`
- How to tell if you're actually on CPU
- Which settings are stable sweet spots

## What's Not Covered (Yet)

- Automated benchmark scripts
- Charts and visualizations
- Linux / WSL / multi-GPU setups
- Larger contexts (128K / 262K)
- Other quantization formats (Q3_K_*, Q5_K_*)

## For Those Who Want to Extend

Feel free to fork and add:

- VRAM curves across different context lengths
- Quality/speed trade-offs for other quant formats
- Windows vs. Linux comparisons
- MTP at higher VRAM budgets (16 GB+)
- Charts and notebooks from `results.csv`

## Details

- Environment: [notes/01_environment.md](./notes/01_environment.md)
- Software: [notes/02_software.md](./notes/02_software.md)
- Models: [notes/03_models.md](./notes/03_models.md)
- Commands: [notes/04_commands.md](./notes/04_commands.md)
- Analysis: [notes/05_analysis.md](./notes/05_analysis.md)
- Conclusion: [notes/06_conclusion.md](./notes/06_conclusion.md)
- Raw logs: [data/raw_log_excerpt.md](./data/raw_log_excerpt.md)
- Full data: [data/results.csv](./data/results.csv)
