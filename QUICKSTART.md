# Quick Start

這份文件是給第一次接手這個 repo 的人用的。
目標只有一個：先確認你真的跑在 GPU，再用最少步驟重現這份 benchmark 的核心結果。

## 1. 準備條件

- Windows
- NVIDIA GPU（本 repo 以 `RTX 3060 12GB` 為主）
- 對應版本的 `llama.cpp`
- 一般版或 MTP 版的 GGUF 模型

## 2. 先確認不是 CPU-only

執行：

```bash
llama-server.exe --list-devices
```

預期輸出：

```text
Available devices:
  CUDA0: NVIDIA GeForce RTX 3060 (12287 MiB, 11255 MiB free)
```

如果沒有列出 `CUDA0`：

- 你很可能拿到了缺少 CUDA runtime 的 binary
- 這時候 benchmark 結果不能拿來跟本 repo 對比

## 3. 最小重現: 一般版最佳配置

```bash
llama-server.exe ^
  -m "Qwen3.6-35B-A3B-Q4_K_M.gguf" ^
  -t 8 ^
  --port 8080 ^
  -ngl 40 ^
  --n-cpu-moe 28 ^
  --no-mmap ^
  --cache-type-k q8_0 ^
  --cache-type-v q8_0 ^
  -c 32768 ^
  -np 1 ^
  --reasoning off ^
  -fa on ^
  -fit off
```

> **`-fa on` 必須加。** 使用 q8_0 KV cache 時，新版 llama.cpp 強制要求開啟 flash attention。
> 不加的話 context 拉不上去，VRAM 會隨著 prompt 長度線性增長，32K 以上幾乎一定 OOM。
> 開啟後 KV cache 幾乎不佔額外 VRAM，實測可穩跑 64K context。

成功特徵：

- `--list-devices` 有列出 `CUDA0`
- 模型載入後 VRAM 上升到約 `8.8 GiB`
- 長文本 decode 約 `26 tok/s`
- 64K context 也可穩跑（`-c 65536`，peak VRAM ~9.1GB）

## 4. 最小重現: MTP 推薦配置

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

成功特徵：

- log 裡出現 `[spec] estimated memory usage of MTP context`
- VRAM 大約接近 `12 GiB`
- decode 約 `26 tok/s`

## 5. 常見踩坑

### `--list-devices` 是空的

代表你不是在真 GPU 模式。

### MTP 模型加了 `--spec-draft-model`

這份 repo 測的是 self-MTP GGUF，不需要再指定 `--spec-draft-model`。

### VRAM 吃滿後反而變慢

尤其是 MTP 版在 `n_cpu_moe=20` 時，速度可能大幅下降。

### 沒開 `-fa on` 導致 long context OOM

使用 q8_0 KV cache 時，若未開啟 flash attention（`-fa on`），VRAM 會隨 prompt 長度線性膨脹。
實測 28K tokens 就多吃 ~4.2 GB，64K 一定會 OOM。新版 llama.cpp 在 q8_0 下甚至會直接報錯不給跑。
**解決：一律加 `-fa on`。**

## 6. 怎麼跟 agent 配合最快

直接把這個 repo 丟給 agent，然後貼上 [ASK_AGENT.md](./ASK_AGENT.md) 裡的 prompt。
