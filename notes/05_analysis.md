# 05 — 分析筆記

## 1. 一般版 vs MTP 版實測差異

### 真實 GPU 加速下的對比（最佳配置）

| 項目 | 一般版 (n_cpu_moe=20) | MTP 版 (n_cpu_moe=24, draft=3) |
|:----|:-------------------:|:----------------------------:|
| decode tok/s | 27.73 | 25.99 |
| prompt tok/s | 260.20 | 175.01 |
| VRAM 用量 | ~11.6 GB | ~12.0 GB |
| RAM 用量 | ~15.3 GB | ~13.2 GB |
| 載入時間 | ~10 秒 | ~20-30 秒 |

**結論：在真實 GPU 加速下，一般版反而比 MTP 版快約 6.7%。**

MTP 模型因多了 ~2.9 GB 的 draft head 權重，壓縮了可用的模型層 VRAM 空間。
雖然 MTP 理論上可以透過 speculative decoding 加速，但在 12GB VRAM 的限制下，
draft head 佔用的資源反而排擠了主模型的 experts 放置，最終得不償失。

## 2. MTP 是否真的有帶來 decode 加速

### 同一條件下的 MTP 效果（ngl=40, n_cpu_moe=41）

| draft | decode tok/s | vs 無 MTP（參考一般版） |
|:----:|:----------:|:--------------------:|
| 1 | 19.13 | — |
| 2 | 18.29 | — |
| 3 | 20.80 🏆 | — |

> 註：此測試使用 n_cpu_moe=41（大部分 expert 在 CPU），decode 偏低。
> 與一般版最佳（n_cpu_moe=20, 27.73 tok/s）無法直接對比，
> 因 MTP 版的 VRAM 已被 draft head 佔用，無法放到 n_cpu_moe=20。

**結論：MTP 自身有加速效果（draft=3 比 draft=2 快 14%），**
**但因為模型更大導致能放在 GPU 的 expert 變少，整體不如一般版。**

## 3. `--spec-draft-n-max` 增加後是否有收益遞減

### 真實 GPU + ngl=40 + n_cpu_moe=41

| draft | decode tok/s | 增幅 |
|:----:|:----------:|:---:|
| 1 | 19.13 | baseline |
| 2 | 18.29 | -4.4% ❌ |
| 3 | 20.80 | +8.7% ✅ |

走勢非線性：draft=2 反而比 draft=1 慢（可能因 accept rate 下降），draft=3 回到最高點。

### CPU-only 時代（錯誤環境）的數據僅供參考

| draft | decode tok/s |
|:----:|:----------:|
| 1 | 8.76 |
| 2 | 9.60 |
| 3 | 10.99 |
| 4 | 11.15 |
| 5 | 11.94 🏆 |
| 7 | 11.08 |

> ⚠️ 以上為 CPU-only 數據，因缺失 CUDA runtime。
> 真實 GPU 下的行為可能完全不同。

## 4. `--n-cpu-moe` 對速度與穩定性的影響

### 一般版（真實 GPU, ngl=99, threads=8, long prompt）

| n_cpu_moe | decode tok/s | VRAM | 備註 |
|:--------:|:----------:|:---:|:----|
| 20 | 27.73 🏆 | 11.6 GB | VRAM 充份利用 |
| 16 | 14.74 | <11 GB | 更多 expert 在 CPU |
| 12 | 12.48 | <10 GB | |
| 8 | 10.50 | <9 GB | |
| 4 | 9.67 | <8 GB | |
| 0 | 8.59 | <7 GB | 全部 expert on CPU，最慢 |

**關鍵發現：n_cpu_moe 越低（更多 expert 在 GPU），decode 越快。**
但到了 n_cpu_moe=0 反而最慢，因為全部 256 個 expert 放在 CPU 上。

### MTP 版（真實 GPU, ngl=40, draft=3, long prompt）

| n_cpu_moe | decode tok/s | VRAM | 備註 |
|:--------:|:----------:|:---:|:----|
| 41 | 20.88 | 4.8 GB | 保守設定 |
| 36 | 24.95 | 6.7 GB | |
| 32 | 24.55 | 8.6 GB | |
| 28 | 23.57 | 10.5 GB | |
| 24 | 25.99 🏆 | 12.0 GB ⚠️ | 最佳，VRAM 緊繃 |
| 20 | 16.26 ❌ | 12.0 GB | VRAM 滿載 thrashing |

**關鍵發現：n_cpu_moe=24 是 MTP 版的甜蜜點，VRAM 剛好吃滿但不溢出。**
n_cpu_moe=20 時 VRAM 完全沒有剩餘空間，導致 CUDA kernel 啟動失敗和 thrashing。

## 5. `--threads` 對速度的影響

### RTX 3060 CUDA 時代（真實 GPU）

在 GPU 加速下，threads 對 decode 的影響顯著減小。
GPU 承擔主要計算，CPU threads 主要用於：
- 資料傳輸
- MoE expert 的 CPU 部分（若有）
- 排程與控制

預設 threads=8 是合理選擇，不需刻意調高或調低。

## 6. `-ngl` 對 VRAM、RAM、速度與 OOM 的影響

### MTP 版（n_cpu_moe=41, draft=2）

| ngl | decode tok/s | VRAM | RAM |
|:--:|:----------:|:---:|:---:|
| 12 | 8.21 | 3.4 GB | 22.0 GB |
| 16 | 8.78 | 3.2 GB | 21.9 GB |
| 20 | 9.83 | 3.8 GB | 21.7 GB |
| 24 | 11.45 | 3.6 GB | 21.5 GB |
| 28 | 11.85 | 4.1 GB | 21.3 GB |
| 32 | 14.00 | 3.9 GB | 21.2 GB |
| 36 | 16.85 | 3.9 GB | ~21 GB |
| 40 | 18.29 🏆 | 4.1 GB | ~21 GB |

**結論：ngl 越高越快，且所有值在 12GB VRAM 內都安全。**
VRAM 用量增加不明顯（因為 n_cpu_moe=41 把 experts 放在 CPU 上），
主要是 attention 層轉移到 GPU。

## 7. context length 的影響

### 目前僅測試 context=4096 與 32768（真實 GPU）

- context=4096：載入快（~10-20 秒），VRAM 用量低
- context=32768：載入稍慢（~60 秒），VRAM 多 ~200-500 MB

KV cache 公式（q4_0）：
- 10 層 attention × 2 頭 × 256 dim × ctx × 0.5625 bytes
- ctx=4096：~23 MB
- ctx=32768：~184 MB
- ctx=262144（原生最大）：~1.5 GB

以 12GB VRAM 來說，開到 128K context 理論上都還在範圍內，
但需要實測驗證。

## 8. 哪些設定穩定可用

### ✅ 高度穩定（大量測試，無 crash）

- 一般版：ngl=99, n_cpu_moe=20, threads=8, ctx=32768, q8_0
- MTP 版：ngl=40, n_cpu_moe=24~41, draft=1~3, ctx=4096, q4_0

### ⚠️ 可跑但不穩定

- MTP 版 n_cpu_moe=20：VRAM 滿載，偶發 thrashing

### ❌ 容易 OOM 或 crash

- MTP 版 + `--spec-draft-model [same_file]`（不需此參數）

## 9. 關於前期 CPU-only 的重大失誤

測試過程中有約 2/3 的數據是在 CPU-only 環境下取得的，
原因是下載了 `llama-b9444-bin-win-cuda-13.3-x64.zip`（缺少 CUDA runtime DLL）。

**判斷方式：**
- `llama-server.exe --list-devices` 輸出為空 → 無 GPU
- `nvidia-smi` 中無 `llama-server` process
- 模型載入後 VRAM 僅 ~932 MB（僅桌面程式）
- 載入時間 ~60 秒（而非 GPU 的 ~10-20 秒）
- RAM 高達 43 GB（MTP 模型全部在系統記憶體）

**修正方式：** 換用 `cudart-llama-bin-win-cuda-13.3-x64.zip`（內附 cublas64_13.dll, cublasLt64_13.dll, cudart64_13.dll）。
