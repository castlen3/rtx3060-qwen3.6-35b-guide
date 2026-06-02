# RTX 3060 12GB 跑 Qwen3.6-35B-A3B：3060 + X99 也能跑到 27 tok/s

> 📖 [好讀版（網頁版）](https://castlen3.github.io/rtx3060-qwen3.6-35b-guide/) — 五分鐘看懂結論、參數、踩坑紀錄

`Qwen3.6-35B-A3B` 是我目前最喜歡的個人本地模型之一。它的知識面夠廣、能力夠強，不管是日常問答、技術協作、整理資料，還是拿來配 agent 使用，都很能打。

問題也很現實：這類 35B MoE 模型就算量化後，模型本體、context 和執行開銷加起來，很多人第一直覺都會覺得至少要 `24GB VRAM` 以上才比較像樣，門檻一下就被拉高。

這份 repo 想回答的就是一個很實際的問題：`普通家用顯卡到底有沒有機會跑？`

答案是有。

我實測在 `RTX 3060 12GB` 搭配 `X99` 老平台上，`Qwen3.6-35B-A3B` 依然可以跑到大約 `27 tok/s`，而且不是勉強能跑，是已經進入日常可用、甚至相當順手的速度。

這份 repo 記錄了我在 `RTX 3060 12GB` 上測試 `Qwen3.6-35B-A3B` / `Qwen3.6-35B-A3B-UD (MTP)` 的完整過程、參數掃描結果、踩坑紀錄與結論。

重點不是只貼一個數字，而是讓別人可以：

- 快速知道 `12GB VRAM` 到底能不能跑這個 35B MoE 模型
- 直接複製我驗證過的 `llama.cpp` 參數
- 避開我前面踩過的 `CPU-only 偽 GPU` 大坑
- 把 repo 丟給 agent 後，能立刻問出「我該怎麼重現」和「這組參數差在哪」

## TL;DR

| 模型版本 | 最佳 decode | 推薦設定 |
|:--------|:-----------:|:--------|
| 一般版 | **27.73 tok/s** | `ngl=99, n_cpu_moe=20, threads=8, ctx=32768, q8_0/q8_0` |
| MTP 版 | **25.99 tok/s** | `ngl=40, n_cpu_moe=24, threads=8, draft=3, ctx=4096, q4_0/q4_0` |

### 一句話結論

`RTX 3060 12GB` 可以順跑 `Qwen3.6-35B-A3B`，而且速度相當實用。

- 一般版是這次測試的首選
- MTP 在 `12GB VRAM` 下沒有打贏一般版
- 前段很多數據其實是 `CPU-only`，因為一開始用了缺少 CUDA runtime 的 binary

## 最重要的發現

### 1. 真 GPU 與假 GPU 差很多

前期使用 `llama-b9444-bin-win-cuda-13.3-x64.zip` 時，`--list-devices` 沒有列出 CUDA 裝置，實際上是 `CPU fallback`。

換成 `cudart-llama-bin-win-cuda-13.3-x64.zip` 後，才真的啟用 GPU，加速幅度約 `2.5x`。

### 2. 一般版比 MTP 更適合 12GB VRAM

MTP 雖然理論上能透過 speculative decoding 加速，但 draft head 會佔掉額外 VRAM。
在 `RTX 3060 12GB` 這種容量邊界上，這筆成本會壓縮可放進 GPU 的 experts，最後讓一般版反而更快、更穩。

### 3. `--n-cpu-moe` 是最值得調的參數

對這個模型來說，`--n-cpu-moe` 對 decode 速度的影響比 `threads` 更大，也比很多人直覺先調的 `-ngl` 更重要。

## 如果你只想直接跑

### 一般版推薦

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

### MTP 版推薦

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

## 開始前先做這兩個檢查

### 檢查 1: `llama.cpp` 是否真的抓到 GPU

```bash
llama-server.exe --list-devices
```

你應該看到類似：

```text
Available devices:
  CUDA0: NVIDIA GeForce RTX 3060 (12287 MiB, 11255 MiB free)
```

如果這裡是空的，後面的 benchmark 幾乎都不可信。

### 檢查 2: 載入時 `nvidia-smi` 有沒有真的吃到 VRAM

成功情況下，載入模型後 VRAM 會從桌面待機的約 `469 MiB` 上升到 `11.6-12.0 GiB` 左右。

## Repo 內容

```text
rtx3060-qwen35b-benchmark/
├── README.md
├── QUICKSTART.md           # 最短重現路線
├── ASK_AGENT.md            # 丟給 agent 的提問模板
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

## 建議閱讀順序

1. 先看 `README.md` 拿結論與推薦參數
2. 再看 `QUICKSTART.md` 重現最小可行測試
3. 想追細節時看 `notes/05_analysis.md`
4. 想自己重跑或畫圖時讀 `data/results.csv`

## 這份資料適合回答什麼問題

- `RTX 3060 12GB` 能不能跑 `Qwen3.6-35B-A3B`？
- `一般版` 跟 `MTP 版` 在 12GB 上誰比較值得用？
- `--n-cpu-moe`、`-ngl`、`threads` 該怎麼調？
- 怎麼判斷自己是不是其實跑在 `CPU-only`？
- 哪些設定是快但不穩，哪些設定是穩定 sweet spot？

## 目前還沒有做的事

- 沒有附自動化 benchmark script
- 沒有附圖表版視覺化
- 沒有涵蓋 Linux / WSL / 多 GPU
- 沒有測更大的 context（例如 `128K` / `262K`）
- 沒有測不同量化（`Q3_K_*`、`Q5_K_*`）

## 給想 fork 後繼續補測的人

歡迎延伸這些方向：

- 不同 `ctx` 長度的 VRAM 曲線
- 不同量化格式的品質 / 速度權衡
- `Windows` 與 `Linux` 的差異
- `16GB+ VRAM` 下 MTP 是否開始反超一般版
- 將 `results.csv` 轉成圖表或 notebook

## 詳細內容

- 環境：[notes/01_environment.md](./notes/01_environment.md)
- 軟體：[notes/02_software.md](./notes/02_software.md)
- 模型：[notes/03_models.md](./notes/03_models.md)
- 指令：[notes/04_commands.md](./notes/04_commands.md)
- 分析：[notes/05_analysis.md](./notes/05_analysis.md)
- 結論：[notes/06_conclusion.md](./notes/06_conclusion.md)
- 原始摘要：[data/raw_log_excerpt.md](./data/raw_log_excerpt.md)
- 完整數據：[data/results.csv](./data/results.csv)
