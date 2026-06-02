# 06 — 結論

## RTX 3060 12GB 是否能跑 Qwen3.6-35B-A3B？

**可以，而且跑得不錯。**

經過完整測試（50+ 組組合），RTX 3060 12GB 可以順暢執行這個 35B 參數的 MoE 模型，
在最佳配置下達到 **25-28 tok/s 的解碼速度**，足以滿足即時對話需求。

## 能跑到什麼程度？

| 指標 | 成績 | 備註 |
|:----|:----:|:----|
| 最大 decode 速度 | **27.73 tok/s**（一般版長文本） | 約每秒 28 個 token |
| 最大 prompt 速度 | **260 tok/s**（一般版長文本） | 約每秒 260 個 token |
| 典型實戰速度 | **20-25 tok/s** | 一般對話場景 |
| VRAM 使用 | **11.6-12.0 GB** | 幾乎吃滿 12GB |
| RAM 使用 | **13-15 GB** | 部分模型在系統記憶體 |
| 載入時間 | **10-20 秒**（GPU） vs **60-210 秒**（CPU） |
| 支援最大 context | 實測 32K，理論可達 128K+ |

## 推薦 Sweet Spot

### 一般版（最穩定、最快速）

| 參數 | 推薦值 |
|:----|:------|
| model | `Qwen3.6-35B-A3B-Q4_K_M.gguf` |
| ngl | 99（全部層在 GPU） |
| n_cpu_moe | 20 |
| threads | 8 |
| context | 32768（32K） |
| KV cache | q8_0 / q8_0 |
| 預計 decode | **~27 tok/s** |

### MTP 版（備選）

| 參數 | 推薦值 |
|:----|:------|
| model | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| ngl | 40（全部層在 GPU） |
| n_cpu_moe | 24 |
| threads | 8 |
| spec-type | draft-mtp |
| spec-draft-n-max | 3 |
| 預計 decode | **~26 tok/s** |

## 一般版 vs MTP 版：哪個值得保留？

**結論：一般版是首選，MTP 版可保留作為實驗用途。**

| 面向 | 一般版 | MTP 版 |
|:----|:-----|:------|
| decode 速度 | 27.73 tok/s 🏆 | 25.99 tok/s |
| VRAM 效率 | 11.6 GB（尚有餘裕） | 12.0 GB（緊繃） |
| RAM 用量 | 15.3 GB | 13.2 GB |
| 載入時間 | 快（~10 秒） | 中等（~20-30 秒） |
| 穩定性 | 高 | 中等（n_cpu_moe < 24 會 thrashing） |
| 模型大小 | 19.7 GB | 22.6 GB |

MTP 版本質上是一個有趣的技術，但在 12GB VRAM 的限制下，
draft head 佔用的 ~2.9 GB 空間造成了明顯的 trade-off：
- 少放 experts → decode 變慢
- 硬塞 experts → VRAM 滿載 thrashing

**如果 VRAM 更大（16GB+），MTP 的優勢可能更明顯。**

## 小顯卡跑大型 MoE 模型的實際定位

1. **12GB VRAM 是跑 35B MoE 的入門門檻。** 低於此規格（如 8GB）無法放足夠的 experts 在 GPU。
2. **Q4_K_M 量化是必要的。** 更低的量化（如 Q3_K 或 Q2_K）可以減小模型體積，但可能影響品質。
3. **MoE 的設計（每次只激活 8/256 experts）讓它特別適合 VRAM 受限的環境。**
   - 總參數 34.66B 但活躍參數僅 ~3.6B
   - 可以在 GPU 放部分 experts，其餘在 CPU
   - `--n-cpu-moe` 參數是調整 GPU/CPU 分配的關鍵
4. **載入時間主要受硬碟讀取速度影響。** 使用 NVMe SSD 可顯著縮短載入時間。
5. **實戰速度 20+ tok/s 對於聊天場景已經非常實用。** 人類閱讀速度約 5-10 tok/s，模型生成比閱讀快 2-4 倍。

## 給想複製測試者的建議

1. **下載 llama.cpp 時注意版本。**
   - 有裝 CUDA Toolkit → 用 `llama-b9444-bin-win-cuda-XX.x-x64.zip`
   - 沒裝 CUDA Toolkit → 用 `cudart-llama-bin-win-cuda-XX.x-x64.zip`
   - 驗證方式：`llama-server.exe --list-devices` 應列出 GPU

2. **先跑一個簡單測試確認 GPU 有正確運作。**
   ```bash
   llama-server.exe --list-devices
   nvidia-smi  # 確認載入後 VRAM 有上升
   ```

3. **不需要 `--spec-draft-model` 指向主模型。**
   Self-MTP（如 Unsloth 的 GGUF）只需 `--spec-type draft-mtp`。

4. **優先調整 `--n-cpu-moe`，再調整 `-ngl`。**
   對 35B MoE 模型來說，n_cpu_moe 的影響遠大於 ngl。

5. **不要在 VRAM 滿載的邊界運作。**
   如 n_cpu_moe=20 + MTP 時 VRAM 100%，decode 反而掉 40%。

6. **如果使用 Windows，注意 Git Bash 的 `$_` 符號在 PowerShell 指令中會被展開。**
   建議將 PowerShell 腳本存成 `.ps1` 檔案執行，避免 shell escaping 問題。

---

*報告完畢。測試日期：2026-06-02*
