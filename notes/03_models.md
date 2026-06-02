# 03 — 模型資訊

本測試使用兩個 Qwen3.6-35B-A3B 模型版本。

## 一般版本模型

| 項目 | 數值 |
|:----|:------|
| 模型名稱 | Qwen3.6-35B-A3B |
| 來源 | LM Studio Community |
| GGUF 檔名 | `Qwen3.6-35B-A3B-Q4_K_M.gguf` |
| 量化格式 | Q4_K_M (Medium) |
| 檔案大小 | 19.70 GiB（4.88 BPW） |
| 總參數數 | 34.66 B |
| 活躍參數數 | ~3.6 B（MoE, 8/256 experts） |
| 包含 MTP | 否 |
| GGUF 架構 | `qwen35moe` |
| KV pairs | 41 |
| Tensors | 733 |
| 原廠 context | 262,144 tokens |
| tokenizer | BPE (GPT-2 風格) |

### 模型架構

| 項目 | 數值 |
|:----|:------|
| 層數 | 40 |
| 全注意力層 | 10（每 4 層 1 層, full_attention_interval=4） |
| SSM 層 | 30 |
| embedding dim | 2048 |
| head count | 16 |
| KV head count | 2（GQA ratio=8） |
| key/value length | 256 |
| expert 總數 | 256 |
| expert 激活數 | 8 |
| expert FF dim | 512 |
| 共享 expert FF dim | 512 |
| SSM inner dim | 4096 |
| SSM state dim | 128 |
| rope dimension | 64 |

## MTP 版本模型

| 項目 | 數值 |
|:----|:------|
| 模型名稱 | Qwen3.6-35B-A3B-UD |
| 來源 | Unsloth（Hugging Face） |
| GGUF 檔名 | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| 量化格式 | Q4_K_M (Medium) |
| 檔案大小 | 22.60 GiB（~5.33 BPW 估算） |
| 總參數數 | 34.66 B + MTP draft head |
| 包含 MTP | ✅ 是（self-MTP，內建於同一 GGUF） |
| GGUF 架構 | `qwen35moe`（與一般版相同） |
| KV pairs | 55（比一般版多 14 組 metadata） |
| Tensors | 753（比一般版多 20 個 tensor，為 MTP head weights） |
| MTP 偵測 | ✅ llama.cpp b9444 可正確偵測（log 顯示 `[spec] estimated memory usage of MTP context is 890.85 MiB`） |
| 額外檔案 | `mmproj-F32.gguf`（多模態投影層，1.7 GB，本次未使用） |

### 與一般版差異

- MTP 版多了 **MTP (Multi-Token Prediction) draft head**
- 模型檔案大 ~2.9 GB（來自 draft head 權重）
- 不需要獨立的 draft model
- 使用 `--spec-type draft-mtp` 啟用，不需 `--spec-draft-model`
- 需 llama.cpp 支援 self-MTP 的版本（b9444 支援）
