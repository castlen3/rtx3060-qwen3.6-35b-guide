# 01 — 硬體環境

## 測試機器

| 項目 | 數值 |
|:----|:------|
| 機器代號 | 恆宇醫師工作站 |
| 主機板 | 未確認 |
| PCIe 世代 | 未確認 |
| 其他 GPU | 無（僅一張 RTX 3060） |

## CPU

| 項目 | 數值 |
|:----|:------|
| 型號 | Intel(R) Xeon(R) CPU E5-2666 v3 @ 2.90GHz |
| 核心數 | 10 核心 / 20 執行緒 |
| 架構 | Haswell-EP |
| TDP | 135W |

## 記憶體

| 項目 | 數值 |
|:----|:------|
| 容量 | 64 GB |
| 類型 | DDR3 |
| 頻率 | 1600 MHz |
| 配置 | 4 條 16 GB（NODE 1: 2 條, NODE 2: 2 條） |
| 廠牌 | Samsung + Undefined |

## GPU

| 項目 | 數值 |
|:----|:------|
| 型號 | NVIDIA GeForce RTX 3060 |
| VRAM | 12288 MiB（12 GB）GDDR6 |
| 驅動版本 | 610.47 (WDDM) |
| CUDA UMD 版本 | 13.3 |
| GPU 架構 | Ampere |
| CUDA 核心數 | 3584 |
| Tensor Cores | 112（第三代） |
| 實測可用 VRAM | ~11255 MiB（扣除系統固定佔用） |
| Power Limit | 預設 170W |
| PCIe | Gen4 x16（推測，未確認） |

## 測試時背景負載

| 項目 | 佔用 |
|:----|:----|
| 桌面環境 | ~469 MiB VRAM（explorer, chrome, terminal 等） |
| LM Studio | 未啟用模型，但有常駐 process |
| 其他大型程式 | 無，測試時已關閉 |

> 註：測試時 nvidia-smi 顯示 GPU 約有 11255 MiB free，
> 實際上模型可用 VRAM 約 11700~11800 MiB（含動態分配 overhead）。
