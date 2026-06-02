# 02 — 軟體環境

## 主要軟體

| 項目 | 數值 |
|:----|:------|
| 作業系統 | Windows 10（版本未確認） |
| Shell 環境 | Git Bash (MSYS2) + PowerShell |
| Python | 3.11.15 |
| 測試日期 | 2026-06-02 |

## llama.cpp

| 項目 | 數值 |
|:----|:------|
| 來源 | GitHub Releases: ggml-org/llama.cpp |
| 版本標籤 | b9444 (6f165c1c6) |
| 編譯方式 | 官方預構建二進位檔（prebuilt binary） |
| 編譯工具鏈 | Clang 19.1.5 for Windows x86_64 |
| 執行檔 | llama-server.exe, llama-cli.exe |

### 重要：CUDA 後端安裝歷程

測試過程中使用過兩種版本：

| 階段 | 版本 | CUDA 支援 | 說明 |
|:----|:----|:---------:|:----|
| 階段一 | `llama-b9444-bin-win-cuda-13.3-x64.zip` | ❌ 無效 | 缺少 CUDA runtime DLL（cublas64_13.dll 等），`--list-devices` 為空，實為 CPU-only |
| 階段二 | `cudart-llama-bin-win-cuda-13.3-x64.zip` | ✅ 正常 | 內附 CUDA runtime DLL（cublas64_13, cublasLt64_13, cudart64_13），`--list-devices` 正確列出 RTX 3060 |

**教訓：** 若僅安裝 NVIDIA Driver（含 CUDA UMD）但無 CUDA Toolkit，
`bin-win-cuda` 版本無法載入 CUDA runtime，會無聲降級為 CPU-only。
應使用 `cudart-llama-bin-win-cuda` 版本。

### CUDA 偵測輸出（正確時）

```
Available devices:
  CUDA0: NVIDIA GeForce RTX 3060 (12287 MiB, 11255 MiB free)
```

### 編譯參數（官方預構建）

經由 `--version` 及 `system_info` log 推測：

```
CUDA : ARCHS = 750,800,860,890,900,1200,1210
USE_GRAPHS = 1
PEER_MAX_BATCH_SIZE = 128
```

## 測試工具

| 工具 | 用途 |
|:----|:----|
| PowerShell `Invoke-RestMethod` | 發送 /completion 請求 |
| curl.exe | 備用 HTTP 請求 |
| nvidia-smi | VRAM 監控 |
| llama-server.exe `--log-file` | 紀錄詳細 log |
| Windows Performance Counters | GPU 專屬記憶體用量 |
