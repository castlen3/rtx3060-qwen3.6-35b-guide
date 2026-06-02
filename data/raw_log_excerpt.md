# 原始 Log 摘要

> 保留原始文字，僅去除 ANSI escape codes 以利閱讀。
> 整理自 2026-06-02 測試過程中的 `--log-file` 輸出與終端機紀錄。

---

## 1. 一般版最佳組（n21）— 完整啟動 + Completion

### 啟動 Log

此測試使用 `cudart-llama-bin-win-cuda-13.3-x64` 版本，正確啟用 CUDA。

**Command:**
```
llama-server.exe -m "Qwen3.6-35B-A3B-Q4_K_M.gguf" -t 8 --port 8080
  -ngl 99 --n-cpu-moe 20 --no-mmap
  --cache-type-k q8_0 --cache-type-v q8_0 -c 32768 -np 1 --reasoning off
```

**啟動 Log（關鍵行）：**
```
I common_params_print_info: build 9444 (6f165c1c6) with Clang 19.1.5 for Windows x86_64
I device_info:
I   - CUDA0   : NVIDIA GeForce RTX 3060 (12287 MiB, 11255 MiB free)
I   - CPU     : Intel(R) Xeon(R) CPU E5-2666 v3 @ 2.90GHz (65376 MiB, 58407 MiB free)
I system_info: n_threads = 8 (n_threads_batch = 8) / 20 |
  CUDA : ARCHS = 750,800,860,890,900,1200,1210 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128
I srv    load_model: loading model 'Qwen3.6-35B-A3B-Q4_K_M.gguf'
D llama_model_loader: loaded meta data with 41 key-value pairs and 733 tensors
D llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3060) (0000:03:00.0) - 11255 MiB free
...
W llama_context: n_ctx_seq (32768) < n_ctx_train (262144) -- the full capacity of the model will not be utilized
I common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
...
I srv  llama_server: model loaded
I srv  llama_server: server is listening on http://127.0.0.1:8080
```

### Completion Response（n21, long prompt 307 tok）

```json
{
  "tokens_predicted": 200,
  "tokens_evaluated": 307,
  "timings": {
    "prompt_n": 307,
    "prompt_ms": 1180.0,
    "prompt_per_token_ms": 3.84,
    "prompt_per_second": 260.20,
    "predicted_n": 200,
    "predicted_ms": 7210.0,
    "predicted_per_token_ms": 36.05,
    "predicted_per_second": 27.73
  }
}
```

> `prompt_per_second=260.20`, `predicted_per_second=27.73`
> 為本次測試 decode 最高紀錄。

---

## 2. MTP 最佳組（m38）— 完整啟動 + Completion

### 啟動 Log

**Command:**
```
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -t 8 --port 8080
  -ngl 40 --n-cpu-moe 24 --no-mmap
  --cache-type-k q4_0 --cache-type-v q4_0 -c 4096 -np 1 --reasoning off
  --spec-type draft-mtp --spec-draft-n-max 3 -fit off
```

**啟動 Log（完整，取自 smoke3_log.txt 同類型啟動）：**
```
I log_info: verbosity = 3
I device_info:
I   - CUDA0   : NVIDIA GeForce RTX 3060 (12287 MiB, 11255 MiB free)
I   - CPU     : Intel(R) Xeon(R) CPU E5-2666 v3 @ 2.90GHz (65376 MiB, 58210 MiB free)
I system_info: n_threads = 8 (n_threads_batch = 8) / 20 |
  CUDA : ARCHS = 750,800,860,890,900,1200,1210 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128
I srv  llama_server: loading model
I srv    load_model: loading model 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
D llama_model_loader: loaded meta data with 55 key-value pairs and 753 tensors
  (version GGUF V3 (latest))
...
I srv    load_model: [spec] estimated memory usage of MTP context is 890.85 MiB
I common_init_result: fitting params to device memory ...
W llama_context: n_ctx_seq (4096) < n_ctx_train (262144)
I common_init_from_params: warming up the model with an empty run - please wait ...
I srv    load_model: creating MTP draft context against the target model
  'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
I common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'
I common_speculative_impl_draft_mtp: - n_max=3, n_min=0, p_min=0.00,
  n_embd=2048, backend_sampling=1
I common_speculative_impl_draft_mtp: - gpu_layers=-1, cache_k=f16, cache_v=f16,
  ctx_tgt=yes, ctx_dft=yes, devices=[default]
I srv    load_model: speculative decoding context initialized
I srv    load_model: prompt cache is enabled, size limit: 8192 MiB
...
I srv  llama_server: model loaded
I srv  llama_server: server is listening on http://127.0.0.1:8080
```

### Completion Response（m38, long prompt 307 tok, draft=3）

```json
{
  "tokens_predicted": 256,
  "tokens_evaluated": 307,
  "timings": {
    "prompt_n": 307,
    "prompt_ms": 1754.0,
    "prompt_per_token_ms": 5.71,
    "prompt_per_second": 175.01,
    "predicted_n": 256,
    "predicted_ms": 9846.0,
    "predicted_per_token_ms": 38.46,
    "predicted_per_second": 25.99
  }
}
```

> `predicted_per_second=25.99`
> 為 MTP 版在 VRAM 12GB 極限下的最佳成績。

---

## 3. CPU-only 代表組（n03 / m05）

### 判斷依據：無 CUDA device

```
$ ./llama-server.exe --list-devices
Available devices:
                                             ← 完全空白！無任何 CUDA 裝置
```

此狀態下即使指定 `-ngl 99`，所有層仍會在 CPU 上執行。

### 對照組：正確的 CUDA 裝置列舉

```
$ ./llama-server.exe --list-devices
Available devices:
  CUDA0: NVIDIA GeForce RTX 3060 (12287 MiB, 11255 MiB free)
```

### VRAM 判斷法

```
# CPU-only（bin-only 版本）：VRAM 僅 ~932 MB（桌面程式佔用）
$ nvidia-smi --query-gpu=memory.used --format=csv,noheader
932 MiB

# 真實 GPU（cudart 版本）：VRAM 上升至 11,600+ MB
$ nvidia-smi --query-gpu=memory.used --format=csv,noheader
11633 MiB
```

### n03 典型 Completion（CPU-only, 16 tok prompt）

```json
{
  "tokens_predicted": 200,
  "tokens_evaluated": 16,
  "timings": {
    "prompt_n": 16,
    "prompt_ms": 569.0,
    "prompt_per_token_ms": 35.56,
    "prompt_per_second": 28.13,
    "predicted_n": 200,
    "predicted_ms": 18724.0,
    "predicted_per_token_ms": 93.62,
    "predicted_per_second": 10.68
  }
}
```

> `predicted_per_second=10.68` — 當時以為是 GPU 成績，
> 實際上是 CPU-only（因缺少 CUDA runtime DLL）。

---

## 4. `--list-devices` 為空輸出

### 情境：使用 `llama-b9444-bin-win-cuda-13.3-x64.zip`（bin-only）

```
$ ./llama-server.exe --list-devices
Available devices:                           ← 完全空白

$ nvidia-smi --query-gpu=memory.used --format=csv,noheader
932 MiB                                       ← 無 llama-server process
                                              ← 僅桌面程式佔用
```

**原因：** `ggml-cuda.dll` 無法載入 CUDA runtime，
因缺少 `cublas64_13.dll`、`cublasLt64_13.dll`、`cudart64_13.dll`。

**解決：** 改下載 `cudart-llama-bin-win-cuda-13.3-x64.zip`，
內附上述 CUDA runtime DLLs。

---

## 5. `--spec-draft-model` 導致錯誤的完整錯誤 Log

### 情境：對 self-MTP 模型使用 `--spec-draft-model [same_file]`

**Command:**
```
llama-server.exe
  -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
  --spec-type draft-mtp
  --spec-draft-model "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"    ← 錯誤：self-MTP 不需要
  --spec-draft-n-max 3
```

**完整錯誤 Log（取自 mtp_error_log.txt，行 3565-3571）：**

```
0.19.473.877 E llama_model_load: error loading model: invalid vector subscript
0.19.473.892 E llama_model_load_from_file_impl: failed to load model
0.19.473.910 E srv    load_model: failed to load draft model,
    'C:\Users\castlen3\.lmstudio\models\unsloth\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
0.19.473.934 I srv    operator(): operator(): cleaning up before exit...
0.19.475.528 E srv  llama_server: exiting due to model loading error
0.19.475.628 D ~llama_context:      CUDA0 compute buffer size is 497.0021 MiB,
    matches expectation of 497.0021 MiB
0.19.475.631 D ~llama_context:  CUDA_Host compute buffer size is  12.2852 MiB,
    matches expectation of  12.2852 MiB
```

**錯誤前的載入歷程（同檔案前段）：**

```
0.00.340.257 I srv    load_model: loading model
  'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
0.00.547.278 D llama_model_loader: loaded meta data with 55 key-value pairs
  and 753 tensors from ... (version GGUF V3 (latest))
...
（55 組 KV pairs 正常載入）
...
0.00.727.944 D print_info: file size   = 21.10 GiB (5.10 BPW)
0.00.728.053 D llama_prepare_model_devices: using device CUDA0
  (NVIDIA GeForce RTX 3060) (0000:03:00.0) - 11255 MiB free
...
（所有 print_info 正常，包含 n_layer=41 的 MTP 特定資訊）
...
0.02.069.592 W common_fit_params: failed to fit params to free device memory:
  n_gpu_layers already set by user to 99, abort
...
0.19.393.437 I load_tensors: loading model tensors, this can take a while...
0.19.473.877 E llama_model_load: error loading model: invalid vector subscript
```

**錯誤解釋：**

`invalid vector subscript` 是 C++ `std::out_of_range` 異常。
此錯誤發生在 `load_tensors` 階段，表示 llama.cpp 嘗試載入 tensor 時
存取了一個不存在的索引。

**原因推測：**
- self-MTP 模型的 GGUF 中已有 MTP head tensor，
  不需透過 `--spec-draft-model` 再次指定同一個檔案。
- 當 `--spec-draft-model` 指向主模型時，llama.cpp 嘗試第二次載入
  並解析 tensor，但某些 tensor 名稱或索引與第一次載入衝突。
- 移除 `--spec-draft-model` 後，僅用 `--spec-type draft-mtp` 即可正常載入。

**修正後的 Command（正確）：**
```
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
  --spec-type draft-mtp --spec-draft-n-max 3
  ← 不需 --spec-draft-model
```
