# 04 — 測試指令

> 所有測試使用 `llama-server.exe` 作為推論引擎，透過 HTTP `/completion` API 取得 timing。
> 未特別標註的皆執行於 `C:\Users\castlen3\.gemini\antigravity\scratch\llama_run` 目錄下。

## 測試命名規則

- `n01`~`nNN`：RTX 3060 一般版模型
- `m01`~`mNN`：RTX 3060 MTP 版模型
- `fake_gpu` 標籤表示該階段因缺失 CUDA runtime 實為 CPU-only
- `real_gpu` 標籤表示已正確使用 CUDA 加速

## 階段 B：RTX 3060 bin-only（缺失 CUDA runtime, CPU-only）

> **⚠️ 重要：** 此階段所有測試因缺少 `cublas64_13.dll` 等 CUDA runtime，
> `ggml-cuda.dll` 無法載入 GPU，實際全部使用 CPU 推論。
> 數據僅供對比，不代表 RTX 3060 真實效能。

### n01~n14 - 一般版 n_cpu_moe sweep（short prompt 16 tok）

```bash
llama-server.exe -m "Qwen3.6-35B-A3B-Q4_K_M.gguf" -t 6 --port 8080 -ngl 99 --n-cpu-moe {moe} --no-mmap --cache-type-k q8_0 --cache-type-v q8_0 -c 32768 -np 1 --reasoning off
```
- 測試 moe：28, 24, 20, 16, 12, 8, 4, 0
- 結果：n_cpu_moe=20 → 10.68 tok/s（最佳）
- 此階段使用 q8_0 KV cache 與 32K context

### m01~m06 - MTP draft sweep（short prompt 16 tok）(CPU-only)

```bash
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -t 6 --port 8080 -ngl 99 --n-cpu-moe 20 --no-mmap --cache-type-k q8_0 --cache-type-v q8_0 -c 32768 -np 1 --reasoning off --spec-type draft-mtp --spec-draft-model {same_file} --spec-draft-n-max {draft}
```
- 測試 draft：1, 2, 3, 4, 5, 7
- 結果：draft=5 → 11.94 tok/s（最佳）
- ⚠️ 此階段使用 `--spec-draft-model` 指向同一個檔案（非必要參數）

### m07~m21 - MTP draft × threads sweep（long prompt 307 tok）(CPU-only)

```bash
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -t {threads} --port 8080 -ngl 99 --n-cpu-moe 20 --no-mmap --cache-type-k q8_0 --cache-type-v q8_0 -c 32768 -np 1 --reasoning off --spec-type draft-mtp --spec-draft-model {same_file} --spec-draft-n-max {draft}
```
- 測試 draft：2, 4, 5 × threads：4, 6, 8, 10, 12
- 結果：draft=5 + threads=8 → 14.99 tok/s（此階段最佳）
- 備註：長文本因預測性高導致 MTP accept rate 虛高

---

## 階段 C：RTX 3060 cudart（真實 GPU 加速）

> 換上 `cudart-llama-bin-win-cuda-13.3-x64.zip` 後，CUDA 正確啟用。

### n15~n26 - 一般版 n_cpu_moe sweep（真實 GPU）(short 16 tok + long 307 tok)

```bash
llama-server.exe -m "Qwen3.6-35B-A3B-Q4_K_M.gguf" -t 8 --port 8080 -ngl 99 --n-cpu-moe {moe} --no-mmap --cache-type-k q8_0 --cache-type-v q8_0 -c 32768 -np 1 --reasoning off
```
- 測試 moe：20, 16, 12, 8, 4, 0
- 結果（long prompt）：n_cpu_moe=20 → **27.73 tok/s** 🏆

### m22~m27 - MTP ngl sweep（真實 GPU）

```bash
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -t 8 --port 8080 -ngl {ngl} --n-cpu-moe 41 --no-mmap --cache-type-k q4_0 --cache-type-v q4_0 -c 4096 -np 1 --reasoning off --spec-type draft-mtp --spec-draft-n-max 2 -fit off
```
- 測試 ngl：12, 16, 20, 24, 28, 32, 36, 40
- 結果：ngl=40 + draft=2 → 18.29 tok/s
- 所有 ngl 皆成功，無 crash

### m28~m30 - MTP draft sweep at ngl=40（真實 GPU）

```bash
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -t 8 --port 8080 -ngl 40 --n-cpu-moe 41 --no-mmap --cache-type-k q4_0 --cache-type-v q4_0 -c 4096 -np 1 --reasoning off --spec-type draft-mtp --spec-draft-n-max {draft} -fit off
```
- 測試 draft：1, 2, 3
- 結果：draft=3 → **20.80 tok/s** 🏆

### m31~m36 - MTP n_cpu_moe sweep at ngl=40, draft=3（真實 GPU）

```bash
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -t 8 --port 8080 -ngl 40 --n-cpu-moe {moe} --no-mmap --cache-type-k q4_0 --cache-type-v q4_0 -c 4096 -np 1 --reasoning off --spec-type draft-mtp --spec-draft-n-max 3 -fit off
```
- 測試 moe：41, 36, 32, 28, 24, 20
- 結果：n_cpu_moe=24 → **25.99 tok/s** 🏆
- n_cpu_moe=20 → VRAM 12GB 滿載，decode 降至 16.26 tok/s（thrashing）

### Smoke Test：MTP 模型載入驗證

```bash
# Test 1: MTP GGUF + CPU-only + 無 MTP
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -ngl 0 -c 512 --port 8080 -t 4 --no-mmap -np 1
# ✅ 成功載入

# Test 2: MTP GGUF + CPU-only + MTP draft=1
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -ngl 0 -c 512 --port 8080 -t 4 --no-mmap -np 1 --reasoning off --spec-type draft-mtp --spec-draft-n-max 1
# ✅ 成功載入，log 顯示 MTP context estimated 890.85 MiB

# Test 3: MTP GGUF + GPU 12 layers + MTP draft=2
llama-server.exe -m "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -ngl 12 --n-cpu-moe 41 -c 4096 --port 8080 -t 8 --no-mmap --cache-type-k q4_0 --cache-type-v q4_0 -np 1 --reasoning off --spec-type draft-mtp --spec-draft-n-max 2 -fit off
# ✅ 成功載入，21.8 GB RAM + GPU 混合
```
