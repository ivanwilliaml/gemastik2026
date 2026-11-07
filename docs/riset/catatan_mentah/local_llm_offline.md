# Local offline LLMs for Indonesian text on GTX 1650 4 GB / 8 GB RAM (checked 5 Nov 2026)

Evidence labels used below: **VERIFIED** means the repo and file sizes were read from the Hugging Face or GitHub API during this session. **MEASURED** means someone reported the number from a run, though not necessarily on this exact laptop. **ESTIMATE** means arithmetic from those numbers, not a run on this hardware. Nothing here was run on the user's laptop.

## What is the shortlist of exact GGUF files that fit this hardware?

### Takeaway
Download `gemma-4-E2B-it-Q4_K_M.gguf` for Indonesian labelling, because the E2B class is the only sub-10B class with a strong Indonesian score on SEA-HELM, and add `qwen2.5-coder-3b-instruct-q4_k_m.gguf` for coding. Plan on one model loaded at a time.

### Cited Findings

Exact files, all VERIFIED:

| Priority | Use | Repo / file | Bytes | Indonesian evidence |
|---|---|---|---|---|
| 1 | Labelling, summaries, topic names | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/api/models/unsloth/gemma-4-E2B-it-GGUF/tree/main) / `gemma-4-E2B-it-Q4_K_M.gguf` | 3,106,738,272 | SEA-HELM ID 64.13 for the model (quant used by the leaderboard not stated) |
| 1b | Lower-memory fallback for #1 | same repo / `gemma-4-E2B-it-Q3_K_M.gguf` | 2,536,786,016 | Same base model; the lower quant has no separate score |
| 2 | Same job, comparison run on the day | [aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF](https://huggingface.co/api/models/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF/tree/main) / `Gemma-SEA-LION-v4.5-E2B-IT-Q4_K_M.gguf` | 3,427,879,360 | SEA-HELM ID 63.96, a tie with #1 |
| 3 | Coding chat | [Qwen/Qwen2.5-Coder-3B-Instruct-GGUF](https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/tree/main) / `qwen2.5-coder-3b-instruct-q4_k_m.gguf` | 2,104,932,800 | Not relevant |
| 4 | Autocomplete, or coding if #3 is too heavy | [bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF](https://huggingface.co/api/models/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF/tree/main) / `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` | 986,048,800 | Not relevant |
| Optional | Fast fallback when memory is tight | [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/api/models/unsloth/Qwen3.5-2B-GGUF/tree/main) / `Qwen3.5-2B-Q4_K_M.gguf` | 1,280,835,840 | None found for this size |
| Optional | General chat and code | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/api/models/unsloth/Qwen3.5-4B-GGUF/tree/main) / `Qwen3.5-4B-Q4_K_M.gguf` | 2,740,937,888 | None found for this size |

Other sizes checked:
- `Qwen3.5-0.8B-Q4_K_M.gguf` is 532,517,120 bytes — [unsloth tree](https://huggingface.co/api/models/unsloth/Qwen3.5-0.8B-GGUF/tree/main)
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf` is 2,019,377,696 bytes — [bartowski tree](https://huggingface.co/api/models/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main)
- `Gemma-SEA-LION-v4-4B-VL-Q4_K_M.gguf` is 2,489,894,176 bytes — [aisingapore tree](https://huggingface.co/api/models/aisingapore/Gemma-SEA-LION-v4-4B-VL-GGUF/tree/main)

The E2B files (1, 1b and 2) exclude the vision projector, which is a separate `mmproj` file of about 986 MB — [Gemma 4 tree](https://huggingface.co/api/models/unsloth/gemma-4-E2B-it-GGUF/tree/main), [SEA-LION tree](https://huggingface.co/api/models/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF/tree/main)

### Inferences
- Gemma 4 E2B goes first rather than SEA-LION v4.5 E2B because the scores are effectively tied, its file is 321 MB smaller, and it is the one with a measured speed on a 1650-class card. SEA-LION is still worth carrying, because it was tuned on SEA instruction data and may handle local phrasing differently. Only a test on real rows can settle that.
- Text labelling should not need the `mmproj` file. The SEA-LION card's example command does include it, so test the command without it the day before.
- A 3.1 GB file on a card with about 3.9 GB usable looks tight once context and overhead are added. Expect some work to spill to the CPU (see the throughput section).

### Gaps
- No file on this list has been run on a non-Ti GTX 1650 with 8 GB RAM in any source I found. Real fit must be tested on the laptop.

## Which Indonesian/SEA-specific LLMs have small GGUFs, and which can run here?

### Takeaway
Only the E2B SEA-LION build is both current and a practical fit. Sahabat-AI 9B is too large at any quant worth using, and Komodo, Cendol and Merak are older models with no Indonesian score I could find that beats E2B.

### Cited Findings
- **SEA-LION v4.5 E2B:** the card describes "an ultra-efficient 2.3B effective parameter model (5.1B including embeddings)", MIT licensed, covering Indonesian among others, based on Gemma 4 E2B. The card itself lists no benchmarks — [model card](https://huggingface.co/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF). The score comes from the [SEA-HELM Indonesian page](https://leaderboard.sea-lion.ai/detailed/ID).
- **Other aisingapore GGUF repos:** Llama-SEA-LION-v3-8B-IT, Llama-SEA-LION-v3.5-8B-R, Gemma-SEA-LION-v3-9B-IT, Gemma-SEA-LION-v4-4B-VL, Apertus-SEA-LION-v4-8B-IT, and much larger 27B to 120B models — [HF API list](https://huggingface.co/api/models?author=aisingapore&search=gguf&limit=50)
- **Sahabat-AI official repos:** `GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct`, `-base`, and `GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct`, `-base`, with GGUF conversions only from third parties — [HF API search](https://huggingface.co/api/models?search=sahabatai&limit=50)
- **Sahabat-AI size:** the 9B i1 Q4_K_M is 5,761,058,944 bytes and the smallest build, IQ1_S, is 2,378,565,760 bytes — [mradermacher tree](https://huggingface.co/api/models/mradermacher/gemma2-9b-cpt-sahabatai-v1-instruct-i1-GGUF/tree/main)
- **Sahabat-AI score:** a model card reportedly lists Indonesian SEA-HELM averages of 33.67 for the 9B fine-tune, 27.75 for Qwen2.5-7B, and 26.44 for Gemma-2-9B. I saw this through a search summary, so it needs checking — [NVIDIA card](https://build.nvidia.com/gotocompany/gemma-2-9b-cpt-sahabatai-instruct/modelcard). An aggregator gives 61.169% on SEA-HELM and 62.6% on IndoMMLU, on an unclear scale — [PromptLayer](https://promptlayer.com/models/gemma2-9b-cpt-sahabatai-v1-instruct-gguf). The two sets cannot be compared, and neither can be compared directly with the 2026 leaderboard.
- **Komodo-7B:** `Yellow-AI-NLP/komodo-7b-base` exists, with community GGUFs `mradermacher/komodo-7b-chat-GGUF`, `kevinasyraf/komodo-7b-base-gguf` and `ukung/komodo-7b-base-GGUF` — [HF search](https://huggingface.co/api/models?search=komodo-7b&limit=15)
- **Cendol:** `indonlp/cendol-llama2-7b-chat` and `-inst` exist. The GGUFs I found are only for the 13B merged chat model, for example `lock-rr/cendol-llama2-13b-merged-chat-gguf` — [HF search](https://huggingface.co/api/models?search=cendol&limit=20)
- **Merak:** `Ichsan2895/Merak-7B-v3-GGUF`, `Ichsan2895/Merak-7B-v4-GGUF` and `asyafiqe/Merak-7B-v3-Mini-Orca-Indo-GGUF` exist — [HF search](https://huggingface.co/api/models?search=merak-7b&limit=15)

### Inferences
- A Q1-level quant of Sahabat-AI 9B would fit but is likely to lose too much quality, so it is not worth carrying.
- Llama 3 8B and 7B-class models at Q4 would run mostly split between GPU and CPU. With a current E2B available, I see no case for them unless a same-day test shows otherwise.

### Gaps
- **Llama-Sahabat-AI 70B:** not found in the first 50 search results. It would not fit this laptop in any case.
- **Komodo, Merak and Cendol:** I did not check their file sizes or find scores on the current SEA-HELM page extract.
- **Sahabat-AI on the current leaderboard:** the main leaderboard page did not mention it, so there is no like-for-like comparison with E2B.

## Which small multilingual models have good Indonesian evidence?

### Takeaway
Gemma 4 E2B is the only small general model with strong, current Indonesian evidence I could find. Small Qwen, Phi, SmolLM3, Llama 3.2 and Granite have no comparable Indonesian score in what I checked, so choosing them for labelling is a gamble unless tested on the day.

### Cited Findings
- **SEA-HELM Indonesian, as of 18 Sep 2026, sub-10B:** Gemma 4 E2B 64.13, SEA-LION v4.5 Gemma E2B 63.96, Tiny Aya Water 3B 33.23. I read the unlabelled first numeric column as the overall score, so that column reading is an assumption — [SEA-HELM ID](https://leaderboard.sea-lion.ai/detailed/ID)
- **Same page, top models:** Gemma 4 31B 79.75, Qwen 3.6 27B 77.96, Qwen 3.5 122B-A10B 77.93 — [SEA-HELM ID](https://leaderboard.sea-lion.ai/detailed/ID)
- **No small Qwen listed:** the main page's sub-10B entries showed only the Gemma E2B models, and its Qwen entries are 27B and above — [leaderboard](https://leaderboard.sea-lion.ai/)
- **IndoCareer, older reference point:** the paper reports average zero-shot accuracy of Gemma-2 at 2B 46.8%, 9B 60.5%, 27B 63.5%, as read from a search extract with the table header missing. Its authors note weakness in fields with strong local context — [arXiv 2409.08564v2](https://arxiv.org/html/2409.08564v2)
- **IndoMMLU, 2023:** GPT-3.5 reached 53.2% overall and mT0xxl 42.5% — [arXiv 2310.04928](https://arxiv.org/pdf/2310.04928). I found no 2026 results for small Qwen or Gemma on it.
- **Non-benchmark Indonesian use:** an Indonesian gambling-content project uses Gemma 4 E2B Q4_K_M and reports 90.87% accuracy on 1,468 images. This is an image task and only weak evidence for text, and the link is a GitHub mirror domain seen only through a search summary — [JudolGuard repo](https://github.blesdmm.dns-dynamic.net/mrayhanfadil/gemma)

### Inferences
- Large Qwen scores do not carry over to Qwen3.5-2B or 4B, since small models usually score lower than their large siblings. Tiny Aya's 33.23 shows that a small model from a good family can still score far lower.
- SEA-HELM measures general Indonesian ability, not MBG-related slang classification. Treat 64 as a reason to try E2B first, not a forecast of label accuracy.

### Gaps
- **Not found:** Indonesian scores for Qwen3.5-2B/4B, Phi-4-mini, SmolLM3, Llama 3.2 3B, Granite and Gemma 3n.
- **SmolLM3:** my search returned only older SmolLM-360M GGUF repos, so I could not confirm a SmolLM3 file from that query.
- **Leaderboard quantisation:** the scores probably do not reflect Q4 or Q3 quality loss, because the leaderboard does not state the quant it tested.

## Which Windows offline runtime is most robust on a GTX 1650, and does it support structured output?

### Takeaway
Use the llama.cpp prebuilt `llama-server`, because it loads the exact text-only GGUF, constrains output to a JSON schema, and needs no installer. Avoid the `gemma4:e2b` Ollama tag on 8 GB RAM, because one measured run reported 7.8 GB of model memory.

### Cited Findings
- **Latest build checked, b12119 (4 Nov 2026):** `llama-b12119-bin-win-cuda-12.4-x64.zip` at 287,037,856 bytes, `cudart-llama-bin-win-cuda-12.4-x64.zip` at 391,443,627 bytes, plus 13.4 CUDA variants, `llama-b12119-bin-win-vulkan-x64.zip` at 37,684,344 bytes, and `llama-b12119-bin-win-cpu-x64.zip` at 24,730,637 bytes — [GitHub releases API](https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5)
- **Trap:** the release marked latest, v0.9.0, lists only a 7-byte `nightly-tag.txt`, so Windows binaries must be taken from a `b####` build — [GitHub latest release API](https://api.github.com/repos/ggml-org/llama.cpp/releases/latest)
- **Structured output:** `/v1/chat/completions` accepts `response_format` with `json_object` or a `json_schema` type and schema. `/completion` accepts `json_schema` or a BNF `grammar` — [llama-server README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md)
- **Memory-related flags:** `-ngl` defaults to `auto`, `--fit` defaults to `on`, and `-ctk`/`-ctv` set KV cache types. `--no-mmap` was not listed in the part of the README I read, which instead shows `-lm, --load-mode` — [llama-server README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md)
- **Disabling thinking:** the SEA-LION card uses `"enable_thinking": false` in `chat_template_kwargs` and shows `llama-server` as its example runtime — [model card](https://huggingface.co/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF)
- **Ollama offline:** a standalone `ollama-windows-amd64.zip` exists, models live in `%HOMEPATH%\.ollama`, and `OLLAMA_MODELS` changes the location — [Ollama Windows docs](https://docs.ollama.com/windows). A third-party guide says copying `blobs` and `manifests` moves models between machines, and warns not to copy a manifest without its blobs — [LocalAIMaster](https://localaimaster.com/blog/transfer-ollama-models-offline)
- **Ollama tag sizes:** `gemma4:e2b` is listed at 4.6 GB to 7.5 GB, `gemma4:e2b-it-q4_K_M` at 4.6 GB, and `gemma4:e2b-it-qat` at 4.3 GB — [Ollama tags](https://ollama.com/library/gemma4/tags)
- **Ollama memory on 4 GB:** on a GTX 1650 Ti Mobile, `ollama ps` showed the model at 7.8 GB, split 74%/26% CPU/GPU — [DEV Community](https://dev.to/mrviduus/i-put-ollama-on-a-4-gb-mobile-gpu-and-got-25-heres-the-vram-math-3mhk)

Starting command, not run on this hardware (flags except `-c`, `-t` and `--port` are from the README; those three are from memory, so check `llama-server --help`):
```
llama-server -m gemma-4-E2B-it-Q4_K_M.gguf -c 2048 -np 1 -t 6 --port 8080
```
Request body fields for labelling:
```json
{"temperature": 0, "max_tokens": 20,
 "chat_template_kwargs": {"enable_thinking": false},
 "response_format": {"type": "json_schema", "json_schema": {"schema":
   {"type": "object", "properties": {"label": {"enum": ["pos", "neg", "net"]}}, "required": ["label"]}}}}
```

### Inferences
- **CUDA version:** carry 12.4 first, as the more conservative choice for a 2019-era Turing card and driver, and the Vulkan zip as a fallback. Test both before going offline. From memory, the two CUDA zips are extracted into one folder; confirm this on the laptop.
- **Python access:** calling `llama-server` over local HTTP with `requests` avoids a `llama-cpp-python` CUDA wheel build, and also keeps the LLM in its own process, so killing it frees VRAM and RAM immediately.
- **Ollama:** it remains a usable fallback with a smaller tag or a local GGUF, but it adds a layer that makes memory placement harder to see.

### Gaps
- **LM Studio and `llama-cpp-python`:** I did not research offline installers, wheels, or their GTX 1650 behaviour, so I cannot rank them from evidence.
- **Windows-specific evidence:** the GTX 1650 reports I found are not on Windows 10. The one detailed run was on Ubuntu with Docker.
- **Prompt caching:** I did not read confirmation that llama-server reuses the prompt prefix by default, and the throughput estimates depend on it. Test it by timing the second request.

## How many rows can be labelled, and can it run while the GPU trains?

### Takeaway
ESTIMATE: plan on 500 to 1,000 rows in 30 minutes and 1,000 to 2,000 in 60 minutes, with a well-tuned best case near 1,800 and 3,600. Do not plan to run it during GPU training on 4 GB, and use the CPU only if labels must continue during training.

### Cited Findings
- **Closest MEASURED run:** GTX 1650 Ti Mobile, Ryzen 5 4600H, Ollama 0.23.1, `gemma4:e2b`, 4096 context, about 60 output tokens per call. CPU only gave 17 tok/s and 5,390 ms total per call. GPU hybrid gave 39 tok/s and 2,174 ms per call. 35 of 36 layers were on the GPU, the output layer stayed on the CPU, and `nvidia-smi` showed 1,998 MiB used and 1,909 MiB free — [DEV Community](https://dev.to/mrviduus/i-put-ollama-on-a-4-gb-mobile-gpu-and-got-25-heres-the-vram-math-3mhk)
- **Faster report:** about 59 tok/s for Gemma 4 E2B Q4_K_M on a GTX 1650 Ti 4 GB under llama.cpp, with tuning giving 2.24× on Qwen3.5-4B and 2.07× on Qwen3-4B. I saw these only through a search summary and a partial excerpt, as the full article returned 403 — [Towards AI](https://pub.towardsai.net/how-i-made-local-llms-up-to-2-24-faster-on-a-4gb-nvidia-gpu-e0c02f2b1821)
- **Bandwidth proxy:** a Quadro T1000 4 GB reached 27.82 tok/s generation and 79.44 tok/s prompt processing on Llama 2 7B Q4_0. The guide calls its bandwidth the same as the GDDR5 GTX 1650, and has no measured 1650 number — [BestLLMFor](https://bestllmfor.com/guide/llm-gtx-1650-1660/)
- **Partial offload:** a 4 GB RTX 3050 report says speed drops sharply when layers spill to system RAM, and recommends a smaller quant — [DEV Community](https://dev.to/shogun444/the-brutal-reality-of-running-gemma-4-locally-29e7)
- **Calculator figures to discount:** Will It Run AI gives about 14 tok/s for Llama 3.2 1B and CanIRunAI gives 25.0 for Gemma 4 E2B. Neither states a measured basis, and the first conflicts with the measured runs above — [Will It Run AI](https://willitrunai.com/es/can-run/llama-3.2-1b-on-gtx-1650-4gb), [CanIRunAI](https://canirunai.vercel.app/gpu/nvidia-gtx-1650)

### Inferences
All rows below are ESTIMATES for sequential single-row calls. Rows/time = time budget divided by seconds per row.

| Scenario | Basis | Sec/row | 30 min | 60 min |
|---|---|---|---|---|
| Measured call as-is | 2.174 s per call, 60 output tokens | 2.17 | ~830 | ~1,660 |
| Short JSON output, prompt prefix cached | 60 new tokens at 79.44 tok/s plus 10 output tokens at 39 tok/s | 1.01 | ~1,780 | ~3,560 |
| Short JSON output, no prefix cache | 400 prompt tokens at 79.44 tok/s plus 10 output tokens at 39 tok/s | 5.29 | ~340 | ~680 |
| CPU only, while the GPU trains | 5.39 s per call | 5.39 | ~330 | ~670 |

- The planning range sits below the middle rows because the user's card is a non-Ti 1650, the 79.44 tok/s prompt speed comes from a different model, and a laptop may throttle over an hour. All the row and token counts above are assumptions, so longer texts will lower them.
- At 1 s per row, all 500,000 rows would take about 139 hours, so the LLM can only label a sample.
- **VRAM contention:** the measured run left about 1.9 GB free, but that was with most of the model's memory held in system RAM. Sharing the card with a training job risks out-of-memory errors in both, and 8 GB RAM is the tighter limit. Run them in turn, and stop the server before training.
- **Test first:** 200 real rows on the day will replace this table within about 10 minutes.

### Gaps
- No measured prompt-processing speed for E2B on a 1650-class card.
- No measured E2B or Qwen3.5 run on the exact GTX 1650 non-Ti with 8 GB RAM. Whether the user's card is GDDR5 or GDDR6 also matters, per [BestLLMFor](https://bestllmfor.com/guide/llm-gtx-1650-1660/).
- No measured run of a GPU training job and an LLM server sharing a 4 GB card.

## What is the best offline coding assistant on 4 GB, and how does it connect to VS Code?

### Takeaway
Use Qwen2.5-Coder 3B Q4_K_M for chat, 1.5B if memory is tight, connected through Continue v1.2.23's Windows VSIX. Treat this as a useful extra: a folder of saved docs and snippets matters more, and the coding model competes with the rest of the pipeline for memory.

### Cited Findings
- **VSIX assets:** Continue's `v1.2.23-vscode` has a `win32-x64` VSIX of 74,470,680 bytes, and `v1.3.40-vscode` has one of 74,470,692 bytes. The names follow `continue-<platform>-<version>.vsix`. `v2.0.0-vscode` and `v2.1.0-vscode` list no VSIX assets — [GitHub releases API](https://api.github.com/repos/continuedev/continue/releases?per_page=8)
- **Version caveat:** v1.2.24 is a re-cut of v1.2.23 without assets, and v1.3.40 is a pre-release described as a final-release cleanup — [GitHub releases API](https://api.github.com/repos/continuedev/continue/releases?per_page=8)
- **Install route:** a third-party README shows `code --install-extension continue-1.3.39.vsix` — [Hugging Face README](https://huggingface.co/codekingpro/continue-vscode). Official docs I found describe only the online Marketplace flow — [Continue docs](https://docs.continue.dev/ide-extensions/install)
- **Config:** Continue uses YAML, and `config.json` is the deprecated route — [Continue autocomplete docs](https://docs.continue.dev/customize/deep-dives/autocomplete). A model entry can use `provider: llama.cpp` with `roles: [autocomplete]`, and one write-up advises keeping the chat endpoint separate from autocomplete — [DEV Community](https://dev.to/vast-cow/getting-fill-in-the-middle-autocomplete-working-in-vs-code-continue-with-llamacpp-552h)
- **FIM endpoint:** `llama-server` offers `/infill` with `input_prefix` and `input_suffix` — [llama-server README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md)

### Inferences
- Running two servers at once (chat plus autocomplete) is unrealistic alongside data work on this laptop. Choose chat only; autocomplete adds constant load for little gain in a 5-hour sprint.
- The model choice rests on purpose-built design and fit, not on a benchmark. Qwen3.5-4B is a reasonable alternative to try the day before on the same pandas and sklearn prompts.
- Pandas, sklearn and PyTorch knowledge in a 2024-era model may lag current APIs, so check the errors it causes rather than trusting it.
- Test Continue with Wi-Fi off the day before. If it fails, fall back to a plain script or any chat client against `llama-server`.

### Gaps
- No benchmark found comparing Qwen2.5-Coder, Qwen3.5-4B and Phi-4-mini on pandas, sklearn or PyTorch tasks. I did not verify a Phi-4-mini GGUF.
- Not confirmed whether Continue needs network access at startup, or which minimum VS Code version v1.2.23 requires.
- Continue's future support is unclear from release notes alone, which is another reason to pin one known VSIX.

## What are the main pitfalls?

### Takeaway
RAM is the most likely point of failure, ahead of VRAM. The most likely quality failure is trusting LLM labels on slang without a hand-labelled check.

### Cited Findings
- **RAM:** the 7.8 GB figure above is close to the laptop's entire 8 GB — [DEV Community](https://dev.to/mrviduus/i-put-ollama-on-a-4-gb-mobile-gpu-and-got-25-heres-the-vram-math-3mhk)
- **Hidden RAM use:** a Reddit thread says llama.cpp can keep cache in system RAM even with all layers on the GPU, and suggests KV cache quantisation and `--cache-ram`. These are unverified commenter suggestions, seen only through a search summary, and this link is my best match among several returned snapshots, so the exact thread is unconfirmed — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1ssbzfc/snapshots/2026-04-25T00%3A46%3A56.728262Z)
- **Context cost:** a VRAM guide notes that long contexts need substantially more VRAM on Gemma 4 — [knightli guide](https://knightli.com/zh-tw/2026/05/01/gemma-4-local-vram-quantization-table/)
- **Misleading tools:** `ollama ps` percentages are a memory split, not a layer count, so check load logs — [DEV Community](https://dev.to/mrviduus/i-put-ollama-on-a-4-gb-mobile-gpu-and-got-25-heres-the-vram-math-3mhk)
- **Heat:** the same run measured about 10 °C lower CPU temperature in GPU hybrid mode under burst load. I found no sustained-load throttling data for this class of laptop — [DEV Community](https://dev.to/mrviduus/i-put-ollama-on-a-4-gb-mobile-gpu-and-got-25-heres-the-vram-math-3mhk)
- **Slang:** an Indonesian-Sundanese IndoBERT study reached 81% accuracy and flagged abbreviated words as a possible cause of errors — [Telkom University](https://repository.telkomuniversity.ac.id/pustaka/196843/sentiment-analysis-on-indonesian-sundanese-code-mixed-data.html)
- **Weak-supervision baseline:** a lexicon-based Indonesian study concluded that lexicon-driven weak supervision works for code-mixed posts — [UGM repository](https://etd.repository.ugm.ac.id/penelitian/detail/272760). A labelling-function study reached only 59.79% accuracy on sentiment — [arXiv 2310.11258](https://arxiv.org/pdf/2310.11258)

### Inferences
- Do not hold a 500,000-row DataFrame, an LLM, and a training job in memory at once. Run them as separate processes in turn, and write labels to disk every few rows so a crash loses little.
- Keep context small (about 2,048) since row text is short. Long context costs VRAM with no benefit for labelling.
- A CUDA clash with PyTorch is less likely when `llama-server` runs from its own folder with its own DLLs, as a separate process. Do not add that folder to the system PATH. This is reasoning from how Windows loads DLLs, not a tested result.
- Hand-label 100 to 200 rows first, measure macro-F1 for the LLM against them, and use LLM labels as one noisy source rather than ground truth. Include a "tidak jelas" class so the model is not forced to guess.
- JSON schema guarantees the format, not the correctness. A forced enum can hide uncertainty.

### Gaps
- No study found on small local LLM label quality for Indonesian slang specifically.
- No sustained-load thermal data for an i7-9750H and GTX 1650 running llama.cpp.

## Is a local LLM worth it compared with a zero-shot/NLI encoder, and what is the minimal setup?

### Takeaway
Yes, as a small secondary tool, chiefly for naming clusters and for short labelling runs of about 1,000 rows on unclear classes. It should not replace the encoder pipeline. Keep the LLM setup under about 5 GB of files and limit it to one fixed time slot.

### Cited Findings
- **Closest comparison:** I found no direct study of LLM versus zero-shot NLI labelling for Indonesian slang. The nearest is a Hinglish misogyny study on just 100 comments, where zero-shot BART-large and few-shot GPT-3 performed best, which is too small and too different to decide this — [arXiv 2403.02121](https://arxiv.org/pdf/2403.02121)
- **Speed limit:** the measured 2.17 s per call makes labelling all rows impossible in 5 hours — [DEV Community](https://dev.to/mrviduus/i-put-ollama-on-a-4-gb-mobile-gpu-and-got-25-heres-the-vram-math-3mhk)

### Inferences

Use by task:

| Task | Local LLM? | Why |
|---|---|---|
| Summarising clusters, naming topics | Yes, the best use | A few dozen calls, a few minutes in total |
| Labelling a sample for weak supervision | Yes, if time allows | About 1,000 rows, then train a fast classifier on all rows |
| Zero-shot over all 500,000 rows | No | Needs about 139 hours even at 1 s per row |
| Augmentation | Rarely | Costs labelling time, and generated text may blur slang and labels |
| Coding help | Yes, if it passes the day-before test | Useful when stuck, but do not wait on it for simple tasks |

Minimal download list, all before going offline:
1. `llama-b12119-bin-win-cuda-12.4-x64.zip` and `cudart-llama-bin-win-cuda-12.4-x64.zip`, or the newest build available that day, plus the Vulkan zip as a fallback.
2. `gemma-4-E2B-it-Q4_K_M.gguf`, plus Q3_K_M only if the Q4 test fails on memory.
3. `qwen2.5-coder-3b-instruct-q4_k_m.gguf`.
4. `Gemma-SEA-LION-v4.5-E2B-IT-Q4_K_M.gguf` if disk and download time allow.
5. `continue-win32-x64-1.2.23.vsix` (name inferred from the pattern; check the release page).

Day-before test, with Wi-Fi off:
1. Start `llama-server` and read the log for layers on GPU.
2. Label 200 Indonesian slang-heavy rows with both E2B models, and record seconds per row and peak RAM.
3. Check agreement with hand labels against the NLI encoder on the same rows. Keep the LLM for labelling only if it clearly beats the encoder.
4. Install the VSIX and ask one pandas question.

If the LLM takes more than about 30 minutes to get working on the day, drop it, since the encoder pipeline does not depend on it.

### Gaps
- No throughput or accuracy figures for an NLI encoder on this hardware were gathered in this workstream, so the comparison is about feasibility. The accuracy comparison has to come from the user's own 200-row test.
