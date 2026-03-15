# Dual RTX 5090 Rigs for Local LLM Inference

## Overview of the RTX 5090 graphics card

The GeForce RTX 5090 is NVIDIA’s flagship consumer GPU built on the **Blackwell architecture**.  Key specifications include 21 760 CUDA cores, 680 fifth‑generation Tensor cores and 170 fourth‑generation RT cores.  The card uses 32 GB of **GDDR7 memory** on a wide **512‑bit bus**, providing approximately **1.79 TB/s** of memory bandwidth【193924165157245†L1015-L1075】【777657413746168†L88-L116】.  The base and boost clocks are ~2.01 GHz and 2.41 GHz respectively【193924165157245†L1015-L1052】.  With a **575 W TDP**, NVIDIA recommends at least a **1 kW system power supply**; the card requires either **4× PCIe 8‑pin** connectors or **1× 600 W PCIe Gen 5** cable【193924165157245†L1070-L1075】.  The RTX 5090 is designed for PCIe 5.0 and offers DisplayPort 2.1b outputs supporting up to **8K @ 165 Hz**【193924165157245†L1060-L1069】.

### AI and LLM performance

The Blackwell architecture adds new Tensor core instructions and **DLSS 4**.  According to benchmark reviews, the RTX 5090 is roughly **27–35 % faster** than the RTX 4090 but draws about **575 W** and often runs at **89–90 °C** under heavy load【907804445390455†L108-L123】.  The larger 32 GB VRAM provides ample head‑room for deep‑learning workloads; tests show the card is **40 % faster** than the RTX 4090 in FP16 AI inference and **12 % faster** in video processing【777657413746168†L134-L137】.  RunPod’s benchmarks on **Qwen2.5‑Coder‑7B** highlight the card’s exceptional throughput: at a 1024‑token batch with batch size 8, the RTX 5090 produced **5 841 tokens/sec**, exceeding an A100 80 GB by 2.6×【413653357748193†L65-L88】.  For smaller models (e.g., Qwen2‑0.5 B or Phi‑3‑Mini‑4k‑Instruct), throughput can exceed **65 000 tokens/sec**【413653357748193†L112-L120】.

### Limitations

* **NVLink / SLI** is **not supported** on the RTX 5090【193924165157245†L1052-L1055】, so multi‑GPU communication must use PCIe and system memory.  NVIDIA’s consumer driver disables peer‑to‑peer (P2P) communication, meaning GPUs cannot directly access each other’s memory【224217239444475†L420-L431】.  Newer versions of NCCL provide work‑arounds, but latency remains higher than on professional GPUs with NVLink.
* Early units suffered from **melting connectors** because the card can draw near the 12V‑2×6 plug’s 600 W limit【777657413746168†L195-L207】.  Users should ensure a proper 600 W cable and adequate airflow.

## Building a dual‑GPU RTX 5090 rig

Running two RTX 5090 cards in one workstation can deliver datacenter‑class performance for local LLM inference, but it requires careful planning around **power, cooling and PCIe bandwidth**.

### Power and cooling requirements

* **Power draw**: A single card draws 575 W, so a dual‑GPU system plus a high‑core‑count CPU can exceed **1 500 W**【792305678010212†L205-L214】.  Puget Systems notes that when including the CPU, motherboard, RAM, drives and fans, total power often exceeds what typical **1 600 W PSUs** and **120 V circuits** can supply【792305678010212†L205-L217】.  Their dual‑5090 workstations use **2 800 W power supplies** that require **200–240 V circuits**【792305678010212†L234-L240】.  Alternative builds use dual‑PSU setups: one 2 000 W PSU for the compute components and a separate 550 W unit for pumps and fans【900884935542970†L64-L84】.
* **Heat management**: Dual‑GPU rigs behave like space heaters.  Modern GPU coolers vent most heat inside the case; with two cards this heat accumulates【792305678010212†L186-L224】.  High‑flow server‑grade fans or **liquid/immersion cooling** are recommended.  Immersion‑cooled builds use dielectric fluid tanks, large radiators and industrial pumps to keep both cards and a Threadripper Pro CPU under 70 °C【900884935542970†L54-L92】.
* **Form factor**: Because of their length (304 mm) and triple‑slot design, the cards require motherboards with at least two physical x16 slots spaced apart.  Many consumer boards cannot provide full PCIe 5.0 ×16 lanes to two GPUs.  Builders often use **AMD Threadripper Pro/WRX80 or EPYC platforms** (e.g., **ASUS Pro WS WRX80E‑SAGE SE**) which supply multiple PCIe 5.0 x16 slots and abundant lanes【900884935542970†L64-L68】.
* **CPU selection**: LLM inference is largely GPU‑bound, but multi‑GPU setups benefit from CPUs with high PCIe lane counts and sufficient cores for pre‑ and post‑processing.  Threadripper Pro 5975WX (32 cores, 280 W) or newer 79xx series are common in dual‑5090 builds【900884935542970†L64-L67】.

### Example build configurations

| Component | Example choice | Rationale | Source |
|---|---|---|---|
| **Motherboard** | ASUS Pro WS WRX80E‑SAGE SE WIFI or WRX90 | WRX80/WRX90 boards provide multiple PCIe 5.0 ×16 slots, ECC memory support and robust VRMs for 280 W CPUs【900884935542970†L64-L68】. | Forum build【900884935542970†L54-L92】 |
| **CPU** | AMD Threadripper Pro 5975WX/7960X or EPYC 9004 series | High core counts and plenty of PCIe lanes ensure both GPUs run at full bandwidth and supply enough host memory bandwidth for data transfer.  Threadripper 7960X is used in NCCL P2P tests【224217239444475†L117-L133】. | NVIDIA developer forums【224217239444475†L117-L133】 |
| **Memory** | 256 GB ECC DDR4/DDR5 | Large system memory is needed when offloading KV cache and quantized weights for models with long context.  ECC improves stability during long runs. | Forum build【900884935542970†L64-L68】 |
| **Storage** | 2 × 4 TB NVMe SSD (Gen4 or Gen5) | Fast NVMe drives reduce model‑loading times and support streaming of documents for retrieval‑augmented generation (RAG). | Forum build【900884935542970†L64-L69】 |
| **Power supply** | 2 800 W PSU (200–240 V) or dual‑PSU setup (2 000 W + 550 W) | Dual 5090s plus a high‑end CPU can draw ~1.5 kW–2 kW.  Puget Systems uses 2 800 W PSUs requiring 200–240 V circuits【792305678010212†L234-L240】; immersion builds split compute and cooling PSUs【900884935542970†L64-L84】. | Puget Systems【792305678010212†L205-L240】 and forum build【900884935542970†L64-L84】 |
| **Cooling** | High‑flow fans or custom liquid/immersion cooling | Dual‑GPU systems generate significant heat; server‑grade fans or immersion tanks with industrial radiators maintain safe temperatures【792305678010212†L219-L224】【900884935542970†L54-L92】. | Puget Systems【792305678010212†L219-L240】, Level1Techs forum【900884935542970†L54-L92】 |

### Multi‑GPU limitations and NCCL considerations

Because the RTX 5090 lacks NVLink and P2P support, multi‑GPU LLM inference uses **tensor parallelism** via frameworks like **vLLM**, **TensorRT‑LLM**, **DeepSpeed** or **Hugging Face Accelerate**.  Tensor parallelism partitions each model layer across GPUs; when GPUs cannot directly communicate, NCCL falls back to host memory transfers, reducing bandwidth.  NVIDIA moderators confirm that **GeForce 50‑series cards do not support P2P**【224217239444475†L420-L431】.  Recent NCCL releases work around this limitation, but throughput scaling is modest – the Private LLM Inference study shows dual RTX 5090 setups achieving only **1.14–1.29×** speed‑ups over a single 5090 for 8 k context RAG workloads【738629419566480†L1029-L1067】.  The main benefit of two GPUs is **increased VRAM and context length**: a dual‑5090 system can handle **32 k–64 k context** lengths and run larger models (e.g., 70B) using tensor parallelism【553366128949704†L94-L102】【738629419566480†L1069-L1073】.

## LLM inference on dual RTX 5090 rigs

### Single‑ vs dual‑GPU throughput

Benchmarks comparing Blackwell GPUs across workloads reveal that the RTX 5090 delivers **411 tokens/sec** for the Qwen3‑8B model in a retrieval‑augmented generation (RAG‑8 k) scenario, while a dual‑5090 setup delivers **530 tokens/sec**, a **1.29×** improvement【738629419566480†L1029-L1067】.  For API‑style high‑concurrency workloads (64 requests), the RTX 5090 achieves **6 894 tokens/sec** and the dual‑5090 achieves **7 438 tokens/sec**【738629419566480†L1029-L1034】.  Latency analysis shows that the single 5090 attains a **450 ms** time‑to‑first‑token (TTFT) for 8 k‑context RAG, compared to **620 ms** on dual 5090s because inter‑GPU synchronization adds overhead【738629419566480†L1030-L1044】.

### Model sizes and context length

* **7B–13B models**: A single RTX 5090 can run quantized 7 B models in FP16 or 4‑bit formats with 8 k–16 k contexts.  Dual GPUs enable **32 k or even 64 k contexts** using tensor parallelism with vLLM【738629419566480†L1069-L1073】.
* **70B models**: Quantized 70B models (e.g., Llama 3 70B or Nemotron 70B) require >170 GB memory.  The **Local LLM Hardware Guide** reports that dual RTX 5090 setups achieve **27 tokens/sec** on a 70B model (likely FP4/4‑bit), matching H100 performance at a fraction of the cost【553366128949704†L94-L102】.
* **Large models (100B+ parameters)**: Running 100B–405B models generally requires datacenter GPUs.  Dual 5090s lack the VRAM and interconnect to handle these models except with aggressive quantization and reduced context【553366128949704†L133-L158】.

### Recommended inference frameworks

* **vLLM** – An open‑source framework optimized for transformers with continuous batching and paged‑attention.  It supports tensor parallelism and multi‑GPU sharding.  Use `--tensor-parallel-size=2` for two GPUs and enable continuous batching to maximize throughput.  Use the `pinned_memory=True` option to reduce host–GPU transfer latency.
* **TensorRT‑LLM** – NVIDIA’s high‑performance inference library for GPUs.  It automatically applies **operator fusion**, **KV‑cache parallelization**, **attention layout optimization** and **quantization** (FP8, INT4, NVFP4).  When building engines, choose **NVFP4** quantization for the best throughput/energy trade‑off – the Private LLM Inference paper found NVFP4 delivers **1.6×** higher throughput than BF16 with **41 % lower energy consumption**【738629419566480†L1099-L1107】.
* **ExLlama V2 & llama.cpp** – For CPU‑side quantization (GGUF) and 4‑bit quantized models, ExLlama V2 provides high throughput on consumer GPUs.  Use Q4_K_M or AWQ 4‑bit quantization (3–4× memory reduction) to fit large models into 32 GB VRAM【553366128949704†L144-L154】.

### Software and driver tuning

* **CUDA & drivers** – Use the latest NVIDIA driver (e.g., 570.124 or later) and CUDA 12.8+ for Blackwell support.  Ensure that NCCL and cuBLAS are updated to versions that include work‑arounds for GeForce P2P limitations【224217239444475†L420-L431】.
* **Power management** – Many builders reduce each card’s power limit to **400 W** using `nvidia-smi -pl 400` to keep total draw within PSU limits and maintain lower temperatures; this often reduces throughput only marginally while improving stability【158557009141140†L0-L2】 (Reddit user build).  Undervolting via MSI Afterburner or NVIDIA SMI can also lower temperature.
* **System BIOS** – On WRX80/WRX90 boards, disable PCIe link power management and enable Above 4G Decoding.  Allocate maximum `Resizable BAR` size to each GPU for improved memory mapping.
* **Thermal monitoring** – Use tools like **nvtop** or **nvidia‑smi dmon** to watch GPU and memory temperatures.  Memory junction temperatures above 90 °C may indicate inadequate cooling【907804445390455†L120-L123】.

### RAG‑oriented optimizations

Local retrieval‑augmented generation workloads involve both CPU‑side retrieval and GPU inference.  Optimizations include:

1. **RAG database indexing** – Use a fast vector database (FAISS or Qdrant) stored on NVMe drives; pre‑compute embeddings with the same embedding model used during inference.  Running the database on the same machine avoids network latency.
2. **Batching and caching** – Use continuous batching (vLLM) so multiple requests are combined into one GPU kernel call.  Pre‑warm the KV cache by running a dummy query to reduce first‑token latency.  For repeated queries, maintain a **KV‑cache** across calls.
3. **Streaming responses** – Use asynchronous streaming to send tokens to the client as soon as they are generated.  This hides some of the synchronization overhead when using two GPUs.
4. **Quantization** – NVFP4 or AWQ quantization provides the best trade‑off between quality and memory footprint【738629419566480†L1099-L1107】【553366128949704†L144-L154】.  For CPU‑side quantization, `--device=cuda` and `--pre_load_kv_cache=True` options (vLLM) reduce overhead.
5. **Tensor parallelism** – When the model does not fit on a single 5090, set `tensor_parallel_size=2` or `tp_size=2` in vLLM/TensorRT‑LLM.  Note that throughput scaling is limited (1.14–1.29×) and TTFT increases【738629419566480†L1029-L1067】.  Use larger batch sizes to hide the synchronization latency.

## Creating a RAG database for agents

A Retrieval‑Augmented Generation (RAG) system stores knowledge in a searchable database and uses an embedding model to retrieve relevant text segments for LLM prompts.  Below is a sample dataset summarizing key facts about dual RTX 5090 rigs.  Each record contains a **topic**, a concise **fact**, and the **citation**.  The dataset is provided as a CSV file so it can be loaded into vector‑store frameworks such as FAISS, Qdrant or Chromadb.

### Summary dataset structure

| Column | Description |
|---|---|
| `topic` | High‑level category (hardware, performance, power, optimization, build, etc.) |
| `fact` | Concise statement about that topic (suitable as a knowledge chunk) |
| `citation` | Tether ID representing the source lines; maintain to validate claims |

An example row looks like `"performance","Dual 5090 achieves 530 TPS on Qwen3‑8B versus 411 TPS on single 5090 (1.29× speed‑up)","【738629419566480†L1029-L1067】"`.

The full dataset is supplied as a CSV file.

