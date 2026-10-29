# Python for GenAI

- **Nama:** Herwin Dermawan
- **NRP:** 5323600022
- **Tanggal:** 16/09/2026

Repository ini berisi jawaban tugas **Workshop Python Gen-AI**: kode latihan Python untuk persiapan pengembangan aplikasi Generative AI, mengikuti materi Phase 1 - Python Core for AI dari panduan *Python for GenAI Complete Guide*.

Setiap module sudah dikerjakan lengkap - kode contoh materi + soal latihan (exercises) beserta solusinya - dan sudah diuji jalan tanpa error.

Selain itu, repository ini juga menyertakan pengerjaan **Phase 2 - LLM APIs & Prompting (Module 06-08)** sebagai pengembangan lanjutan: Module 06 (Anthropic & OpenAI API), Module 07 (Prompt Engineering), dan Module 08 (Embeddings & Semantic Search).

**Phase 3 - GenAI Libraries (Module 09-14)** juga sudah ditulis, semuanya ada di folder `Phase03_GenAI_Libraries/`: Module 09 (LangChain), Module 10 (LlamaIndex), Module 11 (Vector Databases: ChromaDB, Pinecone, Weaviate), Module 12 (Hugging Face Transformers), Module 13 (LangGraph & Agents), dan Module 14 (CrewAI). Strukturnya sama dengan modul sebelumnya: satu file per sub-bab, plus `NN_exercises.py` berisi solusi soal latihan.

> **Status Phase 3:** semua file sudah lolos pengecekan syntax (`py_compile`), dan bagian logika yang tidak butuh library eksternal sudah dites dengan mock. Tapi library Phase 3 (langchain, llama-index, chromadb, transformers, langgraph, crewai, dll.) belum di-install saat kode ditulis, jadi **belum pernah dijalankan dengan library aslinya**. Setiap file akan mencetak pesan "package belum terpasang" dan keluar dengan rapi kalau library-nya belum ada. Bagian yang butuh API key atau mengunduh model besar dijelaskan di `# NOTE:` pada header tiap file.

## Setup

```bash
# 1. Clone repo
git clone https://github.com/22Herwin/Python-For-GenAI.git
cd Python-For-GenAI

# 2. (Opsional) buat virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> Module 01, 02, dan 03 tidak butuh dependency tambahan (murni Python standard library). Hanya Module 04 (`httpx`, `python-dotenv`) dan Module 05 (`numpy`, `pandas`) yang butuh package eksternal - semuanya ada di `requirements.txt`.

## Menjalankan Kode

```bash
python Module01_PythonFoundations/01_variables_and_types.py
python Module05_PythonForData/05_mini_project_eval_pipeline.py
```

Contoh Phase 3 (setelah install library modul terkait):

```bash
python Phase03_GenAI_Libraries/Module09_LangChain/03_lcel.py
python Phase03_GenAI_Libraries/Module13_LangGraph_Agents/05_exercises.py
python Phase03_GenAI_Libraries/Module12_HuggingFace/05_peft_lora_finetuning.py --train   # training LoRA, lambat di CPU
```

## Environment Variables (Module 04)

Beberapa contoh di Module 04 membaca API key dari `.env`. Copy template-nya lalu isi key sendiri:

```bash
cp Module04_FileIO_APIs/.env.example Module04_FileIO_APIs/.env
```
