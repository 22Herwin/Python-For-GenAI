# Python for GenAI - Workshop Modul 01 - 05

- **Nama:** Herwin Dermawan
- **NRP:** 5323600022
- **Tanggal:** 16/09/2026

Repository ini berisi jawaban tugas **Workshop Modul 01 - 05 Python Generative AI**: kode latihan Python untuk persiapan pengembangan aplikasi Generative AI, mengikuti materi Phase 1 - Python Core for AI dari panduan *Python for GenAI Complete Guide*.

Setiap module sudah dikerjakan lengkap - kode contoh materi + soal latihan (exercises) beserta solusinya - dan sudah diuji jalan tanpa error.

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

## Environment Variables (Module 04)

Beberapa contoh di Module 04 membaca API key dari `.env`. Copy template-nya lalu isi key sendiri:

```bash
cp Module04_FileIO_APIs/.env.example Module04_FileIO_APIs/.env
```
