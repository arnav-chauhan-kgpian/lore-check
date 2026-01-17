# 📚 Backstory Consistency NLP - KDSH '26

> **A Retrieval Augmented Generation (RAG) system for verifying character backstory consistency in literature.**

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Groq](https://img.shields.io/badge/Groq-API-orange?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)
![Pathway](https://img.shields.io/badge/Pathway-Data%20Processing-blue?style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

</div>

## 🏆 Project Overview

This project was developed as a solution for the **Kharagpur Data Science Hackathon (KDSH) 2026** Challenge.

The goal of this system is to automatically validate the consistency of character backstories against a provided corpus of books. By leveraging advanced NLP techniques, the system decomposes complex backstories into atomic claims, retrieves relevant evidence from the source texts, and uses a Large Language Model (LLM) to verify whether the claims are supported, contradicted, or not mentioned in the text.

## 🛠️ Tech Stack

This project is built using a modern AI/NLP stack:

*   **🐍 Python**: The core programming language.
*   **🚅 Pathway**: A high-performance data processing framework used for efficient document ingestion and handling live data streams.
*   **🧠 SentenceTransformers**: Used for generating semantic embeddings (`all-MiniLM-L6-v2`) to capture the meaning of text chunks.
*   **⚡ Groq API**: Powers the high-speed inference for the LLM (Llama 3 / Mixtral) used in reasoning and verification.
*   **🔍 Cross-Encoder**: Utilized for precise re-ranking of retrieved evidence to ensure the most relevant context is sent to the LLM.
*   **🐼 Pandas**: Employed for structured data manipulation and results tracking.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `main.py` | 🚀 **Entry point.** Run this script to execute the full pipeline. |
| `data_ingestion.py` | 📥 Handles loading books and creating vector embeddings using **Pathway**. |
| `query_generator.py` | 🧩 Uses an LLM to decompose backstories into verifiable atomic claims and search queries. |
| `evidence_retrieval.py` | 🔎 Retrieves and re-ranks evidence chunks from the book corpus. |
| `verification.py` | ⚖️ The "Judge" - verifies claims against retrieved evidence using the LLM. |
| `validator.py` | ⚙️ Orchestrates the validation logic (called by `main.py`). |
| `Dataset/` | 📁 Contains `train.csv`, `test.csv`, and the `Books/` directory. |

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- A Groq API Key

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/sahilgupta630/KDSH.git
    cd KDSH
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key**
    Create a `.env` file in the project root and add your Groq API key:
    ```env
    GROQ_API_KEY=your_actual_api_key_here
    ```

### Running the Pipeline

To run the consistency check on the sample dataset:

```bash
python main.py
```

This will:
1.  Index the books in `Dataset/Books`.
2.  Decompose and verify stories from `Dataset/train.csv`.
3.  Save the results to `results.csv`.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
