# 📄 CV Analyzer - AI-Powered Resume Screening

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Intelligent CV/Job Offer matching using Fine-tuned LLMs, RAG, and Semantic Search**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture)

</div>

---

## 🎯 Overview

CV Analyzer is an AI-powered recruitment assistant that automatically evaluates candidate resumes against job descriptions. It combines:

- 🧠 **Fine-tuned LLM** (Gemma 3 with LoRA adapters) for expert-level analysis
- 🔍 **RAG Pipeline** (LlamaIndex + ChromaDB) for semantic CV retrieval
- 📊 **Structured Output** with actionable recommendations

> Perfect for HR teams, recruiters, and talent acquisition specialists looking to streamline their screening process.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 **PDF Ingestion** | Upload and index multiple CVs with automatic text extraction |
| 🔎 **Semantic Search** | Find the most relevant CVs for any job description |
| 🎯 **Skill Matching** | Granular analysis of technical & soft skills |
| 📈 **Scoring System** | 0-10 relevance score with Go/No Go recommendations |
| 💡 **Actionable Insights** | Strengths, weaknesses, and interview talking points |
| ⚡ **vLLM Backend** | High-performance inference with LoRA adapter support |

---

## 🖼️ Demo

<div align="center">

```
┌─────────────────────────────────────────────────────────────┐
│  📄 CV Analyzer - vLLM + LoRA + RAG                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📋 Job Description                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Looking for a Senior Python Developer with 5+       │   │
│  │ years experience in ML/AI, FastAPI, and cloud...    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [🔎 Search & Analyze CVs]                                  │
│                                                             │
│  📊 Results                                                 │
│  ┌──────────────┬─────────┬──────────┬────────────┐        │
│  │ CV           │ Score   │ Reco     │ Similarity │        │
│  ├──────────────┼─────────┼──────────┼────────────┤        │
│  │ alice_cv.pdf │ 🟢 8/10 │ ✅ Go    │ 0.892      │        │
│  │ bob_cv.pdf   │ 🟡 5/10 │ 🔍 Review│ 0.756      │        │
│  │ carol_cv.pdf │ 🔴 3/10 │ ❌ No Go │ 0.634      │        │
│  └──────────────┴─────────┴──────────┴────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

</div>

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- A running vLLM server with your fine-tuned model
- (Optional) CUDA-compatible GPU for local inference

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/cv-analyzer.git
cd cv-analyzer/app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app_vllm.py
```

### Environment Variables

Create a `.env` file (optional):

```env
OPENAI_API_KEY for the semantic chunker
```

---

## 📖 Usage

### 1️⃣ Index Your CVs

1. Navigate to the **📤 Indexation CVs** tab
2. Upload PDF resumes
3. Click **🚀 Indexer**

### 2️⃣ Analyze Candidates

1. Go to the **🔍 Recherche & Analyse** tab
2. Paste your job description
3. Click **🔎 Rechercher et Analyser les CVs**
4. Review the ranked results with detailed analysis

### 3️⃣ Explore Results

- Click on any row to see detailed analysis
- View skill-by-skill matching
- Read AI-generated justifications
- Preview the original PDF

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Streamlit  │────▶│  LlamaIndex  │────▶│   ChromaDB   │
│   Frontend   │     │  (RAG/Embed) │     │ (Vector Store│
└──────────────┘     └──────────────┘     └──────────────┘
       │                                         │
       │              ┌──────────────┐           │
       └─────────────▶│    vLLM      │◀──────────┘
                      │ (Gemma+LoRA) │
                      └──────────────┘
```

### Key Components

| File | Purpose |
|------|---------|
| [`app_vllm.py`](app_vllm.py) | Main Streamlit application |
| [`vllm_langchain.py`](vllm_langchain.py) | LLM integration & structured output schemas |
| `chroma_db/` | Persistent vector database |
| `uploads/` | Indexed PDF storage |

---

## 📊 Output Schema

The LLM returns structured JSON with:

```python
{
    "score_global": 8,           # 0-10 relevance score
    "recommandation": "Go",      # Go | No Go | A creuser
    "points_forts": [...],       # Candidate strengths
    "points_faibles": [...],     # Gaps vs requirements
    "points_attention": [...],   # Interview topics
    "competences_techniques": [  # Skill-by-skill analysis
        {
            "competence": "Python",
            "niveau_requis": "Expert",
            "niveau_candidat": "Avancé",
            "match": "partiel"
        }
    ],
    "justification_score": "...",
    "justification_recommandation": "..."
}
```

---

## 🛠️ Configuration

Adjust settings in the sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| vLLM URL | `http://localhost:8000/v1` | Your vLLM server endpoint |
| LoRA Name | `cv-analyzer` | Fine-tuned adapter name |
| Temperature | `0.7` | Response creativity (0-1) |
| Top K CVs | `3` | Number of CVs to analyze |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [LlamaIndex](https://github.com/run-llama/llama_index) - RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Streamlit](https://streamlit.io/) - App framework

---

<div align="center">

**Built with ❤️ for smarter recruitment**

⭐ Star this repo if you find it useful!

</div>
