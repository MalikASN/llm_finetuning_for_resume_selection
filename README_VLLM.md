# vLLM + LoRA + LangChain RAG Pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit App                            │
│                   (app_vllm.py)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  LangChain Agent                            │
│            (vllm_langchain.py)                              │
│  ┌─────────────────┐    ┌─────────────────────────────┐    │
│  │  Tool: search   │    │   VLLMChatModel             │    │
│  │  (RAG ChromaDB) │    │   (API OpenAI-compatible)   │    │
│  └─────────────────┘    └──────────────┬──────────────┘    │
└─────────────────────────────────────────┼───────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────┐
│              vLLM Server (port 8000)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Base Model: Ministral-3-3B-Instruct-2512            │  │
│  │  + LoRA Adapter: cv-analyzer (modelvllm/)            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# 1. Installer vLLM (GPU CUDA requis)
pip install vllm

# 2. Installer les dépendances LangChain
pip install langchain langchain-core

# 3. Installer les autres dépendances
pip install streamlit chromadb llama-index llama-index-embeddings-huggingface
```

## Utilisation

### Étape 1: Lancer le serveur vLLM

```powershell
# Windows
cd app
.\start_vllm_server.ps1
```

```bash
# Linux/Mac
cd app
bash start_vllm_server.sh
```

Ou manuellement:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model unsloth/Ministral-3-3B-Instruct-2512 \
    --enable-lora \
    --lora-modules cv-analyzer=./modelvllm \
    --max-lora-rank 64 \
    --port 8000 \
    --trust-remote-code
```

### Étape 2: Lancer l'application Streamlit

```bash
cd app
streamlit run app_vllm.py
```

## Configuration de l'adapter LoRA

Ton adapter est dans `app/modelvllm/` avec:
- `adapter_config.json` - Configuration LoRA (r=16, lora_alpha=16)
- `adapter_model.safetensors` - Poids de l'adapter
- `tokenizer.json` - Tokenizer

## API vLLM

Le serveur vLLM expose une API compatible OpenAI:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "cv-analyzer",  # Nom du LoRA
        "messages": [
            {"role": "system", "content": "Tu es un expert en recrutement..."},
            {"role": "user", "content": "Analyse ce CV..."}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }
)
print(response.json())
```

## Chargement dynamique de LoRA

vLLM supporte le chargement dynamique de plusieurs adapters:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model unsloth/Ministral-3-3B-Instruct-2512 \
    --enable-lora \
    --lora-modules \
        cv-analyzer=./modelvllm \
        autre-adapter=./autre_modelvllm \
    --max-lora-rank 64
```

Puis dans les requêtes, spécifie `"model": "cv-analyzer"` ou `"model": "autre-adapter"`.

## Intégration LangChain

```python
from vllm_langchain import create_vllm_llm
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Créer le LLM
llm = create_vllm_llm(
    mode="server",
    base_url="http://localhost:8000/v1",
    lora_name="cv-analyzer",
    temperature=0.7,
)

# Créer un agent
agent = create_tool_calling_agent(llm, tools=[...], prompt=...)
executor = AgentExecutor(agent=agent, tools=[...])

# Exécuter
response = executor.invoke({"input": "Trouve les meilleurs candidats..."})
```

## Dépannage

### Erreur: "CUDA out of memory"
- Réduire `--gpu-memory-utilization` (ex: 0.6)
- Utiliser `--quantization awq` pour la quantification

### Erreur: "Model not found"
- Vérifier que le chemin de l'adapter est correct
- Vérifier que `adapter_config.json` existe

### Erreur: "LoRA rank mismatch"
- Ajuster `--max-lora-rank` selon ton `r` dans adapter_config.json
