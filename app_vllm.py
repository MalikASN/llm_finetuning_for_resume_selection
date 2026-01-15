"""
Application Streamlit avec vLLM + LoRA + RAG (LlamaIndex/ChromaDB)
"""

import streamlit as st
import pandas as pd
import json
import base64
from pathlib import Path
import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from vllm_langchain import  analyze_cv, AnalyseStructuree

# Configurer le cache LangChain (persistant sur disque)
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

APP_ROOT = Path(__file__).resolve().parent
ADAPTER_PATH = APP_ROOT / "modelvllm"
UPLOADS_DIR = APP_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR = APP_ROOT / "chroma_db"


# =============================================================================
# FONCTIONS RAG (LlamaIndex + ChromaDB)
# =============================================================================

def setup_llama_index():
    """Configure LlamaIndex avec embeddings locaux."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_chroma_collection():
    """Récupère ou crée la collection ChromaDB."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("cv_collection")

def ingest_pdfs(files: list) -> tuple[bool, str]:
    """Indexe les PDFs dans ChromaDB."""
    try:
        setup_llama_index()
        file_paths = []
        for file in files:
            path = UPLOADS_DIR / file.name
            path.write_bytes(file.getbuffer())
            file_paths.append(str(path))

        collection = get_chroma_collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
        
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

        for doc in documents:
            doc.metadata["source_file"] = Path(doc.metadata.get("file_path", "unknown")).name
            index.insert(doc)

        return True, f"✅ {len(documents)} documents indexés."
    except Exception as e:
        return False, f"❌ Erreur: {str(e)}"

def search_cvs(query: str, top_k: int = 5) -> list[dict]:
    """Recherche sémantique dans la base de CVs."""
    setup_llama_index()
    collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)
    
    formatted = []
    for i, item in enumerate(results):
        node = item.node
        meta = node.metadata or {}
        formatted.append({
            "rank": i + 1,
            "source_file": meta.get("source_file", "Inconnu"),
            "score": round(item.score, 3) if item.score else 0,
            "content": node.get_content()
        })
    return formatted

def get_collection_stats() -> dict:
    """Stats de la collection."""
    try:
        return {"count": get_chroma_collection().count()}
    except:
        return {"count": 0}

def display_pdf(file_path: Path):
    """Affiche un PDF."""
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    st.markdown(f'<embed src="data:application/pdf;base64,{b64}" type="application/pdf" width="100%" height="600px"/>', unsafe_allow_html=True)


# =============================================================================
# APPLICATION STREAMLIT
# =============================================================================

def main():
    st.set_page_config(page_title="CV Analyzer - vLLM + RAG", layout="wide")
    st.title("📄 Analyse CV/Offre (vLLM + LoRA + RAG)")

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        vllm_url = st.text_input("URL vLLM", value="http://localhost:8000/v1")
        lora_name = st.text_input("Nom LoRA", value="cv-analyzer")
        temperature = st.slider("Température", 0.0, 1.0, 0.7)
        top_k = st.slider("Nb CVs à analyser", 1, 10, 3)
        
        st.divider()
        stats = get_collection_stats()
        st.metric("📚 CVs indexés", stats["count"])
    

    # --- Tabs ---
    tab1, tab2 = st.tabs(["🔍 Recherche & Analyse", "📤 Indexation CVs"])
    
    # =========================
    # TAB 1: Recherche & Analyse
    # =========================
    with tab1:
        st.subheader("📋 Offre d'emploi")
        offre = st.text_area("Collez l'offre ici", height=200, key="offre")
        
        if st.button("🔎 Rechercher et Analyser les CVs", type="primary", disabled=not offre):
            try:
                # 1. Recherche RAG
                with st.spinner("Recherche des CVs pertinents..."):
                    cv_results = search_cvs(offre, top_k=top_k)
                
                if not cv_results:
                    st.warning("Aucun CV trouvé. Indexez d'abord des CVs.")
                    return
                
                st.success(f"✅ {len(cv_results)} CVs trouvés")
                
                # 2. Analyse avec vLLM
                llm = ChatOpenAI(
                        base_url= vllm_url,
                        api_key="EMPTY",
                        model="./gemma-3-finetune",
                        max_completion_tokens=6000
                    )
                
                analyses = []
                for cv_data in cv_results:
                    with st.spinner(f"Analyse de {cv_data['source_file']}..."):
                        result = analyze_cv(llm, offre, cv_data["content"])
                        
                        if isinstance(result, AnalyseStructuree):
                            data = result.model_dump()
                        elif isinstance(result, dict):
                            data = result
                        else:
                            data = {"score_global": 0, "recommandation": "Erreur", "raw": str(result)}
                        
                        data["source_file"] = cv_data["source_file"]
                        data["similarity_score"] = cv_data["score"]
                        analyses.append(data)
                
                # 3. Affichage résultats
                st.session_state["analyses"] = analyses
                    
            except Exception as e:
                st.error(f"Erreur: {e}")
                st.info("Vérifiez que le serveur vLLM est lancé.")
        
       
        if "analyses" in st.session_state:
            display_analyses(st.session_state["analyses"])
    
    # =========================
    # TAB 2: Indexation
    # =========================
    with tab2:
        st.subheader("📤 Indexer des CVs (PDF)")
        files = st.file_uploader("Sélectionnez les PDFs", type=["pdf"], accept_multiple_files=True)
        
        if st.button("🚀 Indexer", disabled=not files):
            with st.spinner("Indexation en cours..."):
                success, msg = ingest_pdfs(files)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)


def display_analyses(analyses: list):
    """Affiche les résultats d'analyse."""
    
    if not analyses:
        st.warning("Aucune analyse à afficher.")
        return
    
    # Tableau récapitulatif
    rows = []
    for i, a in enumerate(analyses):
        score = a.get("score_global", 0)
        reco = a.get("recommandation", "N/A")
        emoji = "✅" if reco == "Go" else "❌" if reco == "No Go" else "🔍"
        color = "🟢" if score >= 7 else "🟡" if score >= 4 else "🔴"
        rows.append({
            "idx": i, 
            "CV": a.get("source_file", "?"),
            "Score_num": score, 
            "Score": f"{color} {score}/10",
            "Reco": f"{emoji} {reco}",
            "Similarité": round(a.get("similarity_score", 0), 3)
        })
    
    st.subheader("📊 Résultats")
    df = pd.DataFrame(rows)
    df = df.sort_values(by="Score_num", ascending=False).reset_index(drop=True)
    
    # Colonnes à afficher 
    column_config = {
        "idx": None,
        "Score_num": None,
    }
    event = st.dataframe(
        df, 
        use_container_width=True, 
        hide_index=True, 
        on_select="rerun", 
        selection_mode="single-row",
        column_config=column_config
    )
    
    # Détails du CV sélectionné
    if event.selection and event.selection.rows:
        row_idx = event.selection.rows[0]
        original_idx = int(df.iloc[row_idx]["idx"])
        selected = analyses[original_idx]
        
        st.divider()
        st.subheader(f"📄 Détails: {selected.get('source_file', '?')}")
        
        # Points clés
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**✅ Points Forts**")
            for p in selected.get("points_forts", []):
                st.markdown(f"- {p}")
        with col2:
            st.markdown("**❌ Points Faibles**")
            for p in selected.get("points_faibles", []):
                st.markdown(f"- {p}")
        with col3:
            st.markdown("**⚠️ Attention**")
            for p in selected.get("points_attention", []):
                st.markdown(f"- {p}")
        
        # Compétences
        competences = selected.get("competences_techniques", [])
        if competences:
            st.markdown("**📊 Compétences**")
            comp_rows = []
            for c in competences:
                if isinstance(c, dict):
                    m = c.get("match", "")
                    e = "✅" if m == "exact" else "⚠️" if m == "partiel" else "❌"
                    comp_rows.append({
                        "Compétence": c.get("competence", ""),
                        "Requis": c.get("niveau_requis", ""),
                        "Candidat": c.get("niveau_candidat", ""),
                        "Match": f"{e} {m}"
                    })
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        
        # Justifications
        with st.expander("📝 Justifications"):
            st.write(f"**Score:** {selected.get('justification_score', 'N/A')}")
            st.write(f"**Reco:** {selected.get('justification_recommandation', 'N/A')}")
        
        # PDF
        pdf_path = UPLOADS_DIR / selected.get("source_file", "")
        if pdf_path.exists():
            with st.expander("📄 Voir le PDF"):
                display_pdf(pdf_path)


if __name__ == "__main__":
    main()
