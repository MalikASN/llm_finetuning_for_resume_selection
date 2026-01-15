

from typing import Optional, List, Any, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import requests
import json


# =============================================================================
# SCHÉMAS PYDANTIC (format de sortie du modèle fine-tuné)
# =============================================================================

class CompetenceMatch(BaseModel):
    competence: str = Field(description="Le nom précis de la compétence technique ou soft skill évaluée")
    niveau_requis: str = Field(description="Le niveau d'expertise demandé dans l'offre")
    niveau_candidat: str = Field(description="Le niveau d'expertise estimé du candidat")
    match: Literal["exact", "partiel", "manquant"] = Field(description="Verdict de correspondance")


class AnalyseStructuree(BaseModel):
    """Format de sortie structuré pour l'analyse CV/Offre."""
    
    # Décision Globale
    score_global: int = Field(description="Note de pertinence de 0 à 10")
    recommandation: Literal["Go", "No Go", "A creuser"] = Field(description="Recommandation d'action")
    
    # Analyse Détaillée
    points_forts: List[str] = Field(description="Atouts majeurs du candidat")
    points_faibles: List[str] = Field(description="Lacunes par rapport au poste")
    points_attention: List[str] = Field(description="Points à vérifier en entretien")
    
    # Matching Granulaire
    competences_techniques: List[CompetenceMatch] = Field(description="Analyse des compétences clés")
    experience_adequation: str = Field(description="Analyse de l'expérience vs exigences")
    formation_adequation: str = Field(description="Analyse de la formation vs prérequis")
    
    # Justification
    justification_score: str = Field(description="Explication de la note")
    justification_recommandation: str = Field(description="Argumentaire pour le recruteur")


# =============================================================================
# PROMPTS SYSTÈME (identiques au fine-tuning)
# =============================================================================

SYSTEM_PROMPT = """Tu es un expert Senior en Recrutement et Talent Acquisition.
Ton objectif est d'évaluer la pertinence d'un candidat pour un poste donné.

Instructions d'analyse :
1. **Analyse de compatibilité** : Compare les compétences du CV avec les exigences du poste.
2. **Gestion des données manquantes** : Si le CV ou l'offre contient des champs vides ou "[NON SPÉCIFIÉ]", ne t'arrête pas. Utilise le contexte global pour ton évaluation.
3. **Objectivité** : Base ton score uniquement sur les éléments factuels présents.
4. **Exhaustivité** : Analyse TOUTES les compétences clés mentionnées dans l'offre.

Réponds UNIQUEMENT avec le format JSON structuré demandé."""


def format_user_prompt(offre: str, cv: str) -> str:
    """Formate le prompt utilisateur avec l'offre et le CV."""
    return f"""### OFFRE D'EMPLOI ###
{offre}

### CV DU CANDIDAT ###
{cv}

### INSTRUCTION ###
Procède à l'évaluation complète et structurée de ce candidat pour ce poste."""



def analyze_cv(
    llm: ChatOpenAI,
    offre: str,
    cv: str,
) -> AnalyseStructuree | dict | str:
    """
    Analyse un CV par rapport à une offre d'emploi.
    
    Args:
        llm: Instance VLLMChatModel
        offre: Texte de l'offre d'emploi
        cv: Texte du CV du candidat
    
    Returns:
        AnalyseStructuree si parsing réussi, dict ou str sinon
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=format_user_prompt(offre, cv))
    ]
    
    response = llm.invoke(messages)
    content = response.content

    try:
   
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        data = json.loads(content)
        return AnalyseStructuree(**data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Impossible de parser en AnalyseStructuree: {e}")
        try:
            return json.loads(content)
        except:
            return content


