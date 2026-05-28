"""
agent/nl_to_sparql.py
──────────────────────
Agent LLM (Ollama) qui traduit une question en langage naturel
en requête SPARQL valide sur notre ontologie de support.

Architecture :
  1. Prompt système : décrit l'ontologie et les patterns SPARQL
  2. Few-shot examples : 8 exemples NL→SPARQL couvrant les cas fréquents
  3. Appel Ollama (llama3 ou mistral)
  4. Extraction et validation de la requête générée
  5. Retry automatique si la requête est invalide (max 3 tentatives)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ─── Prompt système ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en SPARQL spécialisé dans l'interrogation de graphes de connaissances de support technique.

## Ontologie disponible

### Classes
- sup:SupportTicket   — ticket de support / rapport de bug
- sup:SoftwareComponent — composant logiciel (editor, terminal, debugger…)
- sup:ProductVersion  — version d'un produit (ex: 1.85.0)
- sup:Fix             — correctif appliqué à un ticket
- sup:Severity        — niveau de sévérité

### Propriétés d'objet
- sup:affectsComponent  (SupportTicket → SoftwareComponent)
- sup:affectsVersion    (SupportTicket → ProductVersion)
- sup:fixedBy           (SupportTicket → Fix)
- sup:introducedIn      (SupportTicket → ProductVersion)
- sup:resolvedIn        (Fix → ProductVersion)
- sup:dependsOn         (SoftwareComponent → SoftwareComponent)
- sup:belongsTo         (ProductVersion → SoftwareComponent)
- sup:hasSeverity       (SupportTicket → Severity)
- sup:relatedTo         (SupportTicket → SupportTicket) [symétrique]

### Propriétés de données
- sup:ticketId, sup:title, sup:description (xsd:string)
- sup:state             ("open" | "closed")
- sup:labels            (xsd:string, liste CSV)
- sup:createdAt, sup:closedAt (xsd:dateTime)
- sup:versionNumber     (xsd:string)
- sup:componentName     (xsd:string)
- rdfs:label            (libellé humain de toute entité)

### Individus de sévérité
sup:Critical, sup:High, sup:Medium, sup:Low

## Préfixes (déjà disponibles, ne pas redéclarer)
PREFIX sup:  <http://support.kg/ontology#>
PREFIX ent:  <http://support.kg/entity/>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

## Règles
1. Génère UNIQUEMENT une requête SPARQL SELECT valide, sans aucun texte avant ou après.
2. N'utilise JAMAIS de préfixes non déclarés.
3. Utilise FILTER(CONTAINS(LCASE(?x), "mot")) pour les recherches textuelles.
4. Limite toujours les résultats avec LIMIT (max 20).
5. Utilise OPTIONAL pour les champs qui peuvent être absents.
6. Pour les agrégations, utilise GROUP BY correctement.
7. La requête doit commencer par SELECT et finir par le } de WHERE.
"""

# ─── Few-shot examples ────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "question": "Quels tickets sont liés au composant terminal ?",
        "sparql": """SELECT ?ticketId ?title ?state WHERE {
  ?ticket a sup:SupportTicket ;
          sup:ticketId ?ticketId ;
          sup:title ?title ;
          sup:state ?state ;
          sup:affectsComponent ?comp .
  ?comp sup:componentName ?compName .
  FILTER(CONTAINS(LCASE(?compName), "terminal"))
}
LIMIT 10""",
    },
    {
        "question": "Combien de tickets critiques sont encore ouverts ?",
        "sparql": """SELECT (COUNT(?ticket) AS ?count) WHERE {
  ?ticket a sup:SupportTicket ;
          sup:hasSeverity sup:Critical ;
          sup:state "open" .
}""",
    },
    {
        "question": "Quels composants dépendent du composant editor ?",
        "sparql": """SELECT ?compName WHERE {
  ?comp sup:dependsOn ?dep ;
        sup:componentName ?compName .
  ?dep sup:componentName ?depName .
  FILTER(CONTAINS(LCASE(?depName), "editor"))
}
LIMIT 15""",
    },
    {
        "question": "Quels sont les tickets résolus dans la version 1.85 ?",
        "sparql": """SELECT ?ticketId ?title WHERE {
  ?ticket a sup:SupportTicket ;
          sup:ticketId ?ticketId ;
          sup:title ?title ;
          sup:fixedBy ?fix .
  ?fix sup:resolvedIn ?version .
  ?version sup:versionNumber ?ver .
  FILTER(CONTAINS(?ver, "1.85"))
}
LIMIT 10""",
    },
    {
        "question": "Donne-moi les 5 composants avec le plus de tickets.",
        "sparql": """SELECT ?compName (COUNT(?ticket) AS ?nbTickets) WHERE {
  ?ticket a sup:SupportTicket ;
          sup:affectsComponent ?comp .
  ?comp sup:componentName ?compName .
}
GROUP BY ?compName
ORDER BY DESC(?nbTickets)
LIMIT 5""",
    },
    {
        "question": "Quels tickets mentionnent un crash ou une erreur de mémoire ?",
        "sparql": """SELECT ?ticketId ?title WHERE {
  ?ticket a sup:SupportTicket ;
          sup:ticketId ?ticketId ;
          sup:title ?title .
  FILTER(
    CONTAINS(LCASE(?title), "crash") ||
    CONTAINS(LCASE(?title), "memory") ||
    CONTAINS(LCASE(?title), "out of memory")
  )
}
LIMIT 15""",
    },
    {
        "question": "Quels tickets sont reliés au ticket numéro 12345 ?",
        "sparql": """SELECT ?relId ?relTitle WHERE {
  {
    ent:ticket_12345 sup:relatedTo ?related .
  } UNION {
    ?related sup:relatedTo ent:ticket_12345 .
  }
  ?related sup:ticketId ?relId ;
           sup:title ?relTitle .
}
LIMIT 10""",
    },
    {
        "question": "Liste les versions affectées par des tickets de haute sévérité.",
        "sparql": """SELECT DISTINCT ?versionNumber WHERE {
  ?ticket a sup:SupportTicket ;
          sup:hasSeverity sup:High ;
          sup:affectsVersion ?version .
  ?version sup:versionNumber ?versionNumber .
}
ORDER BY ?versionNumber
LIMIT 20""",
    },
]


# ─── Agent ────────────────────────────────────────────────────────────────────

@dataclass
class NLToSPARQLAgent:
    model: str = "llama3"
    ollama_url: str = "http://localhost:11434"
    max_retries: int = 3
    temperature: float = 0.1
    _call_count: int = field(default=0, init=False)
    _total_tokens: int = field(default=0, init=False)

    def translate(self, question: str, context: str = "") -> dict[str, Any]:
        """
        Traduit une question NL en SPARQL.
        Retourne un dict avec : sparql, attempts, reasoning, error.
        """
        messages = self._build_messages(question, context)
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"[agent] Tentative {attempt}/{self.max_retries} pour : {question!r}")
            try:
                raw = self._call_ollama(messages)
                sparql = self._extract_sparql(raw)
                if sparql:
                    return {
                        "sparql": sparql,
                        "raw_response": raw,
                        "attempts": attempt,
                        "error": None,
                    }
                last_error = "Aucune requête SPARQL valide extraite."
                # Ajouter un message de correction pour la prochaine tentative
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "Ta réponse ne contient pas de requête SPARQL valide. "
                        "Génère UNIQUEMENT la requête SELECT … WHERE { … } LIMIT N, "
                        "sans aucun texte avant ou après."
                    ),
                })
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[agent] Erreur tentative {attempt} : {e}")
                time.sleep(1)

        return {
            "sparql": None,
            "raw_response": "",
            "attempts": self.max_retries,
            "error": last_error,
        }

    def _build_messages(self, question: str, context: str) -> list[dict]:
        """Construit le prompt complet avec few-shot examples."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Few-shot examples
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["question"]})
            messages.append({"role": "assistant", "content": ex["sparql"]})

        # Question réelle
        user_content = question
        if context:
            user_content += f"\n\nContexte additionnel : {context}"
        messages.append({"role": "user", "content": user_content})
        return messages

    def _call_ollama(self, messages: list[dict]) -> str:
        """Appelle l'API Ollama /api/chat."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        self._call_count += 1
        content = data.get("message", {}).get("content", "")
        # Compter les tokens approximativement
        self._total_tokens += len(content.split())
        return content

    @staticmethod
    def _extract_sparql(text: str) -> str | None:
        """
        Extrait la requête SPARQL du texte généré.
        Gère les blocs ```sparql … ```, les blocs ``` … ```, et le texte brut.
        """
        # Bloc ```sparql
        m = re.search(r"```sparql\s*([\s\S]+?)```", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Bloc ``` générique
        m = re.search(r"```\s*(SELECT[\s\S]+?)```", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Texte brut commençant par SELECT
        m = re.search(r"(SELECT\b[\s\S]+?(?:LIMIT\s+\d+|}\s*$))", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def check_ollama(self) -> bool:
        """Vérifie qu'Ollama est accessible et que le modèle est disponible."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                available = any(self.model in m for m in models)
                if not available:
                    logger.warning(
                        f"Modèle '{self.model}' non trouvé. "
                        f"Disponibles : {models}\n"
                        f"Lancez : ollama pull {self.model}"
                    )
                return available
        except Exception as e:
            logger.error(f"Ollama inaccessible : {e}")
        return False
