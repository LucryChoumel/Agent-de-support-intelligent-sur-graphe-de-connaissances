# Agent de Support Intelligent — Graphe de Connaissances + LLM

Projet académique aligné sur l'offre **CIFRE SAP Labs France** (IA Agentique & Graphes de Connaissances).

Pipeline complet : données GitHub réelles → ontologie RDF/OWL → agent NL→SPARQL (Ollama) → benchmark d'évaluation.

---

## Architecture

```
kg_support_agent/
├── ontology/
│   └── support_ontology.ttl   # Ontologie OWL formelle (classes, propriétés, individus)
├── scripts/
│   └── ingest_github.py       # Ingestion GitHub Issues → triplets RDF
├── kg/
│   └── store.py               # Chargement KG + interface SPARQL
├── agent/
│   └── nl_to_sparql.py        # Agent LLM (Ollama) NL → SPARQL
├── evaluation/
│   └── benchmark.py           # 50 questions, métriques EA/AA/latence
├── data/                      # Fichiers Turtle générés (gitignored)
├── main.py                    # Interface interactive CLI
└── requirements.txt
```

## Ontologie

Inspirée du modèle SAP PPMS, l'ontologie définit :

| Classe              | Rôle                                  |
|---------------------|---------------------------------------|
| `SupportTicket`     | Bug / rapport de support (≈ SAP Note) |
| `SoftwareComponent` | Module logiciel (≈ SimCat)            |
| `ProductVersion`    | Version de produit (≈ PPMS)           |
| `Fix`               | Correctif / SupportPackage            |
| `Severity`          | critical / high / medium / low        |

Relations clés : `affectsComponent`, `affectsVersion`, `fixedBy`, `resolvedIn`, `dependsOn`, `relatedTo`.

---

## Installation

### 1. Dépendances Python

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ollama

```bash
# Installer Ollama : https://ollama.com
ollama serve                        # Démarrer le serveur
ollama pull llama3                  # Télécharger le modèle (~4 GB)
# ou : ollama pull mistral
```

---

## Utilisation

### Étape 1 — Construire le graphe de connaissances

```bash
# Avec token GitHub (recommandé, 5000 req/h)
python scripts/ingest_github.py \
    --repo microsoft/vscode \
    --max-issues 300 \
    --token ghp_VOTRE_TOKEN \
    --output data/kg_triples.ttl

# Sans token (60 req/h, suffisant pour tester)
python scripts/ingest_github.py --repo microsoft/vscode --max-issues 60
```

Autres dépôts riches en issues : `torvalds/linux`, `apache/spark`, `kubernetes/kubernetes`

### Étape 2 — Mode interactif

```bash
python main.py --model llama3

# Exemples de questions :
❯ Quels tickets critiques sont encore ouverts ?
❯ Quel composant a le plus de bugs ?
❯ Quelles versions sont affectées par des crashes ?
❯ /stats         → statistiques du graphe
❯ /sparql        → afficher la requête SPARQL générée
❯ /history       → 5 dernières requêtes
```

### Étape 3 — Benchmark d'évaluation (50 questions)

```bash
python evaluation/benchmark.py \
    --kg data/kg_triples.ttl \
    --model llama3 \
    --output evaluation/results.json

# Test rapide (10 questions)
python evaluation/benchmark.py --quick
```

**Métriques produites :**
- **Execution Accuracy (EA)** : % de requêtes SPARQL qui s'exécutent sans erreur
- **Answer Accuracy (AA)** : % de réponses cohérentes avec la question
- **SPARQL Generated** : % de questions avec une requête générée
- **Latence moyenne** : temps de génération par question

### Étape 4 — Question unique (scripts, CI)

```bash
python main.py \
    --question "Combien de tickets critiques sont ouverts ?" \
    --verbose \
    --output results.json
```

---

## Connexion SAP

| Ce projet                    | Offre SAP CIFRE                        |
|-----------------------------|----------------------------------------|
| Ontologie OWL + RDF          | Standards RDF + ontologie formelle     |
| GitHub Issues                | SAP Notes, alertes système             |
| SoftwareComponent + Version  | SimCat + PPMS (ProductVersion, etc.)   |
| Agent NL→SPARQL              | LLM pilotés par raisonnement sur KG    |
| Benchmark 50 questions       | Pipeline évalué par benchmarks         |
| `dependsOn`, `relatedTo`     | Graphe sémantiquement structuré        |

---

## Pistes d'extension

- **Raisonnement OWL-RL** avec `owlrl` pour inférer de nouveaux faits
- **Fine-tuning** d'un petit modèle sur les paires NL→SPARQL générées
- **Comparaison** avec RAG vectoriel (ChromaDB / FAISS) sur les mêmes questions
- **Visualisation** du graphe avec NetworkX ou Gephi
- **Multi-repo** : fusionner plusieurs dépôts dans un seul KG fédéré
