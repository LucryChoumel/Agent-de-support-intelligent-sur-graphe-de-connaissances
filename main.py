"""
main.py
───────
Interface interactive en ligne de commande pour l'agent de support.
Permet d'interroger le KG en langage naturel via Ollama.

Usage :
    python main.py                          # Mode interactif
    python main.py --question "Quels tickets critiques ?"
    python main.py --stats                  # Stats du KG
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_separator(char: str = "─", width: int = 60):
    print(char * width)


def print_results(rows: list[dict], max_rows: int = 15):
    if not rows:
        print("  (aucun résultat)")
        return
    # En-têtes
    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows[:max_rows]))
                  for h in headers}
    header_line = "  " + " │ ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("  " + "─┼─".join("─" * col_widths[h] for h in headers))
    for row in rows[:max_rows]:
        line = "  " + " │ ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers)
        print(line)
    if len(rows) > max_rows:
        print(f"  … {len(rows) - max_rows} résultats supplémentaires")


def run_query(store, agent, question: str, verbose: bool = False) -> list[dict]:
    """Pipeline complet : NL → SPARQL → exécution → affichage."""
    print(f"\n Question : {question}")
    print_separator()

    # 1. Génération SPARQL
    print(" Génération SPARQL en cours…", end="", flush=True)
    result = agent.translate(question)
    print(f" ({result['attempts']} tentative(s))")

    if not result["sparql"]:
        print(f" ✗ Échec de génération : {result['error']}")
        return []

    sparql = result["sparql"]
    if verbose:
        print(f"\n SPARQL généré :\n")
        for line in sparql.splitlines():
            print(f"   {line}")
        print()

    # 2. Exécution sur le KG
    print(" Exécution sur le graphe…", end="", flush=True)
    try:
        rows = store.query(sparql, question=question)
        print(f" {len(rows)} résultat(s)\n")
        print_results(rows)
        return rows
    except Exception as e:
        print(f"\n ✗ Erreur d'exécution : {e}")
        if verbose:
            print(f"\n Requête fautive :\n{sparql}")
        return []


def run_stats(store):
    """Affiche les statistiques du KG."""
    print("\n Statistiques du graphe de connaissances")
    print_separator()
    stats = store.stats()
    labels = {
        "triples":    "Triplets RDF",
        "tickets":    "Tickets de support",
        "components": "Composants logiciels",
        "versions":   "Versions de produit",
        "fixes":      "Correctifs (Fixes)",
    }
    for key, label in labels.items():
        val = stats.get(key, 0)
        bar = "█" * min(val // 10, 40) if val > 0 else ""
        print(f"  {label:<25} {val:>6}  {bar}")
    print()


def interactive_mode(store, agent):
    """Mode REPL interactif."""
    print("\n" + "═" * 60)
    print("  AGENT DE SUPPORT — Graphe de Connaissances")
    print("  Tapez votre question en français ou anglais.")
    print("  Commandes : /stats  /sparql  /history  /quit")
    print("═" * 60)

    verbose = False
    while True:
        try:
            user_input = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
            print("Au revoir !")
            break

        elif user_input.lower() == "/stats":
            run_stats(store)

        elif user_input.lower() == "/sparql":
            verbose = not verbose
            print(f" Mode SPARQL verbose : {'activé' if verbose else 'désactivé'}")

        elif user_input.lower() == "/history":
            history = store.query_history[-5:]
            if not history:
                print(" Aucune requête dans l'historique.")
            else:
                print(f"\n Dernières {len(history)} requêtes :")
                for i, log in enumerate(history, 1):
                    status = "✓" if not log.error else "✗"
                    print(f"  {status} [{i}] {log.question[:55]}…")
                    print(f"      {len(log.results)} résultats — {log.duration_ms:.0f}ms")

        elif user_input.startswith("/"):
            print(f" Commande inconnue : {user_input}")

        else:
            run_query(store, agent, user_input, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Agent de support NL→SPARQL sur graphe de connaissances"
    )
    parser.add_argument("--kg", default="data/kg_triples.ttl",
                        help="Chemin vers le fichier Turtle du KG")
    parser.add_argument("--model", default="llama3",
                        help="Modèle Ollama (llama3, mistral, mixtral…)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--question", "-q", default=None,
                        help="Question unique (mode non-interactif)")
    parser.add_argument("--stats", action="store_true",
                        help="Afficher les stats du KG et quitter")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Afficher la requête SPARQL générée")
    parser.add_argument("--output", default=None,
                        help="Sauvegarder les résultats en JSON")
    args = parser.parse_args()

    # ── Chargement du KG ──────────────────────────────────────────────────────
    from kg.store import KnowledgeGraphStore
    from agent.nl_to_sparql import NLToSPARQLAgent

    store = KnowledgeGraphStore(ttl_path=args.kg)
    try:
        store.load()
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print("\nPour créer le graphe, lancez :")
        print("  python scripts/ingest_github.py --repo microsoft/vscode --max-issues 200")
        sys.exit(1)

    # ── Vérification Ollama ───────────────────────────────────────────────────
    agent = NLToSPARQLAgent(model=args.model, ollama_url=args.ollama_url)

    if not agent.check_ollama():
        print(f"\n⚠  Ollama inaccessible ou modèle '{args.model}' introuvable.")
        print(f"   Vérifiez qu'Ollama tourne : ollama serve")
        print(f"   Et que le modèle est téléchargé : ollama pull {args.model}")
        sys.exit(1)

    # ── Modes d'exécution ─────────────────────────────────────────────────────
    if args.stats:
        run_stats(store)

    elif args.question:
        rows = run_query(store, agent, args.question, verbose=args.verbose)
        if args.output:
            Path(args.output).write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\nRésultats sauvegardés → {args.output}")

    else:
        interactive_mode(store, agent)


if __name__ == "__main__":
    main()
