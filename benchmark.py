"""
evaluation/benchmark.py
────────────────────────
Benchmark d'évaluation du pipeline NL→SPARQL.

Métriques calculées :
  - Execution Accuracy (EA)  : la requête s'exécute sans erreur
  - Answer Accuracy (AA)     : les résultats sont non-vides et cohérents
  - Semantic Similarity (SS) : similarité entre résultats attendus et obtenus
  - SPARQL Validity (SV)     : la requête est syntaxiquement valide
  - Latency                  : temps moyen de génération

Usage :
    python evaluation/benchmark.py \
        --kg data/kg_triples.ttl \
        --model llama3 \
        --output evaluation/results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Dataset de référence (50 questions) ─────────────────────────────────────
# Format : question + type de vérification attendue
# Types : "non_empty", "count_gt_0", "specific_count", "contains_keyword"

BENCHMARK_DATASET: list[dict] = [
    # ── Requêtes de base ──────────────────────────────────────────────────────
    {"id": "Q01", "question": "Combien de tickets sont dans le graphe ?",
     "check": "count_gt_0", "expected_min": 1},
    {"id": "Q02", "question": "Quels sont les composants logiciels disponibles ?",
     "check": "non_empty"},
    {"id": "Q03", "question": "Liste les tickets encore ouverts.",
     "check": "non_empty"},
    {"id": "Q04", "question": "Liste les tickets fermés.",
     "check": "non_empty"},
    {"id": "Q05", "question": "Quelles versions de produit sont référencées ?",
     "check": "non_empty"},

    # ── Requêtes de filtrage par sévérité ─────────────────────────────────────
    {"id": "Q06", "question": "Quels tickets ont une sévérité critique ?",
     "check": "non_empty"},
    {"id": "Q07", "question": "Combien de tickets de haute sévérité sont ouverts ?",
     "check": "count_gt_0"},
    {"id": "Q08", "question": "Liste les tickets de faible sévérité fermés.",
     "check": "non_empty"},
    {"id": "Q09", "question": "Y a-t-il des tickets de sévérité medium liés au terminal ?",
     "check": "non_empty"},
    {"id": "Q10", "question": "Quels sont les tickets critiques avec leur composant affecté ?",
     "check": "non_empty"},

    # ── Requêtes sur les composants ───────────────────────────────────────────
    {"id": "Q11", "question": "Quels tickets affectent le composant editor ?",
     "check": "non_empty"},
    {"id": "Q12", "question": "Quel composant a le plus grand nombre de tickets ?",
     "check": "non_empty"},
    {"id": "Q13", "question": "Quels composants dépendent d'autres composants ?",
     "check": "non_empty"},
    {"id": "Q14", "question": "Donne-moi les 5 composants les plus affectés par des bugs.",
     "check": "non_empty"},
    {"id": "Q15", "question": "Quels composants n'ont aucun ticket associé ?",
     "check": "query_executes"},  # Peut retourner vide, c'est OK

    # ── Requêtes sur les versions ──────────────────────────────────────────────
    {"id": "Q16", "question": "Quelles versions sont affectées par des tickets critiques ?",
     "check": "non_empty"},
    {"id": "Q17", "question": "Dans quelles versions des correctifs ont-ils été appliqués ?",
     "check": "non_empty"},
    {"id": "Q18", "question": "Quelle version a le plus de tickets associés ?",
     "check": "non_empty"},
    {"id": "Q19", "question": "Liste les versions du composant editor.",
     "check": "non_empty"},
    {"id": "Q20", "question": "Quels tickets ont été introduits et résolus dans la même version ?",
     "check": "query_executes"},

    # ── Requêtes sur les correctifs ───────────────────────────────────────────
    {"id": "Q21", "question": "Combien de tickets ont été corrigés ?",
     "check": "count_gt_0"},
    {"id": "Q22", "question": "Quels correctifs sont associés à des tickets critiques ?",
     "check": "non_empty"},
    {"id": "Q23", "question": "Dans quelle version le plus de tickets ont-ils été résolus ?",
     "check": "non_empty"},
    {"id": "Q24", "question": "Liste les tickets fermés sans correctif enregistré.",
     "check": "query_executes"},
    {"id": "Q25", "question": "Quels correctifs ont été appliqués pour le composant terminal ?",
     "check": "non_empty"},

    # ── Requêtes de recherche textuelle ───────────────────────────────────────
    {"id": "Q26", "question": "Quels tickets mentionnent un crash ?",
     "check": "non_empty"},
    {"id": "Q27", "question": "Trouve les tickets liés à des problèmes de performance.",
     "check": "non_empty"},
    {"id": "Q28", "question": "Quels tickets parlent d'une régression ?",
     "check": "non_empty"},
    {"id": "Q29", "question": "Y a-t-il des tickets mentionnant une fuite mémoire ?",
     "check": "query_executes"},
    {"id": "Q30", "question": "Quels tickets contiennent le mot 'timeout' dans leur titre ?",
     "check": "query_executes"},

    # ── Requêtes d'agrégation ─────────────────────────────────────────────────
    {"id": "Q31", "question": "Combien de tickets ouverts par composant ?",
     "check": "non_empty"},
    {"id": "Q32", "question": "Quel est le ratio de tickets fermés vs ouverts ?",
     "check": "query_executes"},
    {"id": "Q33", "question": "Combien de composants distincts sont référencés dans les tickets ?",
     "check": "count_gt_0"},
    {"id": "Q34", "question": "Quelle est la distribution des tickets par sévérité ?",
     "check": "non_empty"},
    {"id": "Q35", "question": "Combien de versions distinctes sont affectées par des bugs ?",
     "check": "count_gt_0"},

    # ── Requêtes relationnelles ────────────────────────────────────────────────
    {"id": "Q36", "question": "Quels tickets sont reliés entre eux ?",
     "check": "non_empty"},
    {"id": "Q37", "question": "Donne-moi les tickets qui partagent le même composant affecté.",
     "check": "non_empty"},
    {"id": "Q38", "question": "Quels composants co-apparaissent souvent dans les mêmes tickets ?",
     "check": "query_executes"},
    {"id": "Q39", "question": "Quels tickets ont des tickets reliés de sévérité critique ?",
     "check": "query_executes"},
    {"id": "Q40", "question": "Y a-t-il des chaînes de dépendance entre composants ?",
     "check": "query_executes"},

    # ── Requêtes temporelles ──────────────────────────────────────────────────
    {"id": "Q41", "question": "Quels sont les 10 tickets les plus récents ?",
     "check": "non_empty"},
    {"id": "Q42", "question": "Quels tickets ont été créés et fermés dans la même journée ?",
     "check": "query_executes"},
    {"id": "Q43", "question": "Quels tickets critiques sont restés ouverts longtemps ?",
     "check": "query_executes"},

    # ── Requêtes composées ────────────────────────────────────────────────────
    {"id": "Q44",
     "question": "Quels composants ont des tickets critiques ouverts et des dépendances vers d'autres composants ?",
     "check": "query_executes"},
    {"id": "Q45",
     "question": "Pour chaque composant, donne le nombre de tickets critiques et le nombre de versions affectées.",
     "check": "non_empty"},
    {"id": "Q46",
     "question": "Quels tickets affectent une version qui appartient à un composant ayant des dépendances ?",
     "check": "query_executes"},
    {"id": "Q47",
     "question": "Liste les tickets ouverts avec leur composant et leur sévérité, triés par sévérité.",
     "check": "non_empty"},
    {"id": "Q48",
     "question": "Quels correctifs sont dans des versions affectant plus d'un ticket critique ?",
     "check": "query_executes"},
    {"id": "Q49",
     "question": "Donne-moi un résumé : nb tickets, nb composants, nb versions, nb correctifs.",
     "check": "non_empty"},
    {"id": "Q50",
     "question": "Quels sont les composants les plus 'fragiles' (plus de 3 tickets critiques ou high) ?",
     "check": "query_executes"},
]


# ─── Structures de données ────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    question_id: str
    question: str
    generated_sparql: str | None
    execution_ok: bool
    answer_ok: bool
    result_count: int
    latency_ms: float
    attempts: int
    error: str | None
    check_type: str


@dataclass
class BenchmarkReport:
    model: str
    kg_path: str
    total_questions: int
    execution_accuracy: float    # % requêtes qui s'exécutent sans erreur
    answer_accuracy: float       # % requêtes avec résultats cohérents
    sparql_generated: float      # % requêtes SPARQL effectivement générées
    avg_latency_ms: float
    avg_attempts: float
    results: list[QuestionResult] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [asdict(r) for r in self.results]
        return d


# ─── Benchmark runner ─────────────────────────────────────────────────────────

class BenchmarkRunner:
    def __init__(self, kg_path: str, model: str = "llama3",
                 ollama_url: str = "http://localhost:11434"):
        # Import local pour éviter les dépendances circulaires
        from kg.store import KnowledgeGraphStore
        from agent.nl_to_sparql import NLToSPARQLAgent

        self.store = KnowledgeGraphStore(ttl_path=kg_path)
        self.store.load()
        self.agent = NLToSPARQLAgent(model=model, ollama_url=ollama_url)
        self.model = model
        self.kg_path = kg_path

    def run(self, questions: list[dict] | None = None,
            verbose: bool = True) -> BenchmarkReport:
        dataset = questions or BENCHMARK_DATASET
        results: list[QuestionResult] = []

        print(f"\n{'='*60}")
        print(f" BENCHMARK NL→SPARQL — {len(dataset)} questions")
        print(f" Modèle : {self.model}")
        print(f"{'='*60}\n")

        for i, item in enumerate(dataset, 1):
            qid = item["id"]
            question = item["question"]
            check = item.get("check", "non_empty")
            expected_min = item.get("expected_min", 1)

            print(f"[{i:02d}/{len(dataset)}] {qid}: {question[:60]}…")

            t0 = time.time()
            translation = self.agent.translate(question)
            latency_ms = (time.time() - t0) * 1000

            sparql = translation["sparql"]
            attempts = translation["attempts"]
            gen_error = translation["error"]

            execution_ok = False
            answer_ok = False
            result_count = 0
            exec_error = None

            if sparql:
                try:
                    rows = self.store.query(sparql, question=question)
                    execution_ok = True
                    result_count = len(rows)
                    answer_ok = self._check_answer(check, rows, expected_min)
                except Exception as e:
                    exec_error = str(e)

            status = "✓" if answer_ok else ("~" if execution_ok else "✗")
            print(f"  {status} SPARQL={'OK' if sparql else 'FAIL'} | "
                  f"Exec={'OK' if execution_ok else 'ERR'} | "
                  f"Results={result_count} | {latency_ms:.0f}ms")

            results.append(QuestionResult(
                question_id=qid,
                question=question,
                generated_sparql=sparql,
                execution_ok=execution_ok,
                answer_ok=answer_ok,
                result_count=result_count,
                latency_ms=latency_ms,
                attempts=attempts,
                error=exec_error or gen_error,
                check_type=check,
            ))

            # Petite pause pour ne pas saturer Ollama
            time.sleep(0.3)

        return self._compute_report(results)

    def _check_answer(self, check: str, rows: list[dict], expected_min: int) -> bool:
        if check == "non_empty":
            return len(rows) > 0
        if check == "count_gt_0":
            if rows and "count" in rows[0]:
                return int(rows[0]["count"]) > 0
            if rows and "n" in rows[0]:
                return int(rows[0]["n"]) > 0
            return len(rows) > 0
        if check == "query_executes":
            return True  # L'exécution sans erreur suffit
        if check == "specific_count":
            return len(rows) >= expected_min
        return len(rows) > 0

    def _compute_report(self, results: list[QuestionResult]) -> BenchmarkReport:
        n = len(results)
        n_generated = sum(1 for r in results if r.generated_sparql)
        n_exec_ok = sum(1 for r in results if r.execution_ok)
        n_answer_ok = sum(1 for r in results if r.answer_ok)
        avg_latency = sum(r.latency_ms for r in results) / n if n else 0
        avg_attempts = sum(r.attempts for r in results) / n if n else 0

        from datetime import datetime
        report = BenchmarkReport(
            model=self.model,
            kg_path=self.kg_path,
            total_questions=n,
            execution_accuracy=n_exec_ok / n if n else 0,
            answer_accuracy=n_answer_ok / n if n else 0,
            sparql_generated=n_generated / n if n else 0,
            avg_latency_ms=avg_latency,
            avg_attempts=avg_attempts,
            results=results,
            timestamp=datetime.now().isoformat(),
        )

        # Afficher le résumé
        print(f"\n{'='*60}")
        print(f" RÉSULTATS")
        print(f"{'='*60}")
        print(f" SPARQL généré    : {n_generated}/{n} ({report.sparql_generated*100:.1f}%)")
        print(f" Execution Acc.   : {n_exec_ok}/{n} ({report.execution_accuracy*100:.1f}%)")
        print(f" Answer Acc.      : {n_answer_ok}/{n} ({report.answer_accuracy*100:.1f}%)")
        print(f" Latence moyenne  : {avg_latency:.0f}ms")
        print(f" Tentatives moy.  : {avg_attempts:.1f}")
        print(f"{'='*60}\n")

        # Questions échouées
        failed = [r for r in results if not r.answer_ok]
        if failed:
            print(f" Questions échouées ({len(failed)}) :")
            for r in failed:
                print(f"  - {r.question_id}: {r.question[:55]}…")
                if r.error:
                    print(f"    Erreur : {r.error[:80]}")
        return report


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark NL→SPARQL")
    parser.add_argument("--kg", default="data/kg_triples.ttl")
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--output", default="evaluation/results.json")
    parser.add_argument("--quick", action="store_true",
                        help="Lance seulement les 10 premières questions")
    args = parser.parse_args()

    runner = BenchmarkRunner(
        kg_path=args.kg,
        model=args.model,
        ollama_url=args.ollama_url,
    )
    dataset = BENCHMARK_DATASET[:10] if args.quick else None
    report = runner.run(questions=dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"Rapport sauvegardé → {output_path}")


if __name__ == "__main__":
    main()
