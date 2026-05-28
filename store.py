"""
kg/store.py
───────────
Charge le graphe de connaissances RDF en mémoire (rdflib) et expose
une interface SPARQL unifiée avec logging des requêtes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdflib import Graph, Namespace
from rdflib.query import Result

logger = logging.getLogger(__name__)

SUP = Namespace("http://support.kg/ontology#")
ENT = Namespace("http://support.kg/entity/")

# Préfixes injectés automatiquement dans chaque requête SPARQL
SPARQL_PREFIXES = """
PREFIX sup:  <http://support.kg/ontology#>
PREFIX ent:  <http://support.kg/entity/>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
"""


@dataclass
class QueryLog:
    question: str
    sparql: str
    results: list[dict]
    duration_ms: float
    error: str | None = None


@dataclass
class KnowledgeGraphStore:
    ttl_path: str
    graph: Graph = field(default_factory=Graph, init=False)
    query_history: list[QueryLog] = field(default_factory=list, init=False)
    _loaded: bool = field(default=False, init=False)

    def load(self) -> "KnowledgeGraphStore":
        path = Path(self.ttl_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier KG introuvable : {path}\n"
                "Lancez d'abord : python scripts/ingest_github.py"
            )
        logger.info(f"Chargement du graphe RDF depuis {path}…")
        t0 = time.time()
        self.graph = Graph()
        self.graph.parse(str(path), format="turtle")
        elapsed = time.time() - t0
        n = len(self.graph)
        logger.info(f"Graphe chargé : {n} triplets en {elapsed:.2f}s")
        self._loaded = True
        return self

    def query(self, sparql: str, question: str = "") -> list[dict[str, Any]]:
        """
        Exécute une requête SPARQL SELECT et retourne une liste de dicts.
        Les préfixes standards sont injectés automatiquement.
        """
        if not self._loaded:
            self.load()

        full_query = SPARQL_PREFIXES + "\n" + sparql
        t0 = time.time()
        error = None
        rows: list[dict] = []

        try:
            result: Result = self.graph.query(full_query)
            rows = [
                {str(var): self._serialize_term(row[var]) for var in result.vars}
                for row in result
            ]
        except Exception as e:
            error = str(e)
            logger.error(f"Erreur SPARQL : {e}\nRequête :\n{full_query}")

        duration_ms = (time.time() - t0) * 1000
        log = QueryLog(
            question=question,
            sparql=sparql,
            results=rows,
            duration_ms=duration_ms,
            error=error,
        )
        self.query_history.append(log)
        logger.debug(f"SPARQL ({duration_ms:.1f}ms) → {len(rows)} résultats")
        return rows

    def stats(self) -> dict[str, int]:
        """Retourne des statistiques sur le graphe chargé."""
        if not self._loaded:
            self.load()
        queries = {
            "tickets":    "SELECT (COUNT(?t) AS ?n) WHERE { ?t a sup:SupportTicket }",
            "components": "SELECT (COUNT(?c) AS ?n) WHERE { ?c a sup:SoftwareComponent }",
            "versions":   "SELECT (COUNT(?v) AS ?n) WHERE { ?v a sup:ProductVersion }",
            "fixes":      "SELECT (COUNT(?f) AS ?n) WHERE { ?f a sup:Fix }",
            "triples":    None,
        }
        result = {}
        for key, q in queries.items():
            if q is None:
                result[key] = len(self.graph)
            else:
                rows = self.query(q)
                result[key] = int(rows[0]["n"]) if rows else 0
        return result

    @staticmethod
    def _serialize_term(term) -> str:
        if term is None:
            return ""
        # URI → fragment lisible
        s = str(term)
        if "#" in s:
            return s.split("#")[-1]
        if "/" in s:
            return s.split("/")[-1]
        return s
