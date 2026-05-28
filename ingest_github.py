"""
ingest_github.py
────────────────
Récupère des issues GitHub réelles et les convertit en triplets RDF
conformes à l'ontologie support_ontology.ttl.

Usage:
    python scripts/ingest_github.py \
        --repo microsoft/vscode \
        --max-issues 200 \
        --output data/kg_triples.ttl \
        [--token GITHUB_TOKEN]
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

# ─── Namespaces ───────────────────────────────────────────────────────────────
SUP = Namespace("http://support.kg/ontology#")
ENT = Namespace("http://support.kg/entity/")


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def safe_uri(text: str) -> str:
    """Transforme un texte en fragment URI valide."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text).strip("_")


def map_severity(labels: list[str]) -> URIRef:
    label_set = {l.lower() for l in labels}
    if any(k in label_set for k in ("critical", "blocker", "p0")):
        return SUP.Critical
    if any(k in label_set for k in ("high", "major", "p1")):
        return SUP.High
    if any(k in label_set for k in ("medium", "moderate", "p2")):
        return SUP.Medium
    return SUP.Low


def extract_components(labels: list[str], title: str, repo: str) -> list[str]:
    """
    Infère les composants depuis les labels et le titre.
    Retourne une liste de noms de composants normalisés.
    """
    components = set()
    component_keywords = [
        "editor", "terminal", "debugger", "extension", "git", "search",
        "notebook", "explorer", "scm", "tasks", "settings", "themes",
        "intellisense", "emmet", "markdown", "json", "typescript", "python",
        "authentication", "network", "ui", "api", "cli", "language-server",
    ]
    text = " ".join(labels).lower() + " " + title.lower()
    for kw in component_keywords:
        if kw in text:
            components.add(kw)
    # Composant générique basé sur le repo
    repo_name = repo.split("/")[-1]
    if not components:
        components.add(repo_name + "-core")
    return list(components)


def extract_version_refs(title: str, body: str) -> list[str]:
    """Extrait les mentions de version depuis le titre et le body."""
    text = (title or "") + " " + (body or "")
    return list(set(re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", text)))


# ─── Ingestion ────────────────────────────────────────────────────────────────

class GitHubIngester:
    BASE_URL = "https://api.github.com"

    def __init__(self, repo: str, token: str | None = None):
        self.repo = repo
        self.headers = {"Accept": "application/vnd.github+json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def fetch_issues(self, max_issues: int = 200) -> list[dict]:
        issues = []
        page = 1
        per_page = min(100, max_issues)
        print(f"[ingest] Récupération des issues de {self.repo}…")
        while len(issues) < max_issues:
            url = f"{self.BASE_URL}/repos/{self.repo}/issues"
            params = {
                "state": "all",
                "per_page": per_page,
                "page": page,
                "sort": "created",
                "direction": "desc",
            }
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            if resp.status_code == 403:
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 1)
                print(f"[ingest] Rate limit — attente {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            # Exclure les pull requests
            issues.extend(i for i in batch if "pull_request" not in i)
            page += 1
            time.sleep(0.5)
        return issues[:max_issues]

    def to_rdf(self, issues: list[dict]) -> Graph:
        g = Graph()
        g.bind("sup", SUP)
        g.bind("ent", ENT)
        g.bind("owl", OWL)
        g.bind("xsd", XSD)

        # Charger l'ontologie
        onto_path = Path(__file__).parent.parent / "ontology" / "support_ontology.ttl"
        if onto_path.exists():
            g.parse(str(onto_path), format="turtle")

        # Mémoriser les composants et versions déjà créés
        known_components: dict[str, URIRef] = {}
        known_versions: dict[str, URIRef] = {}

        def get_component(name: str) -> URIRef:
            if name not in known_components:
                uri = ENT[f"component_{safe_uri(name)}"]
                g.add((uri, RDF.type, SUP.SoftwareComponent))
                g.add((uri, SUP.componentName, Literal(name, datatype=XSD.string)))
                g.add((uri, RDFS.label, Literal(name)))
                known_components[name] = uri
            return known_components[name]

        def get_version(ver: str, component_uri: URIRef) -> URIRef:
            key = f"{ver}_{str(component_uri)}"
            if key not in known_versions:
                uri = ENT[f"version_{safe_uri(ver)}_{safe_uri(str(component_uri))}"]
                g.add((uri, RDF.type, SUP.ProductVersion))
                g.add((uri, SUP.versionNumber, Literal(ver, datatype=XSD.string)))
                g.add((uri, SUP.belongsTo, component_uri))
                g.add((uri, RDFS.label, Literal(f"v{ver}")))
                known_versions[key] = uri
            return known_versions[key]

        print(f"[ingest] Conversion de {len(issues)} issues en RDF…")
        for issue in issues:
            ticket_id = str(issue["number"])
            ticket_uri = ENT[f"ticket_{ticket_id}"]

            # Labels
            label_names = [l["name"] for l in issue.get("labels", [])]

            # ── Ticket ────────────────────────────────────────────────────
            g.add((ticket_uri, RDF.type, SUP.SupportTicket))
            g.add((ticket_uri, SUP.ticketId, Literal(ticket_id, datatype=XSD.string)))
            g.add((ticket_uri, SUP.title, Literal(issue["title"], datatype=XSD.string)))
            g.add((ticket_uri, RDFS.label, Literal(f"#{ticket_id}: {issue['title']}")))

            if issue.get("body"):
                desc = issue["body"][:500]  # Tronquer pour ne pas gonfler le graphe
                g.add((ticket_uri, SUP.description, Literal(desc, datatype=XSD.string)))

            g.add((ticket_uri, SUP.state, Literal(issue["state"], datatype=XSD.string)))

            if label_names:
                g.add((ticket_uri, SUP.labels,
                       Literal(", ".join(label_names), datatype=XSD.string)))

            # Dates
            if issue.get("created_at"):
                g.add((ticket_uri, SUP.createdAt,
                       Literal(issue["created_at"], datatype=XSD.dateTime)))
            if issue.get("closed_at"):
                g.add((ticket_uri, SUP.closedAt,
                       Literal(issue["closed_at"], datatype=XSD.dateTime)))

            # Sévérité
            severity_uri = map_severity(label_names)
            g.add((ticket_uri, SUP.hasSeverity, severity_uri))

            # ── Composants ────────────────────────────────────────────────
            comp_names = extract_components(label_names, issue["title"], self.repo)
            comp_uris = [get_component(c) for c in comp_names]
            for comp_uri in comp_uris:
                g.add((ticket_uri, SUP.affectsComponent, comp_uri))

            # ── Versions ──────────────────────────────────────────────────
            body = issue.get("body") or ""
            version_strs = extract_version_refs(issue["title"], body)
            for ver in version_strs[:3]:  # Max 3 versions par ticket
                for comp_uri in comp_uris[:1]:
                    ver_uri = get_version(ver, comp_uri)
                    g.add((ticket_uri, SUP.affectsVersion, ver_uri))

            # ── Fix (si fermé) ────────────────────────────────────────────
            if issue["state"] == "closed":
                fix_uri = ENT[f"fix_ticket_{ticket_id}"]
                g.add((fix_uri, RDF.type, SUP.Fix))
                g.add((fix_uri, RDFS.label, Literal(f"Fix for #{ticket_id}")))
                g.add((ticket_uri, SUP.fixedBy, fix_uri))
                # Lier le fix à la version si disponible
                if version_strs and comp_uris:
                    fix_ver_uri = get_version(version_strs[0], comp_uris[0])
                    g.add((fix_uri, SUP.resolvedIn, fix_ver_uri))

        # ── Dépendances entre composants (heuristique co-occurrence) ──────────
        print("[ingest] Calcul des co-occurrences de composants…")
        self._add_component_dependencies(g, issues, known_components)

        # ── Tickets reliés (même labels) ──────────────────────────────────────
        self._add_related_tickets(g, issues)

        triples = len(g)
        print(f"[ingest] ✓ Graphe construit : {triples} triplets, "
              f"{len(known_components)} composants, {len(known_versions)} versions.")
        return g

    def _add_component_dependencies(
        self, g: Graph, issues: list[dict], known_components: dict
    ):
        """Ajoute des dépendances entre composants qui co-apparaissent souvent."""
        from collections import defaultdict
        cooc: dict[tuple, int] = defaultdict(int)
        for issue in issues:
            label_names = [l["name"] for l in issue.get("labels", [])]
            comps = extract_components(label_names, issue["title"], self.repo)
            for i, c1 in enumerate(comps):
                for c2 in comps[i + 1:]:
                    key = tuple(sorted([c1, c2]))
                    cooc[key] += 1
        for (c1, c2), count in cooc.items():
            if count >= 2 and c1 in known_components and c2 in known_components:
                g.add((known_components[c1], SUP.dependsOn, known_components[c2]))

    def _add_related_tickets(self, g: Graph, issues: list[dict]):
        """Relie les tickets partageant les mêmes labels."""
        from collections import defaultdict
        label_to_tickets: dict[str, list[str]] = defaultdict(list)
        for issue in issues:
            for label in issue.get("labels", []):
                label_to_tickets[label["name"]].append(str(issue["number"]))
        added = set()
        for label, ticket_ids in label_to_tickets.items():
            for i, t1 in enumerate(ticket_ids[:10]):  # Max 10 par label
                for t2 in ticket_ids[i + 1: 10]:
                    key = tuple(sorted([t1, t2]))
                    if key not in added:
                        uri1 = ENT[f"ticket_{t1}"]
                        uri2 = ENT[f"ticket_{t2}"]
                        g.add((uri1, SUP.relatedTo, uri2))
                        added.add(key)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingestion GitHub → RDF")
    parser.add_argument("--repo", default="microsoft/vscode",
                        help="Dépôt GitHub (ex: microsoft/vscode)")
    parser.add_argument("--max-issues", type=int, default=200)
    parser.add_argument("--output", default="data/kg_triples.ttl")
    parser.add_argument("--token", default=None, help="GitHub personal access token")
    args = parser.parse_args()

    ingester = GitHubIngester(repo=args.repo, token=args.token)
    issues = ingester.fetch_issues(max_issues=args.max_issues)
    g = ingester.to_rdf(issues)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(output_path), format="turtle")
    print(f"[ingest] ✓ Graphe sauvegardé → {output_path}")


if __name__ == "__main__":
    main()
