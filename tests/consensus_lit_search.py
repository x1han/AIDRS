"""Submit two literature queries to the Consensus API for the 3-state polyA + intra-priming design.

Prompt 1 (filtering): polyA filter + intra-priming rescue in direct RNA-seq pipelines
Prompt 2 (biology): distinguishing genuine non-polyadenylated RNAs (histone mRNAs) from
                    degradation intermediates / library artifacts
"""
import os
import json
import requests

API_KEY = "ak_ZPDYMGFJBVF3BPDSM00N0TKZVGFZRAWN"
BASE_URL = "https://api.consensus.app/v1/search"

# Two well-formed queries covering the two science questions behind Stage 2.5b.
QUERIES = {
    "polyA_intra_priming_filter": (
        "direct RNA-seq polyA tail filtering intra-priming A-rich downstream "
        "SQANTI3 rescue nanopore isoform classification"
    ),
    "non_polyA_histone_vs_degradation": (
        "replication-dependent histone mRNA non-polyadenylated genuine transcript "
        "degradation intermediate discrimination direct RNA-seq nanopore"
    ),
}


def run_search(label: str, query: str) -> dict:
    """Submit one query and return the parsed JSON response."""
    headers = {"x-api-key": API_KEY}
    params = {"query": query, "limit": 10}
    resp = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    out_path = "/datf/hanxi/software/AIDRS/repo/tests/consensus_lit_results.json"
    results = {}
    for label, query in QUERIES.items():
        print(f"\n=== Query: {label} ===")
        print(f"Query text: {query}\n")
        try:
            data = run_search(label, query)
            results[label] = {"query": query, "response": data}
            # Brief preview
            for i, paper in enumerate(data.get("results", [])[:5], start=1):
                title = paper.get("title", "(no title)")
                journal = paper.get("journal_name", "(no journal)")
                year = paper.get("publish_year", "?")
                cites = paper.get("citation_count", "?")
                print(f"  [{i}] ({year}) {title}")
                print(f"       {journal}  | citations: {cites}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results[label] = {"query": query, "error": str(exc)}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to: {out_path}")


if __name__ == "__main__":
    main()