"""Submit Q5 RT-switching literature query to Consensus API."""
import os
import json
import requests

API_KEY = "ak_ZPDYMGFJBVF3BPDSM00N0TKZVGFZRAWN"
BASE_URL = "https://api.consensus.app/v1/search"

QUERY = (
    "In long-read RNA sequencing (Oxford Nanopore direct RNA and cDNA), "
    "what exact sequence algorithm and microhomology length criteria "
    "(direct repeats at non-canonical splice junctions) are used by SQANTI3 "
    "or other transcriptome curation tools to filter reverse transcriptase "
    "template switching (RT-switching) artifacts?"
)


def main():
    headers = {"x-api-key": API_KEY}
    params = {"query": QUERY, "limit": 10}
    resp = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"Total results: {len(data.get('results', []))}\n")
    for i, paper in enumerate(data.get("results", [])[:5], 1):
        title = paper.get("title", "(no title)")
        journal = paper.get("journal_name", "(no journal)")
        year = paper.get("publish_year", "?")
        cites = paper.get("citation_count", "?")
        doi = paper.get("doi", "?")
        abstract = paper.get("abstract", "")[:300]
        print(f"[{i}] ({year}) {title}")
        print(f"    {journal}  | citations: {cites}  | DOI: {doi}")
        if abstract:
            print(f"    Abstract: {abstract}...")
        print()
    out_path = "/datf/hanxi/software/AIDRS/repo/tests/consensus_rt_switching.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Full results: {out_path}")


if __name__ == "__main__":
    main()