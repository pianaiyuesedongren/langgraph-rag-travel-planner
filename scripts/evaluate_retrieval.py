#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

from gaode.rag.retriever import TravelRetriever  # noqa: E402


def evaluate(cases_path: Path, k: int) -> dict[str, float | int]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    retriever = TravelRetriever()
    hits = 0
    reciprocal_rank = 0.0

    for case in cases:
        if case["kind"] == "restaurant":
            documents = retriever.retrieve_restaurants(case["query"], city=case["city"], k=k)
        else:
            documents = retriever.retrieve_attractions(case["query"], city=case["city"], k=k)
        titles = [str(document.metadata.get("name", "")) for document in documents]
        expected = set(case["expected_titles"])
        rank = next(
            (index for index, title in enumerate(titles, start=1) if title in expected),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_rank += 1.0 / rank
        print(
            json.dumps(
                {
                    "query": case["query"],
                    "expected": sorted(expected),
                    "retrieved": titles,
                    "hit": rank is not None,
                    "rank": rank,
                },
                ensure_ascii=False,
            )
        )

    total = len(cases)
    return {
        "cases": total,
        f"recall@{k}": hits / total if total else 0.0,
        "mrr": reciprocal_rank / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate travel knowledge retrieval")
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "evals" / "retrieval_cases.json",
    )
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.cases, args.k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
