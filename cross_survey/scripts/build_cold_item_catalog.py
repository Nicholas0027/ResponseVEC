#!/usr/bin/env python3
"""Build option-level, family-disjoint cold-target catalogue from CES PDFs.

Only fixed-text, item-specific post-wave questions with adequate coverage and
entropy are retained. Dynamic candidate-name items, implicit parent fallbacks,
and near-degenerate targets are excluded before any cold-item model is fit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


POST_OPTION_RE = re.compile(
    r"(\d+)\s*○\s*(.*?)(?=\s+\d+\s*○|\s+\d+\s+(?:Skipped|Not Asked)|$)",
    re.IGNORECASE,
)
PRE_OPTION_RE = re.compile(
    r"[◯○]\s*\[(\d+)\]\s*(.*?)(?=\s*[◯○]\s*\[\d+\]|\s*\[CC20_|$)",
    re.IGNORECASE,
)
BRACKET_OPTION_RE = re.compile(
    r"\[(\d+)\]\s*(.*?)(?=\s*\[\d+\]|\s*\[CC20_|$)", re.IGNORECASE
)


def parse_values(value) -> list[int]:
    parsed = ast.literal_eval(str(value))
    return [int(float(x)) for x in parsed]


def local_block(item: str, snippet: str) -> str:
    positions = [m.start() for m in re.finditer(re.escape(item), snippet, re.I)]
    # The first occurrence is the definition/row label. Later occurrences are
    # commonly branch conditions in the following question.
    return snippet[positions[0]:] if positions else snippet


def option_map(item: str, snippet: str, observed: list[int]) -> dict[int, str]:
    block = local_block(item, snippet)
    found = {}
    for search_text, pattern in (
        (block, POST_OPTION_RE), (block, PRE_OPTION_RE),
        # Grid column labels often precede the item row, so inspect the whole
        # family snippet as a fallback.
        (snippet, PRE_OPTION_RE), (snippet, BRACKET_OPTION_RE),
    ):
        for code, label in pattern.findall(search_text):
            code = int(code)
            label = re.sub(r"\s+", " ", label).strip()
            if code in observed and code not in found and label:
                found[code] = label
    # Grid snippets can be truncated before the last column. CES uses a fixed
    # five-point agreement scale for the 440/441 blocks; recover it only when
    # the wording and observed codes jointly make the mapping unambiguous.
    if observed == [1, 2, 3, 4, 5] and "agree or disagree" in snippet.lower():
        found.update({
            1: "Strongly agree", 2: "Somewhat agree",
            3: "Neither agree nor disagree", 4: "Somewhat disagree",
            5: "Strongly disagree",
        })
    if observed == [1, 2] and "support or oppose" in snippet.lower():
        found.update({1: "Support", 2: "Oppose"})
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="cross_survey/metadata/ces2020_question_catalog.csv")
    parser.add_argument("--item-catalog", default="cross_survey/metadata/ces2020_item_catalog.csv")
    parser.add_argument("--output", default="cross_survey/metadata/ces2020_cold_item_options.csv")
    parser.add_argument("--manifest", default="cross_survey/metadata/ces2020_cold_item_folds.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--waves", nargs="*", default=["target_post"])
    args = parser.parse_args()
    catalog = pd.read_csv(args.catalog)
    support = pd.read_csv(args.item_catalog)[["item", "values"]]
    catalog = catalog.merge(support, on="item", how="left", validate="one_to_one")
    items = catalog[
        catalog.wave.isin(args.waves)
        & catalog.found
        & catalog.item_specific_text.fillna(False)
        & ~catalog.dynamic_text.fillna(False)
        & (catalog.paired_coverage >= 0.85)
        & (catalog.entropy_nats >= 0.25)
    ].copy()
    items["observed_codes"] = items["values"].map(parse_values)

    # Build family-level option dictionaries first; truncated PDF snippets for
    # one grid row can be completed from a sibling row with the identical scale.
    family_options: dict[str, dict[int, str]] = {}
    for row in items.itertuples(index=False):
        mapping = option_map(row.item, str(row.raw_snippet), row.observed_codes)
        current = family_options.setdefault(row.family, {})
        for code, label in mapping.items():
            if len(label) > len(current.get(code, "")):
                current[code] = label

    option_rows, excluded = [], {}
    retained_items = []
    for row in items.itertuples(index=False):
        mapping = option_map(row.item, str(row.raw_snippet), row.observed_codes)
        mapping = {**family_options.get(row.family, {}), **mapping}
        missing = [code for code in row.observed_codes if code not in mapping]
        if missing:
            excluded[row.item] = f"missing option labels for observed codes {missing}"
            continue
        retained_items.append(row.item)
        for position, code in enumerate(row.observed_codes):
            option_rows.append({
                "item": row.item,
                "family": row.family,
                "wave": row.wave,
                "question_text": str(row.proxy_text),
                "option_code": code,
                "option_label": mapping[code],
                "option_position": position,
                "n_options": len(row.observed_codes),
                "option_text": (
                    f"Question: {row.proxy_text} Response option: {mapping[code]}"
                ),
                "paired_coverage": float(row.paired_coverage),
                "entropy_nats": float(row.entropy_nats),
            })
    options = pd.DataFrame(option_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    options.to_csv(output, index=False)

    target_options = options[options.wave.eq("target_post")]
    families = sorted(target_options.family.unique())
    # Hash order then round-robin balances family count while remaining fully
    # deterministic and independent of response outcomes.
    families = sorted(
        families,
        key=lambda x: hashlib.sha256(f"{args.seed}|{x}".encode()).digest(),
    )
    fold_map = {family: index % args.folds for index, family in enumerate(families)}
    payload = {
        "dataset": "ces2020",
        "post_freeze": True,
        "seed": args.seed,
        "n_folds": args.folds,
        "retained_items": target_options.item.drop_duplicates().tolist(),
        "retained_item_count": int(target_options.item.nunique()),
        "all_option_bank_items": retained_items,
        "all_option_bank_item_count": len(retained_items),
        "retained_families": families,
        "family_to_fold": fold_map,
        "excluded": excluded,
        "filters": {
            "item_specific_text": True,
            "dynamic_text": False,
            "minimum_coverage": 0.85,
            "minimum_entropy_nats": 0.25,
            "all_observed_option_labels_required": True
        },
        "warning": "Preliminary cold-item folds; question proxy text still contains PDF formatting artifacts."
    }
    manifest = Path(args.manifest)
    manifest.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "items": len(retained_items), "families": len(families),
        "options": len(options), "excluded_for_options": excluded,
        "fold_sizes": pd.Series(fold_map).value_counts().sort_index().to_dict(),
    }, indent=2))
    print(f"wrote {output}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
