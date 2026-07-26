#!/usr/bin/env python3
"""Extract auditable question snippets and text proxies from CES PDFs.

CES questionnaires are not distributed as a machine-readable codebook. Their
PDF text has two layouts (pre uses bracketed variable names; post uses page
blocks). This extractor does not pretend to perfectly parse either. It records
page-local evidence, flags branch/dynamic/multiple-choice risks, and constructs
a conservative text proxy by combining the nearest family stem with the item's
own block. All primary targets still require manual review.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def family(item: str) -> str:
    match = re.match(r"^(CC20_\d+)", item, re.IGNORECASE)
    return match.group(1) if match else item


def parent_candidates(item: str) -> list[str]:
    """Most specific to broadest block IDs for implicit punch variables."""
    candidates = []
    stripped = re.sub(r"_\d+$", "", item)
    if stripped != item:
        candidates.append(stripped)
    base = family(item)
    if base not in candidates and base != item:
        candidates.append(base)
    return candidates


def clean_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()
            if re.sub(r"\s+", " ", line).strip()]


def occurrence(lines: list[str], item: str) -> list[int]:
    patterns = [
        re.compile(rf"^\[{re.escape(item)}\](?:\s|$)", re.I),
        re.compile(rf"^{re.escape(item)}(?:-|\s|$)", re.I),
    ]
    exact = [i for i, line in enumerate(lines) if any(p.search(line) for p in patterns)]
    if exact:
        return exact
    return [i for i, line in enumerate(lines) if item.lower() in line.lower()]


def extract(pages: list[list[str]], item: str) -> dict:
    fam = family(item)
    hits = []
    for page_number, lines in enumerate(pages, start=1):
        for index in occurrence(lines, item):
            hits.append((page_number, index, lines))
    if not hits:
        return {"item": item, "family": fam, "found": False, "proxy_text": ""}

    # Prefer the first definition. References in later branch conditions are
    # usually secondary occurrences.
    page_number, index, lines = hits[0]
    family_indices = [
        i for i, line in enumerate(lines[: index + 1])
        if re.search(rf"(?:^|\[){re.escape(fam)}(?:grid|\]|-|\s|$)", line, re.I)
    ]
    start = max(0, index - 8)
    if family_indices:
        start = max(0, family_indices[-1])
    end = min(len(lines), index + 16)
    snippet_lines = lines[start:end]
    snippet = " ".join(snippet_lines)

    # Strip formatting boilerplate but preserve stems, row labels, options, and
    # branch conditions. The proxy is for retrieval only; the raw snippet stays
    # beside it for audit.
    proxy = re.sub(r"\b(Questionnaire|Page:)\b[^\[]*", " ", snippet, flags=re.I)
    proxy = re.sub(r"\b(varlabel|required|displaymax|collapsible|width):?\s*\w*", " ", proxy, flags=re.I)
    proxy = re.sub(r"\s+", " ", proxy).strip()
    lower = snippet.lower()
    return {
        "item": item,
        "family": fam,
        "found": True,
        "page": page_number,
        "definition_count": len(hits),
        "branch_risk": bool("show if" in lower or "askvote" in lower),
        "randomized": bool("random" in lower),
        "multiple_select": bool("multiple" in lower or "check all" in lower),
        "dynamic_text": bool("$" in snippet),
        "raw_snippet": snippet,
        "proxy_text": proxy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", default="cross_survey/metadata/ces2020_item_catalog.csv"
    )
    parser.add_argument(
        "--raw-root", default="cross_survey/data/raw/ces2020"
    )
    parser.add_argument(
        "--output", default="cross_survey/metadata/ces2020_question_catalog.csv"
    )
    parser.add_argument(
        "--review", default="cross_survey/metadata/ces2020_target_review.md"
    )
    parser.add_argument(
        "--manifest", default="cross_survey/metadata/ces2020_split_manifest.json"
    )
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)
    catalog = catalog[catalog.eligible_closed_choice].copy()
    raw_root = Path(args.raw_root)
    page_cache = {}
    for wave, filename in {
        "source_pre": "CES20_Common_pre_qx.pdf",
        "target_post": "CES20_Common_post_qx.pdf",
    }.items():
        reader = PdfReader(str(raw_root / filename))
        page_cache[wave] = [clean_lines(page.extract_text() or "") for page in reader.pages]

    records = []
    for row in catalog.itertuples(index=False):
        item = str(row.item)
        record = extract(page_cache[row.wave], item)
        if not record["found"]:
            # Multi-select punches such as CC20_420_6 exist as separate CSV
            # columns but only the parent CC20_420 block appears in the PDF.
            # Fall back to the most specific parent and retain an explicit flag
            # so this cannot be mistaken for item-level parsing.
            for parent in parent_candidates(item):
                parent_record = extract(page_cache[row.wave], parent)
                if parent_record["found"]:
                    parent_record["item"] = item
                    parent_record["family"] = family(item)
                    parent_record["proxy_text"] = f"{item} {parent_record['proxy_text']}"
                    parent_record["parent_fallback"] = parent
                    parent_record["item_specific_text"] = False
                    record = parent_record
                    break
        record.setdefault("parent_fallback", None)
        record.setdefault("item_specific_text", True)
        record.update({
            "wave": row.wave,
            "paired_coverage": float(row.paired_coverage),
            "n_unique": int(row.n_unique),
            "entropy_nats": float(row.entropy_nats),
        })
        records.append(record)
    output = pd.DataFrame(records).sort_values(["wave", "family", "item"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    manifest = json.loads(Path(args.manifest).read_text())
    targets = output[output.item.isin(manifest["target_items"])].set_index("item")
    lines = [
        "# CES 2020 preliminary target review",
        "",
        "Generated evidence for manual validity review. Inclusion decisions must",
        "be based on wording/branching validity, never observed model performance.",
        "",
    ]
    for item in manifest["target_items"]:
        row = targets.loc[item]
        flags = [name for name in ("branch_risk", "randomized", "multiple_select", "dynamic_text")
                 if bool(row.get(name, False))]
        lines.extend([
            f"## {item}",
            "",
            f"- Page: {int(row.page) if pd.notna(row.page) else 'not found'}",
            f"- Coverage: {row.paired_coverage:.4f}; categories: {int(row.n_unique)}",
            f"- Automated flags: {', '.join(flags) if flags else 'none'}",
            "- Manual decision: PENDING",
            "- Construct: PENDING",
            "- Reason: PENDING",
            "",
            str(row.raw_snippet),
            "",
        ])
    review_path = Path(args.review)
    review_path.write_text("\n".join(lines), encoding="utf-8")

    found = int(output.found.sum())
    print(f"extracted {found}/{len(output)} eligible items")
    print(f"wrote {output_path}")
    print(f"wrote {review_path}")


if __name__ == "__main__":
    main()
