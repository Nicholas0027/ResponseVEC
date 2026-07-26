#!/usr/bin/env python3
"""Build cached, label-free prompts for SurveyBridge item structuring.

Prompts contain question and option text only. They never contain respondent
answers, target response distributions, fitted loadings, model performance, or
factor labels learned from outcomes. Two paraphrases support a stability gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


TEMPLATES = {
    "v1": """Analyze this survey item as a measurement instrument, not as a respondent.

Item ID: {item_id}
Question: {question}
Response options:
{options}

Return exactly one JSON object matching the supplied schema. Identify the item type,
up to three substantive domains, up to four latent constructs, the response scale,
and option-level stance direction and behavior intensity. Direction is relative
within this item: -1 and +1 are opposite substantive ends, not political labels.
Use high uncertainty when direction is not meaningful (nominal/factual options).
Do not predict how common an answer is. Do not mention any respondent.

JSON schema:
{schema}
""",
    "v2": """You are coding a questionnaire for a cross-instrument psychometric model.

Variable: {item_id}
Wording: {question}
Choices:
{options}

Produce only valid JSON under the schema below. Describe what is measured and how
each choice moves along the item's substantive or behavioral direction. Do not
estimate population frequencies, do not simulate a person, and do not infer from
any observed answers. Mark ambiguous or nominal direction with high uncertainty.

Schema:
{schema}
"""
}


def cache_key(item_id: str, question: str, options: list[dict], template: str) -> str:
    payload = json.dumps({"item": item_id, "question": question,
                          "options": options, "template": template},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", default="cross_survey/metadata/ces2020_all_item_options.csv")
    parser.add_argument("--schema", default="cross_survey/metadata/surveybridge_item_schema.json")
    parser.add_argument("--output", default="cross_survey/metadata/surveybridge_item_prompts.jsonl")
    args = parser.parse_args()
    frame = pd.read_csv(args.options)
    schema = json.loads(Path(args.schema).read_text())
    # Compact schema keeps prompts affordable while retaining exact constraints.
    schema_text = json.dumps(schema, separators=(",", ":"))
    records = []
    for item, group in frame.groupby("item", sort=True):
        group = group.sort_values("option_position")
        question = str(group.question_text.iloc[0])
        options = [{"code": int(row.option_code), "label": str(row.option_label)}
                   for row in group.itertuples(index=False)]
        option_text = "\n".join(f"- [{x['code']}] {x['label']}" for x in options)
        for template, text in TEMPLATES.items():
            prompt = text.format(item_id=item, question=question,
                                 options=option_text, schema=schema_text)
            records.append({
                "item": item,
                "wave": str(group.wave.iloc[0]),
                "family": str(group.family.iloc[0]),
                "template": template,
                "cache_key": cache_key(item, question, options, template),
                "prompt": prompt,
            })
    output = Path(args.output)
    output.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    print(f"wrote {len(records)} prompts for {frame.item.nunique()} items to {output}")


if __name__ == "__main__":
    main()
