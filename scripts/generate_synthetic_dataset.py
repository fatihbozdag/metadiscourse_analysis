"""Synthetic Dataset Generator for Metadiscourse Training
----------------------------------------------------------------
Creates a CSV file `synthetic_metadiscourse_dataset.csv` containing
positive and negative examples for two tricky categories:
  • self_mentions   (I, we, our …)
  • evidentials     (show, demonstrate, indicate …)

Each record has the following columns expected by the ML pipeline:
    text               – full sentence
    marker_text        – the surface marker being evaluated
    marker_category    – one of ['self_mentions', 'evidentials']
    is_metadiscourse   – boolean label

Usage::
    python scripts/generate_synthetic_dataset.py --size 4000

If the file already exists it will be overwritten.
"""

from __future__ import annotations
import random
import argparse
from pathlib import Path
import pandas as pd

POSITIVE_SELF_TEMPLATES = [
    "In this paper, {pronoun} {verb} that {{claim}}.",
    "Here, {pronoun} {verb} the results of our study.",
    "{pronoun_cap} {verb} to demonstrate that {{claim}}.",
    "In what follows, {pronoun} will {verb} how {{claim}}.",
]
NEGATIVE_SELF_TEMPLATES = [
    "The teacher asked {object_pronoun} to submit the assignment.",
    "Yesterday, the manager met with {object_pronoun} at noon.",
    "John invited {object_pronoun} to the presentation.",
    "The gift was for {object_pronoun}.",
]

POSITIVE_EVIDENTIAL_TEMPLATES = [
    "The results {verb} that {{claim}}.",
    "Table 1 clearly {verb}s the difference between the two groups.",
    "Our data {verb} how {{claim}}.",
    "Figure 2 {verb}s a significant trend.",
]
NEGATIVE_EVIDENTIAL_TEMPLATES = [
    "She {verb}s her paintings at the local gallery every summer.",
    "The guide will {verb} you around the museum.",
    "They {verb} up late to the party.",
    "He {verb}ed his ticket at the entrance.",
]

PRONOUNS = [
    ("I", "me"),
    ("we", "us"),
]
STANCE_VERBS = ["argue", "show", "demonstrate", "suggest", "propose"]
EVIDENTIAL_VERBS = ["show", "demonstrate", "indicate"]

FAKE_CLAIMS = [
    "the proposed method outperforms the baseline",
    "there is a significant correlation between X and Y",
    "the hypothesis holds for all test cases",
    "further research is required to confirm these findings",
]


def synthesize_self_mentions(n: int) -> list[dict]:
    data = []
    half = n // 2
    for _ in range(half):
        pronoun, obj_pron = random.choice(PRONOUNS)
        verb = random.choice(STANCE_VERBS)
        claim = random.choice(FAKE_CLAIMS)
        template = random.choice(POSITIVE_SELF_TEMPLATES)
        sentence = template.format(pronoun=pronoun, pronoun_cap=pronoun.capitalize(), verb=verb).replace("{claim}", claim)
        data.append({
            "text": sentence,
            "marker_text": pronoun,
            "marker_category": "self_mentions",
            "is_metadiscourse": True,
        })

    for _ in range(n - half):
        pronoun, obj_pron = random.choice(PRONOUNS)
        template = random.choice(NEGATIVE_SELF_TEMPLATES)
        sentence = template.format(object_pronoun=obj_pron)
        data.append({
            "text": sentence,
            "marker_text": obj_pron,
            "marker_category": "self_mentions",
            "is_metadiscourse": False,
        })
    return data


def synthesize_evidentials(n: int) -> list[dict]:
    data = []
    half = n // 2
    for _ in range(half):
        verb = random.choice(EVIDENTIAL_VERBS)
        claim = random.choice(FAKE_CLAIMS)
        template = random.choice(POSITIVE_EVIDENTIAL_TEMPLATES)
        sentence = template.format(verb=verb).replace("{claim}", claim)
        data.append({
            "text": sentence,
            "marker_text": verb,
            "marker_category": "evidentials",
            "is_metadiscourse": True,
        })
    for _ in range(n - half):
        verb = random.choice(EVIDENTIAL_VERBS)
        template = random.choice(NEGATIVE_EVIDENTIAL_TEMPLATES)
        sentence = template.format(verb=verb)
        data.append({
            "text": sentence,
            "marker_text": verb,
            "marker_category": "evidentials",
            "is_metadiscourse": False,
        })
    return data


def generate_dataset(size: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate a balanced synthetic dataset."""
    random.seed(seed)
    # 50-50 split between the two categories,
    # internally each function balances positive/negative.
    half = size // 2
    records = synthesize_self_mentions(half) + synthesize_evidentials(size - half)
    random.shuffle(records)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for metadiscourse training")
    parser.add_argument("--size", type=int, default=4000, help="Total number of examples to generate (default 4000)")
    parser.add_argument("--output", type=str, default="synthetic_metadiscourse_dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_dataset(size=args.size)
    df.to_csv(args.output, index=False)
    print(f"Synthetic dataset created: {args.output} (rows: {len(df)})")
    print(df.head())


if __name__ == "__main__":
    main() 