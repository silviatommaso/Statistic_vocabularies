"""
Estrae un campione stratificato di termini dal vocabolario V per la
valutazione manuale dello step 6 (partizione in M / N / A / U).

Assunzioni sui file di input (adatta i path/nomi se diversi):
  output/vocabulary_by_tag/measures.csv
  output/vocabulary_by_tag/dimension_names.csv
  output/vocabulary_by_tag/dimension_values.csv
  output/vocabulary_by_tag/units.csv
ognuno con almeno una colonna "term" contenente i termini con quel tag.
Un termine multi-tag compare in più di uno di questi file.

Output: eval_sample.csv con colonne:
  term, tag_pipeline, tag_gold, note
tag_pipeline puo' contenere piu' tag separati da "|" per i termini multi-tag.
tag_gold e note sono lasciate vuote da compilare a mano.
"""

import random
from pathlib import Path
import pandas as pd

random.seed(42)

VOCAB_DIR = Path("output/vocabulary_by_tag")
TAG_FILES = {
    "M": VOCAB_DIR / "measures.csv",
    "N": VOCAB_DIR / "dimension_names.csv",
    "A": VOCAB_DIR / "dimension_values.csv",
    "U": VOCAB_DIR / "units.csv",
}

N_PER_CATEGORY = 30       # campione stratificato per singolo tag
N_MULTI_TAG = 20          # campione a parte per i termini multi-tag
OUTPUT_FILE = Path("eval_sample.csv")


def load_term_to_tags():
    """Costruisce term -> set di tag leggendo i quattro file."""
    term_to_tags = {}
    for tag, path in TAG_FILES.items():
        if not path.exists():
            print(f"Attenzione: file mancante {path}, salto tag {tag}")
            continue
        terms = pd.read_csv(path)["term"].dropna().astype(str).tolist()
        for term in terms:
            term_to_tags.setdefault(term, set()).add(tag)
    return term_to_tags


def stratified_single_tag_sample(term_to_tags):
    """Per ciascun tag, campiona N termini che hanno SOLO quel tag."""
    rows = []
    by_tag_only = {tag: [] for tag in TAG_FILES}
    for term, tags in term_to_tags.items():
        if len(tags) == 1:
            only_tag = next(iter(tags))
            by_tag_only[only_tag].append(term)

    for tag, terms in by_tag_only.items():
        sample = random.sample(terms, min(N_PER_CATEGORY, len(terms)))
        for term in sample:
            rows.append({"term": term, "tag_pipeline": tag, "tag_gold": "", "note": ""})
    return rows


def multi_tag_sample(term_to_tags):
    """Campiona termini con piu' di un tag (i casi ambigui)."""
    multi_terms = [t for t, tags in term_to_tags.items() if len(tags) > 1]
    sample = random.sample(multi_terms, min(N_MULTI_TAG, len(multi_terms)))
    rows = []
    for term in sample:
        tags_str = "|".join(sorted(term_to_tags[term]))
        rows.append({"term": term, "tag_pipeline": tags_str, "tag_gold": "", "note": ""})
    return rows


def main():
    term_to_tags = load_term_to_tags()
    print(f"Vocabolario totale caricato: {len(term_to_tags)} termini distinti")

    rows = stratified_single_tag_sample(term_to_tags)
    rows += multi_tag_sample(term_to_tags)
    random.shuffle(rows)

    df = pd.DataFrame(rows, columns=["term", "tag_pipeline", "tag_gold", "note"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Campione salvato in {OUTPUT_FILE}: {len(df)} righe "
          f"({sum('|' not in r['tag_pipeline'] for r in rows)} single-tag, "
          f"{sum('|' in r['tag_pipeline'] for r in rows)} multi-tag)")


if __name__ == "__main__":
    main()