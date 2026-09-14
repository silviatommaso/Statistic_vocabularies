# Statistic Vocabularies

Construction and structuring of a statistical vocabulary from a corpus of
Eurostat tables, developed for the *Statistic Vocabularies* assignment
(Ioana Manolescu, Inria and IP Paris, May 2026).

The pipeline extracts, from each table, its dimension names, dimension
values, unit information, and the measure described by its title;
merges these into a single global vocabulary $\mathcal{V}$, partitioned
into four categories (`M`, `N`, `A`, `U`); groups the resulting measures
into a small number of thematic domains; and identifies semantic
relationships between pairs of measures. The full methodology, design
rationale, and evaluation results are described in
[`Statistic_Vocabularies.pdf`](./Statistic_Vocabularies.pdf).

## Repository structure

```
.
├── pipeline.py                     # entry point: runs the full pipeline
├── utils.py                        # temporal/geographic/title parsing, vocabulary I/O
├── clustering/
│   ├── clustering.py                # measure-to-tree mapping, cluster vocabularies,
│   │                                 # domain propagation (step 7)
│   ├── llm_requests.py              # domain prompt, fixed domain dictionary,
│   │                                 # Groq API calls
│   └── similarity_forest/
│       └── build_forest.py          # similarity forest construction (steps 7 & 8)
├── input/
│   ├── tables/                      # Eurostat CSV tables + table_titles.csv
│   └── auxiliar_files/              # NUTS.tsv (NUTS dictionary)
├── output/
│   ├── vocabulary_by_tag/           # measures.csv, dimension_names.csv,
│   │                                 # dimension_values.csv, units.csv
│   └── clustering/                  # measure_domain.csv
├── quality_evaluation/              # evaluation scripts and ground truth (steps 6-8)
└── Statistic_Vocabularies.pdf       # full report
```

## Requirements

- Python 3.10+
- `pandas`, `scikit-learn`, `sentence-transformers`, `numpy`
- `groq`, `python-dotenv`

```bash
pip install pandas scikit-learn sentence-transformers numpy groq python-dotenv
```

## Setup

1. **API key.** Domain assignment (step 7) queries `openai/gpt-oss-120b`
   via the Groq API. Create a `.env` file in the repository root:

   ```
   API_KEY=your_groq_api_key
   ```

2. **Input data.** Download the Eurostat tables and their titles from the
   links provided in the assignment, and place them as follows:

   | File / folder | Expected location |
   |---|---|
   | Eurostat CSV tables (from [Zenodo](https://zenodo.org/records/15681384)) | `input/tables/eurostat_2000_tables/` |
   | Table titles (`table_titles.csv`) | `input/tables/table_titles.csv` |
   | NUTS geographic dictionary (`NUTS.tsv`) | `input/auxiliar_files/NUTS.tsv` |

## Usage

Run the full pipeline from the repository root:

```bash
python pipeline.py
```

This sequentially:

1. builds the per-table vocabularies $D(t)$, $S(t)$, $Geo(t)$, $V(t)$ and
   merges them into the global vocabulary $\mathcal{V}$, writing
   `output/vocabulary_by_tag/{measures,dimension_names,dimension_values,units}.csv`;
2. constructs the similarity forest over `measures.csv`, writing
   `clustering/similarity_forest/{forest_export.csv,roots.csv}`;
3. clusters measures via the forest and assigns a domain to each cluster
   through the LLM, writing `output/clustering/measure_domain.csv`.

Similarity-forest parameters (`max_split_depth`, `cos_sim_threshold`) are
set where `export_forest_csv()` is called in `pipeline.py`; the
configurations used for domain clustering and for semantic-relationship
evaluation differ and are documented in the report.

## Evaluation

Quality evaluation scripts, ground-truth files, and their outputs live
under `quality_evaluation/`, covering:

- **Step 6** (vocabulary classification): a self-consistency test over
  the four categories (no manual ground truth — see report, §Evaluation,
  for the rationale).
- **Step 7** (domain clustering): classification metrics
  (accuracy, macro precision/recall/F1, per-domain breakdown) against a
  manually labelled sample of measures, plus an upstream/downstream
  error-attribution analysis.
- **Step 8** (semantic relationships): per-tree domain purity and
  relationship-preservation checks under splitting.

Each evaluation script prints its results to console and writes the
corresponding CSV outputs to `quality_evaluation/`.

## Report

The full report — methodology, design rationale for every step,
evaluation results and discussion — is available in
[`Statistic_Vocabularies.pdf`](./Statistic_Vocabularies.pdf).

## Author

Silvia Tommaso
