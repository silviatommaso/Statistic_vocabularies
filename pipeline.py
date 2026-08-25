import pandas as pd
from pathlib import Path

from utils import is_numeric, is_date, parse_title, split_attributes, save_vocabulary_by_tag, save_code_measure
from clustering.clustering import build_clusters


"""
Constructs the temporal, string, geographic, and vocabulary sets
for each CSV table in the Eurostat dataset.

D_t[file_name] = set of temporal attributes appearing on the top
                 line of table t (e.g., {2000, 2001, 2002, ...})

S_t[file_name] = set of distinct string attributes and values
                 appearing in the leftmost (non-temporal) columns
                 of table t

Geo_t[file_name] = subset of S_t[file_name] containing values
                   identified as geographic units through the
                   Eurostat geographic vocabulary

V_t[file_name] = vocabulary of table t, represented as a dictionary
                 mapping each term to one or more tags describing
                 its role in the table:
                 - N: attribute name
                 - A: dimension value
                 - U: unit information
                 - M: measure extracted from the table title

V = union of V_t over all tables. Each distinct term is associated
    with the union of all tags assigned to it across the tables,
    preserving cases in which the same term has different roles.
"""


TABLES = Path("input/tables")
AUXILIAR_FILES = Path("input/auxiliar_files")

OUTPUT = Path("output")
OUTPUT.mkdir(parents=True, exist_ok=True)


def vocaboulary_set():

    directory = TABLES / "eurostat_2000_tables"

    nuts = pd.read_csv( AUXILIAR_FILES / "ESTAT_GEO_28.0_EN.tsv", sep="\t")
    tab_titles = pd.read_csv( TABLES / "table_titles.csv", sep=",")

    dict_nuts = set(nuts.astype(str).stack())
    titles_dict = dict(zip(tab_titles.iloc[:, 0], tab_titles.iloc[:, 1]))

    D_t = {}
    S_t = {}
    V_t = {}
    Geo_t = {}

    M = {}

    csv_files = [file for file in sorted(directory.iterdir()) if file.suffix == ".csv"]
    total_files = len(csv_files)

    print("Costruzione del vocabolario iniziata...")

    for i, file in enumerate(csv_files, start=1):

        tables = pd.read_csv(file, low_memory=False)
        attributes = tables.columns.tolist()

        string_attributes, temporal_attributes = split_attributes(attributes)

        # title
        title_remaining = parse_title(titles_dict[file.name], dict_nuts)

        # 1. Temporal attributes
        D_t[file.name] = temporal_attributes

        S_t[file.name] = set()
        Geo_t[file.name] = set()
        V_t[file.name] = {}


        for attribute in string_attributes:

            # 2.1 Attribute name -> N
            S_t[file.name].add(attribute)

            if attribute not in V_t[file.name]:
                V_t[file.name][attribute] = set()

            V_t[file.name][attribute].add("N")

            values = tables[attribute].dropna()

            for value in values:

                if not isinstance(value, str) or is_numeric(value) or is_date(value):
                    continue

                # 2.2 String value
                S_t[file.name].add(value)

                # 3.1 Geographic value
                if value in dict_nuts:
                    Geo_t[file.name].add(value)
                    continue

                # 3.2 Non-geographic value -> V(t) = S(t) \ Geo(t)
                if value not in V_t[file.name]:
                    V_t[file.name][value] = set()

                if attribute in {"Unit of measure", "Time frequency"}:
                    V_t[file.name][value].add("U")
                else:
                    V_t[file.name][value].add("A")

        # 4. Remaining title
        if title_remaining:

            # Store the measure with its M tag
            if title_remaining not in V_t[file.name]:
                V_t[file.name][title_remaining] = set()
            V_t[file.name][title_remaining].add("M")

            # Store the files associated with the measure
            if title_remaining not in M:
                M[title_remaining] = set()
            M[title_remaining].add(file.stem)

            (OUTPUT / "vocabulary_by_tag/measures").mkdir(parents=True, exist_ok=True)

        percentage = (i / total_files) * 100
        print(f"\rVocabulary construction: {percentage:.1f}% of completion", end="", flush=True)

    print("\nVocabulary construction completed.")

    # 5-6. Global mapped vocabulary
    V = {}

    for vocabulary in V_t.values():
        for term, tags in vocabulary.items():
            if term not in V:
                V[term] = set()
            V[term].update(tags)


    save_vocabulary_by_tag(V, OUTPUT / "vocabulary_by_tag")
    save_code_measure(M, OUTPUT / "vocabulary_by_tag")

    # 7. Domain clustering
    CLUSTER = OUTPUT / "clustering"
    CLUSTER.mkdir(parents=True, exist_ok=True)

    print("Domain clustering started...")

    build_clusters(OUTPUT / "vocabulary_by_tag/measures/code_measures.csv", CLUSTER)




if __name__ == "__main__":
    vocaboulary_set()