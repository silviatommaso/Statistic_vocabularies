import pandas as pd
from pathlib import Path

from utils import is_numeric, parse_title, split_attributes, strip_time_marker, save_vocabulary_by_tag
from clustering.clustering import build_clusters
from clustering.similarity_forest.build_forest import export_forest_csv


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


def vocabulary_set():

    directory = TABLES / "eurostat_2000_tables"

    nuts = pd.read_csv( AUXILIAR_FILES / "ESTAT_GEO_28.0_EN.tsv", sep="\t")
    tab_titles = pd.read_csv( TABLES / "table_titles.csv", sep=",")

    dict_nuts = set(nuts.astype(str).stack())
    titles_dict = dict(zip(tab_titles.iloc[:, 0], tab_titles.iloc[:, 1]))

    D_t = {}
    S_t = {}
    V_t = {}
    Geo_t = {}

    csv_files = [file for file in sorted(directory.iterdir()) if file.suffix == ".csv"]
    total_files = len(csv_files)

    print("Costruzione del vocabolario iniziata...")

    for i, file in enumerate(csv_files, start=1):

        print(f"__{file.name}")
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

            cleaned_attribute = strip_time_marker(attribute)

            # 2.1 Attribute name -> N
            S_t[file.name].add(cleaned_attribute)

            if cleaned_attribute not in V_t[file.name]:
                V_t[file.name][cleaned_attribute] = set()
            V_t[file.name][cleaned_attribute].add("N")

            values = tables[attribute].dropna()

            for value in values:

                if not isinstance(value, str) or is_numeric(value):
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


        percentage = (i / total_files) * 100
        print(f"\rVocabulary construction: {percentage:.1f}% of completion", end="", flush=True)

    print("\nVocabulary construction completed.")

    # 5-6. Global mapped vocabulary
    V = {}
    
    for vocabulary in V_t.values():
        for term, tags in vocabulary.items():
            V.setdefault(term, set()).update(tags)

    # ora il collapse va fatto sull'unione globale, non per-tabella
    for term, tags in V.items():
        if tags == {"U", "A"}:
            V[term] = {"U"}


    (OUTPUT / "vocabulary_by_tag").mkdir(parents=True, exist_ok=True)
    save_vocabulary_by_tag(V, OUTPUT / "vocabulary_by_tag")


    # 8. Extract hierarchical relationships between measures and build a tree forest
    FOREST = Path("clustering/similarity_forest")
    print("Building tree-based clusters...")

    export_forest_csv(
        OUTPUT / "vocabulary_by_tag/measures.csv",
        FOREST / "forest_export.csv",
        FOREST / "roots.csv",
        params={"max_split_depth": None, "cos_sim_threshold": 0.65}
    )


    # 7. Domain clustering
    CLUSTER = OUTPUT / "clustering"
    CLUSTER.mkdir(parents=True, exist_ok=True)

    print("Domain clustering started...")

    build_clusters(OUTPUT / "vocabulary_by_tag/measures.csv", FOREST / "forest_export.csv", CLUSTER)



if __name__ == "__main__":
    vocabulary_set()