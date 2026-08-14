import pandas as pd
from pathlib import Path

import utils


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
def vocaboulary_set():

    directory = Path("tables/eurostat_2000_tables")

    nuts = pd.read_csv("tables/ESTAT_GEO_28.0_EN.tsv", sep="\t")
    tab_titles = pd.read_csv("tables/table_titles.csv", sep=",")

    dict_nuts = set(nuts.astype(str).stack())
    titles_dict = dict(zip(tab_titles.iloc[:, 0], tab_titles.iloc[:, 1]))

    D_t = {}
    S_t = {}
    V_t = {}
    Geo_t = {}


    for file in sorted(directory.iterdir()):

        if file.suffix != ".csv":
            continue

        tables = pd.read_csv(file)
        attributes = tables.columns.tolist()

        string_attributes, temporal_attributes = utils.split_attributes(attributes)

        # title
        title_remaining = utils.parse_title(titles_dict[file.name], dict_nuts)

        # 1. Temporal attributes
        D_t[file.name] = temporal_attributes

        S_t[file.name] = set()
        Geo_t[file.name] = set()
        V_t[file.name] = {}


        for attribute in string_attributes:

            # 2. Attribute name -> N
            S_t[file.name].add(attribute)

            if attribute not in V_t[file.name]:
                V_t[file.name][attribute] = set()

            V_t[file.name][attribute].add("N")

            values = tables[attribute].dropna()

            for value in values:

                if not isinstance(value, str):
                    continue

                # 2. String value
                S_t[file.name].add(value)

                # 3. Geographic value
                if value in dict_nuts:
                    Geo_t[file.name].add(value)
                    continue

                # 3. Non-geographic value -> V(t) = S(t) \ Geo(t)
                if value not in V_t[file.name]:
                    V_t[file.name][value] = set()

                if attribute in {"Unit of measure", "Time frequency"}:
                    V_t[file.name][value].add("U")
                else:
                    V_t[file.name][value].add("A")

        # 4. Remaining title
        if title_remaining:
            if title_remaining not in V_t[file.name]:
                V_t[file.name][title_remaining] = set()

            V_t[file.name][title_remaining].add("M")


    # 5. Global vocabulary
    V = {}

    for vocabulary in V_t.values():
        for term, tags in vocabulary.items():
            if term not in V:
                V[term] = set()
            V[term].update(tags)


    return V

print(vocaboulary_set())