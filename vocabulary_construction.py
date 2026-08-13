import pandas as pd
from pathlib import Path

import utils


"""
Constructs the temporal, string, geographic, and vocabulary sets
for each CSV table in the Eurostat dataset.

D_t[file_name] = set of temporal attributes appearing on the top
                 line of table t (e.g., {2000, 2001, 2002, ...})

S_t[file_name] = set of distinct string values appearing in the
                 leftmost (non-temporal) columns of table t

Geo_t[file_name] = subset of S_t[file_name] containing values
                   identified as geographic units through the
                   Eurostat geographic vocabulary

V_t[file_name] = vocabulary of table t, obtained as
                 S(t) \ Geo(t), augmented with the terms remaining
                 in the table title after removing temporal and
                 geographic terms

V = union of V_t over all tables, i.e., the total vocabulary
    of distinct terms appearing in one or more tables
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


        # tables
        tables = pd.read_csv(file)
        attributes = tables.columns.tolist()
        string_attributes, temporal_attributes = utils.split_attributes(attributes)

        # table title
        _, _,title_remaining = utils.parse_title(titles_dict[file.name], dict_nuts)
        


        # 1. Set of time intervals appearing on the top line of each table t
        D_t[file.name] = temporal_attributes

        S_t[file.name] = set()
        Geo_t[file.name] = set()
        V_t[file.name] = set()

        for attribute in string_attributes:
            values = tables[attribute].dropna()
            for value in values:

                # 2. Set of all strings (not numbers) that appear in leftmost column  
                if isinstance(value, str):
                    S_t[file.name].add(value)

                    # 3.1 Units whose names appear in the NUTS geographic dictionary
                    if value in dict_nuts:
                        Geo_t[file.name].add(value)

                    # 3.2 S(t) \ Geo(t): strings that are not geographic values
                    else:
                        V_t[file.name].add(value)

        # 4. Remaining words in the title that are not geographic values
        V_t[file.name].update(title_remaining)


    # 5. Union of all vocabulary sets across all files
    V = set().union(*V_t.values())


    return V


V = vocaboulary_set()