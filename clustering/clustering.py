import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from clustering.llm_requests import prompt
from utils import normalize_code, assign_domains_to_codes


"""
Build the prefix associated with each code.

Priority:
1. If '$' is present, everything before '$'.
2. Otherwise, everything before the first '_'.
3. If digits are present, everything before the first digit.
"""
def build_prefixes(codes):

    codes = [normalize_code(code) for code in codes]
    code_to_prefix = {}

    for code in codes:
        digit_match = re.search(r"\d", code)

        if "$" in code:
            prefix = code.split("$")[0]
        elif "_" in code:
            prefix = code.split("_", 1)[0]
        elif digit_match:
            prefix = code[:digit_match.start()].rstrip("_")
        else:
            prefix = code

        code_to_prefix[code] = prefix

    return code_to_prefix



def build_cluster_prefix(code_measures, llm_files_path):

    # code normalization
    code_measures["code"] = code_measures["code"].apply(normalize_code)

    # code-prefix mapping
    code_to_prefix = build_prefixes(code_measures["code"])
    code_measures["prefix"] = code_measures["code"].map(code_to_prefix)
    
    # cluster-id mapping
    prefix_to_cluster = {prefix: cluster_id for cluster_id, prefix in enumerate(sorted(code_measures["prefix"].dropna().unique()))}
    code_measures["cluster"] = code_measures["prefix"].map(prefix_to_cluster)

    # save cluster_prefix file
    cluster_prefixes = code_measures[["cluster", "prefix", "code", "measure"]].copy()
    cluster_prefixes = cluster_prefixes.sort_values(["cluster", "code"]).reset_index(drop=True)

    cluster_prefixes.to_csv( llm_files_path / "cluster_prefixes.csv", index=False)

    return cluster_prefixes


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Build a list of unique meaningful words for each cluster.

For every cluster, all the associated measures are processed by:
1. extracting alphabetical words;
2. normalizing words to lowercase for duplicate detection;
3. removing English stop words;
4. keeping only the first occurrence of each word.

The result associates every cluster with a list containing the
unique words extracted from all its measures.
"""
def build_cluster_words(cluster_prefixes, llm_files_path):

    cluster_rows = []
    
    for cluster_id, group in cluster_prefixes.groupby("cluster", sort=True):

        words = []
        seen_words = set()

        for measure in group["measure"].dropna():

            measure_words = re.findall(r"[A-Za-z]+", measure)

            for word in measure_words:
                # normalize words
                word_normalized = word.lower()
                # delete stop words
                if word_normalized in ENGLISH_STOP_WORDS:
                    continue

                if word_normalized not in seen_words:
                    seen_words.add(word_normalized)
                    words.append(word)

        cluster_rows.append({"cluster": cluster_id, "words": words})

        # save cluster words
        cluster_words = pd.DataFrame(cluster_rows, columns=["cluster", "words"])
    
        cluster_words["cluster"] = cluster_words["cluster"].astype(int)
        cluster_words = cluster_words.sort_values("cluster").reset_index(drop=True)

        cluster_words.to_csv(llm_files_path / "cluster_words.csv", index=False)

    return cluster_rows

    
####################################################################################################################################################################################################

"""
Build table clusters and assign a domain to each table.

The clustering process consists of three main steps:
1. Group tables into clusters according to their code prefixes.
2. Extract the unique meaningful words from the measures
    associated with each cluster.
3. Use the LLM to assign one domain to each cluster.

Finally, the domain assigned to each cluster is propagated
to all the tables belonging to that cluster, and the resulting
table-to-domain associations are saved to the specified output path.

Parameters
----------
M : dict
    Dictionary mapping each measure to the set of table codes
    in which the measure occurs.

output_path : pathlib.Path
    Path where the final table-to-domain associations are saved.

Returns
-------
pandas.DataFrame
    DataFrame containing the code and the domain assigned to
    each table.
"""
def build_clusters(M, output_path):

    llm_files_path = Path("clustering/llm_files")
    llm_files_path.mkdir(parents=True, exist_ok=True)

    # 7.1 Build prefix clusters
    cluster_prefixes = build_cluster_prefix(M, llm_files_path)

    # 7.2 Build word clusters
    cluster_words = build_cluster_words(cluster_prefixes, llm_files_path)

    # 7.3 Prompt LLM
    cluster_domains = prompt(cluster_words, llm_files_path)

    result = assign_domains_to_codes(cluster_domains, cluster_prefixes, output_path)

    return result