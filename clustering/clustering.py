import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from clustering.llm_requests import prompt
from clustering.utils import normalize, word_tokens


"""
Tokenize a title using the EXACT same pipeline used to build the
forest's full_path (normalize() then word_tokens()), so that matching
a measure back to its tree never diverges from how the tree itself
was constructed.
"""
def _normalize_text(text):
    return tuple(word_tokens(normalize(text)))


"""
Load the tree forest (one row per node) and build:

- measure_to_tree: normalized measure tokens -> tree_id, for every node
    where a measure terminates exactly on that node's path
    (n_terminal_here >= 1). This is what lets us map an original
    'measure' string back to the tree it belongs to.
- tree_root_prefix: tree_id -> root edge_label (depth 0), kept only as a
    human-readable label in the output, not used for clustering itself.
"""
def load_forest(forest_path):
    
    forest = pd.read_csv(forest_path)

    measure_to_tree = {}
    tree_root_prefix = {}

    for _, row in forest.iterrows():

        if row["depth"] == 0:
            tree_root_prefix[row["tree_id"]] = row["edge_label"]

        if row["n_terminal_here"] >= 1:
            key = tuple(str(row["full_path"]).split(" > "))
            measure_to_tree[key] = row["tree_id"]

    return measure_to_tree, tree_root_prefix


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
Assign each measure to the tree it belongs to.

Clusters are derived directly from the forest: a cluster IS a tree_id.
No code join is needed anymore -- the input is just a list of measures
(e.g. the same measures.csv used to build the forest itself). A measure
is assigned to a tree by tokenizing its title the same way the forest was
built (lowercase, alphanumeric words) and looking up the resulting
sequence among the nodes where a measure terminates (n_terminal_here >= 1).
"""
def build_cluster_prefix(measures_path, forest_path, llm_files_path, measure_col="term"):

    measures_df = pd.read_csv(measures_path, dtype=str)
    measures_df = measures_df.rename(columns={measure_col: "measure"})

    measure_to_tree, tree_root_prefix = load_forest(forest_path)

    clusters, prefixes = [], []
    unmatched = 0

    for measure in measures_df["measure"]:
        tree_id = measure_to_tree.get(_normalize_text(str(measure)))

        if tree_id is None:
            unmatched += 1
            clusters.append(pd.NA)
            prefixes.append(pd.NA)
        else:
            clusters.append(tree_id)
            prefixes.append(tree_root_prefix.get(tree_id))

    if unmatched:
        print(f"[build_cluster_prefix] {unmatched} measure(s) did not match any tree node and were left unclustered.")

    measures_df["cluster"] = clusters
    measures_df["prefix"] = prefixes

    cluster_prefixes = measures_df[["cluster", "prefix", "measure"]].copy()
    cluster_prefixes = cluster_prefixes.dropna(subset=["cluster"])
    cluster_prefixes["cluster"] = cluster_prefixes["cluster"].astype(int)
    cluster_prefixes = cluster_prefixes.sort_values(["cluster", "measure"]).reset_index(drop=True)

    cluster_prefixes.to_csv(llm_files_path / "cluster_prefixes.csv", index=False)

    return cluster_prefixes


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
Build a list of unique meaningful words for each cluster.
(unchanged in spirit: still groups by "cluster" and reads "measure" --
agnostic to how the cluster id was computed. Fixed: the file is now
written once after the loop, not on every iteration.)
"""
def build_cluster_words(cluster_prefixes, llm_files_path):

    cluster_rows = []

    for cluster_id, group in cluster_prefixes.groupby("cluster", sort=True):

        words = []
        seen_words = set()

        for measure in group["measure"].dropna():

            measure_words = re.findall(r"[A-Za-z]+", measure)

            for word in measure_words:
                word_normalized = word.lower()
                if word_normalized in ENGLISH_STOP_WORDS:
                    continue
                if word_normalized not in seen_words:
                    seen_words.add(word_normalized)
                    words.append(word)

        cluster_rows.append({"cluster": cluster_id, "words": words})

    cluster_words = pd.DataFrame(cluster_rows, columns=["cluster", "words"])
    cluster_words["cluster"] = cluster_words["cluster"].astype(int)
    cluster_words = cluster_words.sort_values("cluster").reset_index(drop=True)

    cluster_words.to_csv(llm_files_path / "cluster_words.csv", index=False)

    return cluster_rows


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
Merge the per-cluster domain assignment back onto individual measures,
producing a direct measure -> domain mapping. No code involved anywhere.
"""
def assign_domains_to_measures(cluster_domains, cluster_prefixes, output_path):
    merged = cluster_prefixes.merge(cluster_domains, on="cluster", how="left")
    result = merged[["measure", "domain"]].copy()
    result.to_csv(output_path, index=False)
    return result


####################################################################################################################################################################################################

"""
Build table clusters (from the tree forest) and assign a domain to each
measure directly.

Parameters
----------
measures_path : pathlib.Path
    CSV with the list of measures (e.g. the same file used to build the forest).
forest_path : pathlib.Path
    CSV with the tree forest (tree_id, node_id, parent_id, depth, edge_label,
    full_path, n_children, n_terminal_here, n_measures_subtree, is_branch_point, examples).
output_path : pathlib.Path
    Path where the final measure-to-domain associations are saved.
measure_col : str
    Name of the column holding the measure text in measures_path (default: "term").
"""
def build_clusters(measures_path, forest_path, output_path, measure_col="term"):

    llm_files_path = Path("clustering/llm_files")
    llm_files_path.mkdir(parents=True, exist_ok=True)

    # 7.1 Build tree-based clusters
    cluster_prefixes = build_cluster_prefix(measures_path, forest_path, llm_files_path, measure_col=measure_col)

    # 7.2 Build word clusters
    cluster_words = build_cluster_words(cluster_prefixes, llm_files_path)

    # 7.3 Prompt LLM
    cluster_domains = prompt(cluster_words, llm_files_path)

    # 7.4 Merge domain assignment back onto individual measures
    assign_domains_to_measures(cluster_domains, cluster_prefixes, output_path / "measure_domain.csv")


# assign_domains_to_measures(
#     pd.read_csv("clustering/llm_files/cluster_domain.csv"),
#     pd.read_csv("clustering/llm_files/cluster_prefixes.csv"),
#     Path("output/clustering/measure_domain.csv")
# )