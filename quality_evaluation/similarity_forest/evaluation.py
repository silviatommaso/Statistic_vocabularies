import re
import pandas as pd
from pathlib import Path


"""
Tokenizes text the SAME way full_path was built when the forest was
constructed (lowercase, then split on any run of non-alphanumeric
characters). Inferred empirically from forest_export.csv examples,
e.g. "on-call" -> "on > call", "1.1" -> "1 > 1", "C-O" -> "c > o".
This must match the original build_forest.py tokenization exactly, or
measures will silently fail to match any tree.
"""
def tokenize(text):
    return tuple(re.findall(r"[a-z0-9]+", str(text).lower()))


"""
Build a mapping from a measure's tokenized title to the tree_id it
terminates in, using only nodes where a measure's title actually ends
(n_terminal_here >= 1). Mirrors clustering.py's load_forest().
"""
def build_measure_to_tree(forest_path):
    forest = pd.read_csv(forest_path, dtype=str)
    forest["n_terminal_here"] = forest["n_terminal_here"].astype(int)

    terminal = forest[forest["n_terminal_here"] >= 1]

    path_to_tree = {}
    for _, row in terminal.iterrows():
        key = tuple(str(row["full_path"]).split(" > "))
        path_to_tree[key] = row["tree_id"]

    return path_to_tree


def match_ground_truth_to_trees(ground_truth_path, forest_path):
    gt = pd.read_csv(ground_truth_path, dtype=str)[["measure", "domain"]]
    path_to_tree = build_measure_to_tree(forest_path)

    gt["tree_id"] = gt["measure"].apply(lambda m: path_to_tree.get(tokenize(m)))

    n_matched = gt["tree_id"].notna().sum()
    n_total = len(gt)

    print(f"Ground truth measures matched to a tree: {n_matched} / {n_total} "
          f"({100 * n_matched / n_total:.1f}%)")
    if n_matched < n_total:
        print(f"  -> {n_total - n_matched} measure(s) could not be matched to any tree node "
              f"(tokenization mismatch or measure absent from the forest input); "
              f"excluded from both checks below.")

    return gt[gt["tree_id"].notna()].copy()


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


"""
CHECK 1 -- Per-tree domain purity.

For every final tree_id (i.e. every tree as it exists after all splits
were applied), checks how many of the ground-truth-labeled measures it
contains actually belong to the same domain. A tree is "pure" if all
its ground-truth measures share one domain.

Trees containing a single ground-truth measure are trivially pure and
are reported separately, since they don't tell us anything about
whether the tree correctly grouped multiple related measures.
"""
def evaluate_tree_purity(gt_matched):

    rows = []
    for tree_id, group in gt_matched.groupby("tree_id"):
        counts = group["domain"].value_counts()
        rows.append({
            "tree_id": tree_id,
            "n_measures": len(group),
            "n_domains": counts.shape[0],
            "majority_domain": counts.idxmax(),
            "purity": counts.max() / counts.sum(),
        })

    trees = pd.DataFrame(rows)

    multi = trees[trees["n_measures"] > 1]

    n_pure_multi = (multi["purity"] == 1.0).sum()
    weighted_purity = (multi["purity"] * multi["n_measures"]).sum() / multi["n_measures"].sum() if len(multi) else float("nan")

    print("\n--- Check 1: per-tree domain purity ---")
    print(f"    fully pure (100% same domain): {n_pure_multi} / {len(multi)} "
          f"({100 * n_pure_multi / len(multi):.1f}%)" if len(multi) else "    (none)")
    print(f"    mean purity, weighted by # measures: {weighted_purity:.4f}" if len(multi) else "")

    return trees


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


"""
CHECK 2 -- Relationship preservation under splitting.

The tree encodes a relationship between two measures: they are
considered related if they end up in the same final tree. Splitting a
tree can destroy a relationship (if it separates two measures that
really do belong to the same domain) or correctly remove a spurious
one (if it separates two measures from different domains that had
only their initial words in common).

To isolate the effect of splitting specifically -- as opposed to the
initial root-level grouping by first word, which is a different,
separate source of missed relationships -- this check only considers
pairs of ground-truth measures that started out together, i.e. that
share the same ORIGINAL root tree (the tree before any split was
ever applied to it). For every such pair:

  - same domain, same final tree      -> relationship PRESERVED (good)
  - same domain, different final tree -> relationship LOST by splitting (bad)
  - different domain, different tree  -> correctly separated (good)
  - different domain, same final tree -> still incorrectly merged (impurity)

relationship_recall = preserved / (preserved + lost) answers exactly
the question "of the true semantic relationships that existed before
any splitting, how many does the splitting step still catch?".
"""
def find_root_ancestor(tree_id, parent_of):
    seen = set()
    current = tree_id
    while current in parent_of and parent_of[current] is not None:
        if current in seen:
            break  # safety net against any cyclic data
        seen.add(current)
        current = parent_of[current]
    return current


def evaluate_relationship_preservation(gt_matched, roots_path):

    roots = pd.read_csv(roots_path, dtype=str)

    def norm_id(x):
        return None if pd.isna(x) else str(int(float(x)))

    roots["tree_id"] = roots["tree_id"].apply(norm_id)
    roots["parent_tree_id"] = roots["parent_tree_id"].apply(norm_id)
    parent_of = dict(zip(roots["tree_id"], roots["parent_tree_id"]))

    gt_matched = gt_matched.copy()
    gt_matched["tree_id"] = gt_matched["tree_id"].apply(norm_id)
    gt_matched["root_ancestor"] = gt_matched["tree_id"].apply(lambda t: find_root_ancestor(t, parent_of))

    preserved = lost = separated = still_merged = 0
    lost_pairs = []

    for root_id, group in gt_matched.groupby("root_ancestor"):
        records = group[["measure", "domain", "tree_id"]].to_dict("records")
        n = len(records)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = records[i], records[j]
                same_domain = a["domain"] == b["domain"]
                same_tree = a["tree_id"] == b["tree_id"]

                if same_domain and same_tree:
                    preserved += 1
                elif same_domain and not same_tree:
                    lost += 1
                    lost_pairs.append({
                        "root_ancestor": root_id,
                        "domain": a["domain"],
                        "measure_1": a["measure"],
                        "measure_2": b["measure"],
                        "tree_1": a["tree_id"],
                        "tree_2": b["tree_id"],
                    })
                elif not same_domain and not same_tree:
                    separated += 1
                else:
                    still_merged += 1

    total_same_domain_pairs = preserved + lost
    total_diff_domain_pairs = separated + still_merged

    print("\n--- Check 2: relationship preservation under splitting ---")
    print(f"(scope: pairs of ground-truth measures that started in the same original root tree)")
    print(f"\nSame-domain pairs (true relationships): {total_same_domain_pairs}")
    print(f"  preserved (still together after splitting): {preserved}")
    print(f"  LOST (split apart despite same domain):      {lost}")
    if total_same_domain_pairs:
        recall = preserved / total_same_domain_pairs
        print(f"  relationship_recall = preserved / total = {recall:.4f} ({100*recall:.1f}%)")

    print(f"\nDifferent-domain pairs (spurious relationships): {total_diff_domain_pairs}")
    print(f"  correctly separated by splitting: {separated}")
    print(f"  still incorrectly merged:         {still_merged}")
    if total_diff_domain_pairs:
        precision_gain = separated / total_diff_domain_pairs
        print(f"  fraction correctly separated = {precision_gain:.4f} ({100*precision_gain:.1f}%)")

    if lost:
        print(f"\n  {len(lost_pairs)} lost relationship(s) -- inspect these (same domain, split apart):")
        lost_df = pd.DataFrame(lost_pairs)

    return lost_df


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    GROUND_TRUTH = "quality_evaluation/ground_truth.csv"
    FOREST = "clustering/similarity_forest/forest_export.csv"
    ROOTS = "clustering/similarity_forest/roots.csv"

    gt_matched = match_ground_truth_to_trees(GROUND_TRUTH, FOREST)

    tree_purity = evaluate_tree_purity(gt_matched)
    lost_relationships = evaluate_relationship_preservation(gt_matched, ROOTS)

    tree_purity.to_csv("quality_evaluation/similarity_forest/tree_purity.csv", index=False)
    lost_relationships.to_csv("quality_evaluation/similarity_forest/lost_relationships.csv", index=False)