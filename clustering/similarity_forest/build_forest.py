import pandas as pd
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from clustering.utils import normalize, word_tokens


"""
Build an uncompressed tree over the given measure indices, using only
the tokens starting from `start_pos` onward.
"""
class RawNode:
    def __init__(self):
        self.children = {}          # token -> RawNode
        self.terminal_measures = [] # indices of measures whose title ends exactly here


def build_raw_tree(indices, tokens, start_pos):
    root = RawNode()
    for idx in indices:
        node = root
        for tok in tokens[idx][start_pos:]:
            if tok not in node.children:
                node.children[tok] = RawNode()
            node = node.children[tok]
        node.terminal_measures.append(idx)
    return root

#------------------------------------------------------------------------------------------------------------------------

"""
Chains of single-child nodes are merged into one edge with a
multi-word label, so the tree only branches where there is
real ambiguity (more than one continuation).
"""
class CompressedNode:
    def __init__(self, edge_label="", path=()):
        self.edge_label = edge_label  # the (possibly multi-word) phrase on the edge INTO this node
        self.path = path              # full sequence of words from the tree's true root to this node
        self.children = []
        self.terminal_measures = []   # measures whose title ends exactly at this node


def compress(raw_node, path_prefix):
    node = CompressedNode(path=path_prefix)
    node.terminal_measures = list(raw_node.terminal_measures)

    for tok, child_raw in raw_node.children.items():
        chain_tokens = [tok]
        cur = child_raw
        while len(cur.children) == 1 and not cur.terminal_measures:
            (next_tok, next_node), = cur.children.items()
            chain_tokens.append(next_tok)
            cur = next_node

        edge_label = " ".join(chain_tokens)
        child_path = path_prefix + tuple(chain_tokens)
        child_compressed = compress(cur, child_path)
        child_compressed.edge_label = edge_label
        node.children.append(child_compressed)

    return node


def collect_subtree_measures(node):
    """Return all measure indices contained anywhere in this node's subtree."""
    result = list(node.terminal_measures)
    for c in node.children:
        result.extend(collect_subtree_measures(c))
    return result


def build_root_groups(tokens, min_root_words=1):
    groups = defaultdict(list)
    for idx, toks in enumerate(tokens):
        key_len = min(min_root_words, len(toks) // 2)
        key = tuple(toks[:key_len])
        groups[key].append(idx)
    return groups


# ============================================================================
# Embeddings: computed once for the whole corpus, reused everywhere below.
# ============================================================================

def embed_terms(terms, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(terms, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(embeddings)


def centroid(embeddings, indices):
    vecs = embeddings[indices]
    c = vecs.mean(axis=0)
    norm = np.linalg.norm(c)
    return c / norm if norm > 0 else c


def cosine_sim(a, b):
    return float(np.dot(a, b))


# ============================================================================
# Split logic
# ============================================================================

DEFAULT_PARAMS = {
    "max_examples": 3,
    "max_split_depth": None,
    "cos_sim_threshold": 0.50,
}


def find_outlier_children(node, embeddings, params):
    if len(node.children) < 2:
        return []

    if params["max_split_depth"] is not None and len(node.path) > params["max_split_depth"]:
        return []

    child_indices = [collect_subtree_measures(c) for c in node.children]

    outliers = []
    for i, indices_i in enumerate(child_indices):
        rest_indices = []
        for j, indices_j in enumerate(child_indices):
            if j != i:
                rest_indices.extend(indices_j)

        if not rest_indices:
            continue

        centroid_i = centroid(embeddings, indices_i)
        centroid_rest = centroid(embeddings, rest_indices)
        sim = cosine_sim(centroid_i, centroid_rest)

        if sim <= params["cos_sim_threshold"]:
            outliers.append(node.children[i])

    return outliers


# ============================================================================
# Forest construction
# ============================================================================

def build_forest_with_splits(terms, params=None):

    params = {**DEFAULT_PARAMS, **(params or {})}

    normalized = [normalize(t) for t in terms]
    tokens = [word_tokens(n) for n in normalized]

    print("Computing embeddings...")
    embeddings = embed_terms(terms)

    root_groups = build_root_groups(tokens, min_root_words=1)

    rows = []
    tree_records = []
    tree_id_counter = [0]

    def walk_and_split(node, tree_id, parent_id, depth, node_counter, total_in_tree):
        node_id = node_counter[0]
        node_counter[0] += 1

        subtree_measures = collect_subtree_measures(node)
        examples = [terms[i] for i in subtree_measures[:params["max_examples"]]]

        rows.append({
            "tree_id": tree_id,
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": depth,
            "edge_label": node.edge_label,
            "full_path": " > ".join(node.path),
            "n_children": len(node.children),
            "n_terminal_here": len(node.terminal_measures),
            "n_measures_subtree": len(subtree_measures),
            "is_branch_point": len(node.children) >= 2,
            "examples": " | ".join(examples),
        })

        outlier_children = find_outlier_children(node, embeddings, params)
        outlier_ids = {id(c) for c in outlier_children}

        for c in node.children:
            if id(c) in outlier_ids:
                new_tid = tree_id_counter[0]
                tree_id_counter[0] += 1
                new_total = len(collect_subtree_measures(c))

                tree_records.append({
                    "tree_id": new_tid,
                    "parent_tree_id": tree_id,
                    "split_from_path": " > ".join(node.path),
                    "root_path": " > ".join(c.path),
                    "n_measures": new_total,
                })

                new_counter = [0]
                walk_and_split(c, new_tid, -1, 0, new_counter, new_total)
            else:
                walk_and_split(c, tree_id, node_id, depth + 1, node_counter, total_in_tree)

    for root_key, indices in root_groups.items():
        tid = tree_id_counter[0]
        tree_id_counter[0] += 1
        total = len(indices)

        tree_records.append({
            "tree_id": tid,
            "parent_tree_id": None,
            "split_from_path": None,
            "root_path": " ".join(root_key),
            "n_measures": total,
        })

        raw_tree = build_raw_tree(indices, tokens, start_pos=len(root_key))
        tree_root = compress(raw_tree, path_prefix=root_key)
        tree_root.edge_label = " ".join(root_key)

        node_counter = [0]
        walk_and_split(tree_root, tid, -1, 0, node_counter, total)

    return pd.DataFrame(rows), pd.DataFrame(tree_records)


def export_forest_csv(input_file, output_file, roots_file, params):

    df = pd.read_csv(input_file, dtype=str)
    terms = df["term"].tolist()

    nodes_df, trees_df = build_forest_with_splits(terms, params=params)

    nodes_df = nodes_df.sort_values(["tree_id", "node_id"])
    nodes_df.to_csv(output_file, index=False)

    trees_df = trees_df.sort_values("tree_id")
    trees_df.to_csv(roots_file, index=False)

    n_trees = len(trees_df)
    n_root_trees = trees_df["parent_tree_id"].isna().sum()
    n_split_trees = n_trees - n_root_trees

    print(f"Total final trees: {n_trees}")
    print(f"  - initial single-word roots: {n_root_trees}")
    print(f"  - trees created by splitting: {n_split_trees}")
    print(f"Trees containing a single measure: {(trees_df['n_measures'] == 1).sum()}")

    return nodes_df, trees_df