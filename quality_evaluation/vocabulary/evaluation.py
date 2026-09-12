import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


CATEGORY_FILES = {
    "N": "dimension_names.csv",
    "A": "dimension_values.csv",
    "U": "units.csv",
}

MAX_TERMS_PER_CATEGORY = 20000
RANDOM_SEED = 42
K_NEIGHBORS = 5
EPSILON = 1e-12


def load_vocabulary(vocab_dir):
    vocab_dir = Path(vocab_dir)

    frames = []

    for category, filename in CATEGORY_FILES.items():
        df = pd.read_csv(vocab_dir / filename, dtype=str)
        df = df[["term"]].dropna().drop_duplicates()
        df["category"] = category
        frames.append(df)

    all_terms = pd.concat(frames, ignore_index=True)

    print("Vocabulary loaded:")
    for category in CATEGORY_FILES:
        n = (all_terms["category"] == category).sum()
        print(f"  {category}: {n} terms")

    return all_terms


def subsample(
    all_terms,
    max_per_category=MAX_TERMS_PER_CATEGORY,
    seed=RANDOM_SEED
):
    parts = []

    for category, group in all_terms.groupby("category"):
        if len(group) > max_per_category:
            group = group.sample(
                max_per_category,
                random_state=seed
            )
            print(
                f"  {category}: subsampled to "
                f"{max_per_category} terms"
            )

        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def evaluate_category_consistency(
    all_terms,
    k_neighbors=K_NEIGHBORS
):
    sampled = subsample(all_terms)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.5,
        min_df=1,
    )

    tfidf = vectorizer.fit_transform(
        sampled["term"].astype(str)
    )

    sim_matrix = cosine_similarity(tfidf)

    categories = list(CATEGORY_FILES.keys())
    category_array = sampled["category"].to_numpy()

    idx_by_category = {
        category: np.where(
            category_array == category
        )[0]
        for category in categories
    }

    print("\n--- Category self-consistency ---")

    results = []

    for category in categories:
        own_idx = idx_by_category[category]
        other_idx = np.concatenate([
            idx_by_category[other]
            for other in categories
            if other != category
        ])

        n = len(own_idx)

        # --------------------------------------------------
        # 1. Within-category similarity
        # --------------------------------------------------

        within = sim_matrix[np.ix_(own_idx, own_idx)]

        if n > 1:
            within_similarity = (
                within.sum() - n
            ) / (n * (n - 1))
        else:
            within_similarity = np.nan

        # --------------------------------------------------
        # 2. Category-relative separation
        # --------------------------------------------------

        own_to_own = within.copy()
        np.fill_diagonal(own_to_own, np.nan)

        own_similarity_per_term = np.nanmean(
            own_to_own,
            axis=1
        )

        own_to_other = sim_matrix[
            np.ix_(own_idx, other_idx)
        ]

        other_similarity_per_term = np.mean(
            own_to_other,
            axis=1
        )

        separation_per_term = (
            own_similarity_per_term
            - other_similarity_per_term
        )

        category_separation = np.nanmean(
            separation_per_term
        )

        # --------------------------------------------------
        # 3. Relative separation in [0, 1]
        # --------------------------------------------------

        relative_separation_per_term = (
            own_similarity_per_term
            /
            (
                own_similarity_per_term
                + other_similarity_per_term
                + EPSILON
            )
        )

        relative_separation = np.nanmean(
            relative_separation_per_term
        )

        # --------------------------------------------------
        # 4. Nearest-neighbor consistency
        # --------------------------------------------------

        nn_consistency_per_term = []

        for idx in own_idx:
            similarities = sim_matrix[idx].copy()

            # Exclude the term itself
            similarities[idx] = -np.inf

            k = min(k_neighbors, len(similarities) - 1)

            nearest_indices = np.argpartition(
                similarities,
                -k
            )[-k:]

            nearest_categories = category_array[
                nearest_indices
            ]

            same_category_fraction = np.mean(
                nearest_categories == category
            )

            nn_consistency_per_term.append(
                same_category_fraction
            )

        nn_consistency = np.mean(
            nn_consistency_per_term
        )

        results.append({
            "category": category,
            "n_terms": n,
            "within_similarity": within_similarity,
            "category_separation": category_separation,
            "relative_separation": relative_separation,
            "nearest_neighbor_consistency": nn_consistency,
        })

        print(
            f"  {category}: "
            f"within={within_similarity:.4f}  "
            f"separation={category_separation:.4f}  "
            f"relative_separation={relative_separation:.4f}  "
            f"NN_consistency={nn_consistency:.4f}"
        )

    results_df = pd.DataFrame(results)

    print(
        "\nOverall nearest-neighbor consistency: "
        f"{results_df['nearest_neighbor_consistency'].mean():.4f}"
    )

    print(
        "Overall relative separation: "
        f"{results_df['relative_separation'].mean():.4f}"
    )

    return results_df


if __name__ == "__main__":

    VOCAB_DIR = "output/vocabulary_by_tag"

    all_terms = load_vocabulary(VOCAB_DIR)

    results = evaluate_category_consistency(
        all_terms,
        k_neighbors=5
    )

    out_dir = Path("quality_evaluation/vocabulary")
    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        out_dir / "category_consistency_scores.csv",
        index=False
    )