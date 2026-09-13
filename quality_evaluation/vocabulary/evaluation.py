import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CATEGORY_FILES = {
    "N": "dimension_names.csv",
    "A": "dimension_values.csv",
    "U": "units.csv",
}

MAX_TERMS_PER_CATEGORY = 5000
K_NEIGHBORS = 5
RANDOM_SEED = 42


def load_vocabulary(vocabulary_dir):
    dataframes = []

    for category, filename in CATEGORY_FILES.items():
        path = vocabulary_dir / filename

        df = pd.read_csv(path, usecols=["term"])
        df = df.dropna().drop_duplicates()

        # Limit term numbers 
        if len(df) > MAX_TERMS_PER_CATEGORY:
            df = df.sample(n=MAX_TERMS_PER_CATEGORY, random_state=RANDOM_SEED)

        df["category"] = category

        dataframes.append(df)

        print(f"{category}: {len(df)} termini caricati")

    return pd.concat(dataframes, ignore_index=True)


def evaluate_self_consistency(df):
    terms = df["term"].tolist()
    categories = df["category"].to_numpy()

    # TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform(terms)

    # terms similarity
    similarity = cosine_similarity(tfidf)

    scores = []
    neighbor_records = []

    for i in range(len(terms)):

        # Similarity of a term with the others
        similarities = similarity[i].copy()

        # do not consider the term itself
        similarities[i] = -1

        k = min(K_NEIGHBORS, len(terms) - 1)

        # k most similar terms
        nearest_indices = np.argpartition(similarities, -k)[-k:]

        same_category = (categories[nearest_indices] == categories[i])

        score = same_category.mean()
        scores.append(score)

        for j in nearest_indices:
            neighbor_records.append({
                "term": terms[i],
                "category": categories[i],
                "neighbor": terms[j],
                "neighbor_category": categories[j],
                "similarity": similarities[j]
            })

    # Self-consistency per term
    df = df.copy()
    df["self_consistency"] = scores

    neighbors_df = pd.DataFrame(neighbor_records)

    return df, neighbors_df


if __name__ == "__main__":

    # 1. Load vocabulary
    vocabulary_dir = Path("output/vocabulary_by_tag")
    vocabulary = load_vocabulary(vocabulary_dir)


    # 2. Self-consistency
    evaluated, neighbors = evaluate_self_consistency(vocabulary)

    print("\n--- Self-consistency ---")
    print(evaluated.groupby("category")["self_consistency"].mean())
    print( "\nOverall:", evaluated["self_consistency"].mean())


    # 3. Neighbour categories
    print("\n--- Neighbor categories ---")
    neighbor_distribution = pd.crosstab(neighbors["category"], neighbors["neighbor_category"], normalize="index")
    print(neighbor_distribution)