import pandas as pd
import re


def normalize_title(title):
    """
    Normalize a title for comparison.
    """
    if pd.isna(title):
        return ""

    title = str(title).strip().lower()
    title = re.sub(r"\s+", " ", title)

    return title


clustering = pd.read_csv(
    "output/clustering/breadcrumbs_with_domains.csv",
    dtype=str
)

m = pd.read_csv(
    "output/vocabulary_by_tag/measures.csv",
    dtype=str
)

# Normalize titles
clustering_titles = {
    normalize_title(title)
    for title in clustering["title"]
}

m_titles = m["term"].apply(normalize_title)

# Check which M titles are present in the clustering file
matched = m_titles.isin(clustering_titles)

print(f"Total terms in M: {len(m)}")
print(f"Matched: {matched.sum()}")
print(f"Not matched: {(~matched).sum()}")
print(f"Coverage: {matched.mean() * 100:.2f}%")

# Print unmatched terms
unmatched = m.loc[~matched, "term"]

print("\nUnmatched terms:")
print(unmatched.to_string(index=False))