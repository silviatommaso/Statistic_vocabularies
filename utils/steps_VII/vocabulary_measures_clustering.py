import pandas as pd
import re
from pathlib import Path
from difflib import SequenceMatcher


DOMAINS = [
    "General and regional statistics",
    "Economy and finance",
    "Population and social conditions",
    "Industry, trade and services",
    "Agriculture, forestry and fisheries",
    "International trade",
    "Transport",
    "Environment and energy",
    "Science, technology, digital society",
    "Macroeconomic imbalance procedure indicators",
    "Euro indicators / PEEIs",
    "Circular economy indicators",
    "Sustainable development indicators",
    "Employment and social policy indicators",
    "European pillar of social rights (EPSR)",
    "Quality of life",
    "Migrant integration and children in migration",
    "Economic globalisation indicators",
    "Equality and non-discrimination",
    "Quality of employment",
    "Agri-environmental indicators",
    "Climate change",
    "Skills-related statistics",
    "Children and youth",
    "Housing"
]


######################################################################
# Vocabulary measures clustering utils functions (Phase III)
######################################################################


"""
Assign a predefined domain to a breadcrumb.

The domain is identified by looking for an exact match
among the hierarchical components of the breadcrumb.
"""
def get_domain(breadcrumb, domains):

    parts = [part.strip() for part in breadcrumb.split(">")]

    # Find the domain among the hierarchical components
    for domain in domains:
        if domain in parts:
            return domain

    return None


##############################################################################################################################################################################################################################################################################################################################################################

def normalize_code(filename):

    """
    Normalize a table code before prefix matching.

    Everything starting from '$' is removed.

    Examples:
        lfsa_igar$dv_1902 -> lfsa_igar
        lfsa_igar$dv_1903 -> lfsa_igar
        trng_cvt_12s      -> trng_cvt_12s
    """

    stem = Path(filename).stem
    stem = stem.split("$")[0]
    return stem


"""
Generate candidate prefixes from a filename, from the most
specific to the most general.
"""
def get_prefixes(filename):

    stem = normalize_code(filename)

    parts = stem.split("_")
    prefixes = []

    # Prefixes based on "_": ttr_r_p_01 -> ttr_r_p -> ttr_r -> ttr
    if len(parts) > 1:
        for i in range(len(parts) - 1, 0, -1):
            prefixes.append("_".join(parts[:i]))

    # Prefix before a trailing numeric suffix: tps2000 -> tps
    numeric_prefix = re.sub(r"\d+$", "", stem)

    if numeric_prefix and numeric_prefix != stem:
        prefixes.append(numeric_prefix.rstrip("_"))

    return list(dict.fromkeys(prefixes))


"""
Check whether a filename belongs to a given prefix.

A file matches if:
    1. the filename is exactly the prefix;
    2. the prefix is followed by "_";
    3. the prefix is followed only by digits.

Examples:
    ttr_r_p_01.csv matches "ttr_r_p"
    trng_cvt_12s.csv matches "trng_cvt"
    tps2000.csv matches "tps"
"""
def matches_prefix(filename, prefix):

    stem = normalize_code(filename)

    # Exact match
    if stem == prefix:
        return True

    # Prefix followed by "_"
    if stem.startswith(prefix + "_"):
        return True

    # Prefix followed only by numbers
    if stem.startswith(prefix):
        suffix = stem[len(prefix):]
        if suffix.isdigit():
            return True

    return False


##############################################################################################################################################################################################################################################################################################################################################################


"""
Assigns a domain and a title to each table and extends the clustering
with unmatched table codes using filename-prefix matching.

Matched tables are assigned to a domain according to their breadcrumb.
Unmatched tables are assigned to every domain whose share of matching
tables for the most specific matching prefix is above the threshold.
If no domain reaches the threshold, the table is assigned to "Other".
"""
def build_clusters(breadcrumbs_file, unmatched_file, M):

    # 1. Build initial clusters from breadcrumbs
    clusters = pd.read_csv(breadcrumbs_file, dtype=str)
    clusters["domain"] = clusters["breadcrumb"].apply(lambda x: get_domain(x, DOMAINS))

    # Build mapping from table code to measure title
    code_to_title = {}

    for title, codes in M.items():
        for code in codes:
            code_to_title[code] = title

    # Assign title to each table
    clusters["title"] = clusters["code"].map(code_to_title)
    # Keep only the required columns
    clusters = clusters[["domain", "code", "title"]].copy()


    # 2. Assign unmatched tables
    unmatched = pd.read_csv(unmatched_file, dtype=str)

    new_rows = []

    for code in unmatched["code"]:

        # Candidate prefixes, from most specific to most general
        unmatched_prefixes = get_prefixes(code)

        assigned_domains = []

        for prefix in unmatched_prefixes:

            # Find clustered codes having this prefix
            matches = clusters[clusters["code"].apply(lambda x: matches_prefix(x, prefix))]

            if matches.empty:
                continue

            # Count matches for each domain
            domain_counts = matches["domain"].dropna().value_counts()

            if domain_counts.empty:
                continue

            # Find the maximum number of matches
            max_count = domain_counts.max()

            # Select all domains having the maximum number of matches
            assigned_domains = domain_counts[domain_counts == max_count].index.tolist()

            # Use the first prefix for which matches exist
            break

        # If no prefix matches any clustered table, assign to Other
        if not assigned_domains:
            assigned_domains = ["Other"]

        # Retrieve title
        title = code_to_title.get(code)

        # Add the unmatched table to the selected domains
        for domain in assigned_domains:
            new_rows.append({"domain": domain, "code": code, "title": title})


    # 3. Add unmatched tables directly to the existing clusters
    if new_rows:
        clusters = pd.concat([clusters, pd.DataFrame(new_rows)], ignore_index=True)


    # 4. Save final clustering
    output_path = Path("output/clustering/clusters.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clusters.to_csv(output_path, index=False)

    return clusters