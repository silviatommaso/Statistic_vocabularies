import pandas as pd
import re
from pathlib import Path


######################################################################
# Vocabulary measures clustering utils functions (Phase III)
######################################################################


FIRST_LEVEL_TO_FINAL_DOMAIN = {
    "General and regional statistics": "General and regional statistics",
    "Economy and finance": "Economy and finance",
    "Population and social conditions": "Population and social conditions",
    "Industry, trade and services": "Industry, trade and services",
    "Agriculture, forestry and fisheries": "Agriculture, forestry and fisheries",
    "International trade": "Industry, trade and services",
    "Transport": "Transport",
    "Environment and energy": "Environment and energy",
    "Science, technology, digital society": "Science, technology, digital society"
}


SECOND_LEVEL_TO_FIRST_LEVEL = {
    "Macroeconomic imbalance procedure indicators": "Economy and finance",
    "Euro indicators / PEEIs": "Economy and finance",
    "Circular economy indicators": "Environment and energy",
    "Sustainable development indicators": "General and regional statistics",
    "Employment and social policy indicators": "Population and social conditions",
    "European pillar of social rights (EPSR)": "Population and social conditions",
    "Quality of life": "Population and social conditions",
    "Migrant integration and children in migration": "Population and social conditions",
    "Economic globalisation indicators": "Economy and finance",
    "Equality and non-discrimination": "Population and social conditions",
    "Quality of employment": "Population and social conditions",
    "Agri-environmental indicators": "Agriculture, forestry and fisheries",
    "Climate change": "Environment and energy",
    "Skills-related statistics": "Population and social conditions",
    "Children and youth": "Population and social conditions",
    "Housing": "Population and social conditions"
}


######################################################################
# DOMAIN EXTRACTION
######################################################################


def get_domain(breadcrumb):
    """
    Assign a final domain to a table according to its breadcrumb.

    First-level domains are mapped directly to the final domain.
    In particular:

        International trade
            ->
        Industry, trade and services

    Second-level domains are mapped to one of the
    final first-level domains.
    """

    if not isinstance(breadcrumb, str):
        return None

    parts = [
        part.strip()
        for part in breadcrumb.split(">")
    ]

    # --------------------------------------------------------------
    # 1. First-level domains
    # --------------------------------------------------------------

    if len(parts) > 1:

        first_level_domain = parts[1]

        if first_level_domain in FIRST_LEVEL_TO_FINAL_DOMAIN:

            return FIRST_LEVEL_TO_FINAL_DOMAIN[
                first_level_domain
            ]

    # --------------------------------------------------------------
    # 2. Second-level domains
    # --------------------------------------------------------------

    if len(parts) > 2:

        second_level_domain = parts[2]

        if second_level_domain in SECOND_LEVEL_TO_FIRST_LEVEL:

            return SECOND_LEVEL_TO_FIRST_LEVEL[
                second_level_domain
            ]

    return None


######################################################################
# CODE NORMALIZATION
######################################################################


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


######################################################################
# PREFIX GENERATION
######################################################################


def get_prefixes(filename):
    """
    Generate candidate prefixes from a filename,
    from the most specific to the most general.

    Examples:
        ttr_r_p_01 -> ttr_r_p, ttr_r, ttr
        tps2000    -> tps
    """

    stem = normalize_code(filename)

    parts = stem.split("_")

    prefixes = []

    # Prefixes based on "_"
    if len(parts) > 1:

        for i in range(
            len(parts) - 1,
            0,
            -1
        ):

            prefixes.append(
                "_".join(parts[:i])
            )

    # Prefix before trailing numeric suffix
    numeric_prefix = re.sub(
        r"\d+$",
        "",
        stem
    )

    if (
        numeric_prefix
        and numeric_prefix != stem
    ):

        prefixes.append(
            numeric_prefix.rstrip("_")
        )

    # Remove duplicates while preserving order
    return list(
        dict.fromkeys(prefixes)
    )


######################################################################
# PREFIX MATCHING
######################################################################


def matches_prefix(filename, prefix):
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

    stem = normalize_code(filename)

    # Exact match
    if stem == prefix:
        return True

    # Prefix followed by "_"
    if stem.startswith(prefix + "_"):
        return True

    # Prefix followed only by digits
    if stem.startswith(prefix):

        suffix = stem[len(prefix):]

        if suffix.isdigit():
            return True

    return False


######################################################################
# BUILD CLUSTERS
######################################################################


def build_clusters(breadcrumbs_file):
    """
    Assign a final domain to every table according
    to its breadcrumb.

    Mapping rules:

    - First-level domains are preserved, except:

        International trade
            ->
        Industry, trade and services

    - Second-level domains are mapped according to
      SECOND_LEVEL_TO_FIRST_LEVEL.

    The output contains one row for every unique code.
    """

    # --------------------------------------------------------------
    # 1. Read breadcrumbs
    # --------------------------------------------------------------

    clusters = pd.read_csv(
        breadcrumbs_file,
        dtype=str
    )

    # --------------------------------------------------------------
    # 2. Assign domain
    # --------------------------------------------------------------

    clusters["domain"] = (
        clusters["breadcrumb"]
        .apply(get_domain)
    )

    # --------------------------------------------------------------
    # 3. Keep only required columns
    # --------------------------------------------------------------

    clusters = clusters[
        ["domain", "code"]
    ].copy()

    # --------------------------------------------------------------
    # 4. Remove duplicate codes
    # --------------------------------------------------------------

    clusters = clusters.drop_duplicates(
        subset=["code"],
        keep="first"
    )

    # --------------------------------------------------------------
    # 5. Sort
    # --------------------------------------------------------------

    clusters = clusters.sort_values(
        ["domain", "code"],
        na_position="last"
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------------

    output_path = Path(
        "quality_evaluation/step_VII/cluster_eval.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    clusters.to_csv(
        output_path,
        index=False
    )

    return clusters


######################################################################
# MAIN
######################################################################


if __name__ == "__main__":

    clusters = build_clusters(
        breadcrumbs_file=(
            "input/auxiliar_files/TOC/"
            "breadcrumbs_filtered.csv"
        )
    )

    print(
        clusters.to_string(
            index=False
        )
    )