import pandas as pd


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


########################################################################################################################################################################################################################
# Vocabulary measures clustering utils functions (Phase III)
########################################################################################################################################################################################################################

"""
Assign a predefined domain to a breadcrumb.

The domain is identified by looking for an exact match
among the hierarchical components of the breadcrumb.
"""
def assign_domain(breadcrumb, domains):

    parts = [part.strip() for part in breadcrumb.split(">")]

    # Find the domain among the hierarchical components
    for domain in domains:
        if domain in parts:
            return domain

    return None



"""
Assign a predefined domain to each table in the breadcrumb file.
"""
def assign_domains(breadcrumbs_file):
    breadcrumbs = pd.read_csv(breadcrumbs_file, dtype=str)

    breadcrumbs["domain"] = breadcrumbs["breadcrumb"].apply(lambda x: assign_domain(x, DOMAINS))
    breadcrumbs["title"] = breadcrumbs["breadcrumb"].apply(lambda x: x.split(">")[-1].strip())

    result = breadcrumbs[["domain", "code", "title"]].copy()
    
    result.to_csv("M_clustering/breadcrumbs_with_domains.csv", index=False)

    return breadcrumbs