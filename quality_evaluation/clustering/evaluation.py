import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


DOMAINS_TO_EVALUATE = [
    "Money & Markets",
    "People, Work & Living Conditions",
    "Health & Safety",
    "Learning & Education",
    "Environment & Natural Resources and Production",
    "Transports",
    "Tech & Innovation",
    "Living Well"
]


def evaluate_clustering(ground_truth_file, predicted_file):

    ground_truth = pd.read_csv(ground_truth_file, dtype=str)
    ground_truth = ground_truth[["measure", "domain"]].copy()
    ground_truth = ground_truth.rename(columns={"domain": "domain_true"})

    predicted = pd.read_csv(predicted_file, dtype=str)
    predicted = predicted[["measure", "domain"]].copy()
    predicted = predicted.rename(columns={"domain": "domain_predicted"})

    comparison = pd.merge(ground_truth, predicted, on="measure", how="inner")

    comparison["correct"] = (comparison["domain_true"] == comparison["domain_predicted"])

    #--------------------------------------------------------------------------------------------------------------------------------

    # Global metrics

    accuracy = accuracy_score(comparison["domain_true"], comparison["domain_predicted"])

    precision_macro = precision_score(
        comparison["domain_true"],
        comparison["domain_predicted"],
        labels=DOMAINS_TO_EVALUATE,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        comparison["domain_true"],
        comparison["domain_predicted"],
        labels=DOMAINS_TO_EVALUATE,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        comparison["domain_true"],
        comparison["domain_predicted"],
        labels=DOMAINS_TO_EVALUATE,
        average="macro",
        zero_division=0
    )

    print(f"Domains evaluated: {DOMAINS_TO_EVALUATE}")
    print(f"Total measures compared: {len(comparison)}")
    print(f"Correct classifications: {comparison['correct'].sum()}")
    print(f"Incorrect classifications: {(~comparison['correct']).sum()}")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(f"\nMacro Precision: {precision_macro:.4f}")
    print(f"Macro Precision: {precision_macro * 100:.2f}%")

    print(f"\nMacro Recall: {recall_macro:.4f}")
    print(f"Macro Recall: {recall_macro * 100:.2f}%")

    print(f"\nMacro F1-score: {f1_macro:.4f}")
    print(f"Macro F1-score: {f1_macro * 100:.2f}%")

    #--------------------------------------------------------------------------------------------------------------------------------

    # Metrics per domain

    report = classification_report(
        comparison["domain_true"],
        comparison["domain_predicted"],
        labels=DOMAINS_TO_EVALUATE,
        zero_division=0,
        output_dict=True
    )

    report_df = (pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "domain"}))

    print("\nMetrics by domain:\n")
    print(report_df[report_df["domain"].isin(DOMAINS_TO_EVALUATE)].to_string(index=False))

    return comparison, report_df


#--------------------------------------------------------------------------------------------------------------------------------


def evaluate_domain_assignment_error_source(ground_truth_file, cluster_prefixes_file, comparison):
    """
    Attribute every misclassified measure to one of two error sources,
    reusing the SAME ground truth already built for step 7 (measure -> domain).

    A cluster (tree_id) is the unit the LLM actually sees (step 7 output,
    from clustering/llm_files/cluster_prefixes.csv, i.e. BEFORE the LLM
    assigns a domain). If a cluster already mixes measures whose ground
    truth domain differs, no single LLM answer for that cluster could ever
    be correct for all of them -> that is an upstream (clustering) error.
    If instead the cluster agrees with (or is dominated by) the measure's
    true domain and the LLM still picked something else, the error is
    attributable to the LLM step itself (downstream).

    Parameters
    ----------
    ground_truth_file : CSV with columns [domain, measure] (the step 7 ground truth)
    cluster_prefixes_file : CSV with columns [cluster, prefix, measure]
        (output of build_cluster_prefix, i.e. clustering/llm_files/cluster_prefixes.csv)
    comparison : DataFrame returned by evaluate_clustering(), with columns
        [measure, domain_true, domain_predicted, correct]
    """

    ground_truth = pd.read_csv(ground_truth_file, dtype=str)[["measure", "domain"]]
    clusters = pd.read_csv(cluster_prefixes_file, dtype=str)[["measure", "cluster"]]

    # Ground-truth domain for every clustered measure (not just the ones
    # covered by `comparison`), so purity reflects the whole cluster.
    gt_clusters = clusters.merge(ground_truth, on="measure", how="inner")

    def cluster_stats(group):
        counts = group["domain"].value_counts()
        return pd.Series({
            "cluster_majority_domain": counts.idxmax(),
            "cluster_purity": counts.max() / counts.sum(),
            "cluster_n_gt_measures": int(counts.sum()),
            "cluster_n_domains": int(counts.shape[0]),
        })

    cluster_summary = (
        gt_clusters.groupby("cluster")
        .apply(cluster_stats, include_groups=False)
        .reset_index()
    )

    enriched = comparison.merge(clusters, on="measure", how="left")
    enriched = enriched.merge(cluster_summary, on="cluster", how="left")

    def classify(row):
        if pd.isna(row["cluster"]):
            return "unclustered (step 7 dropped this measure)"
        if row["correct"]:
            return "correct"
        if row["cluster_n_domains"] > 1 and row["domain_true"] != row["cluster_majority_domain"]:
            return "upstream (clustering, step 7)"
        return "downstream (LLM domain choice, step 8)"

    enriched["error_source"] = enriched.apply(classify, axis=1)

    weighted_purity = (
        (cluster_summary["cluster_purity"] * cluster_summary["cluster_n_gt_measures"]).sum()
        / cluster_summary["cluster_n_gt_measures"].sum()
    )

    print("\n--- Step 8 error attribution ---")
    print(f"Clusters covered by ground truth: {len(cluster_summary)}")
    print(f"Mean cluster purity (weighted by # measures): {weighted_purity:.4f}")
    print(f"Pure clusters (100%% single domain): {(cluster_summary['cluster_purity'] == 1.0).sum()} / {len(cluster_summary)}")

    incorrect = enriched[~enriched["correct"]]
    print(f"\nIncorrect measures: {len(incorrect)}")
    print("\nBreakdown by error source (share of incorrect measures):")
    print((incorrect["error_source"].value_counts(normalize=True) * 100).round(1).astype(str) + "%")
    print("\nBreakdown by error source (counts):")
    print(incorrect["error_source"].value_counts())

    return enriched, cluster_summary


if __name__ == "__main__":
    comparison, report_df = evaluate_clustering(
        "quality_evaluation/ground_truth.csv",
        "output/clustering/measure_domain.csv"
    )

    enriched, cluster_summary = evaluate_domain_assignment_error_source(
        "quality_evaluation/ground_truth.csv",
        "clustering/llm_files/cluster_prefixes.csv",
        comparison
    )

    Path("quality_evaluation/clustering").mkdir(parents=True, exist_ok=True)
    enriched.to_csv("quality_evaluation/clustering/error_attribution.csv", index=False)
    cluster_summary.to_csv("quality_evaluation/clustering/cluster_purity.csv", index=False)