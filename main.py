"""Compare models, recommend operating conditions, and save report evidence."""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.constants import (
    BATCH_ID, INITIAL_GLUCOSE, INITIAL_GLUTAMINE, INITIAL_VCD,
    MAB, SPACE_TIME_PROFIT, TIME_D,
)
from src.functions import (
    load_initial_conditions, load_batches, attach_initial_conditions,
    check_data, add_economics, build_one_step_dataset, split_by_condition,
    evaluate_recursive_var, compare_knn_nvar, compare_direct_regression,
    check_polynomial_design, evaluate_harvest_decisions, evaluate_direct_test,
    summarize_best_observed_harvest, search_operating_conditions,
    estimate_prediction_intervals, plot_mab_time_series,
    plot_harvest_economics, plot_initial_condition_effects,
    plot_followup_predictions,
)

# Output locations are anchored to main.py, regardless of PyCharm's working directory.
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_FOLDER = BASE_DIR / "tables"
FIGURES_FOLDER = BASE_DIR / "figures"
SELECTED_MODEL = "Reduced polynomial"


def save_table(table, filename):
    """Export full precision; round only when displaying results."""
    table.to_csv(OUTPUTS_FOLDER / filename, index=False)


def save_figure(figure, filename):
    """Save without opening a blocking plot window."""
    figure.savefig(FIGURES_FOLDER / filename, dpi=200)
    plt.close(figure)


def harvest_metrics(results):
    """Summarize economic errors without printing every batch."""
    missed = ((results["Best observed STP"] > 0)
              & (results["Observed STP at selected harvest"] < 0))
    return {
        "Mean STP gap": results["Observed STP gap"].mean(),
        "Median STP gap": results["Observed STP gap"].median(),
        "Worst STP gap": results["Observed STP gap"].max(),
        "Harvest STP MAE": results["STP prediction error"].abs().mean(),
        "Missed profitable batches": int(missed.sum()),
    }


def main():
    OUTPUTS_FOLDER.mkdir(parents=True, exist_ok=True)
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    initial_conditions = load_initial_conditions()
    data = attach_initial_conditions(load_batches(), initial_conditions)
    check_data(data)
    data = add_economics(data)

    # Preserve the original split and folds, including replicate grouping.
    x, y, metadata = build_one_step_dataset(data)
    x_train, _, y_train, _, metadata_train, metadata_test = split_by_condition(x, y, metadata)
    split = metadata[[BATCH_ID, "Condition group", INITIAL_GLUCOSE,
                      INITIAL_GLUTAMINE, INITIAL_VCD]].drop_duplicates().copy()
    split["Subset"] = np.where(
        split["Condition group"].isin(metadata_test["Condition group"]),
        "Follow-up (previously examined)", "Development",
    )
    save_table(split, "condition_split.csv")
    print(f"Loaded {data[BATCH_ID].nunique()} batches: "
          f"{metadata_train[BATCH_ID].nunique()} development, "
          f"{metadata_test[BATCH_ID].nunique()} follow-up.")
    print("Evaluating models using grouped development-data validation...")

    # Retain recursive alternatives, but omit redundant one-step scoring and plots.
    _, recursive_summary, _ = evaluate_recursive_var(x_train, y_train, metadata_train, data)
    _, knn_comparison, _ = compare_knn_nvar(x_train, y_train, metadata_train, data)
    linear = recursive_summary.loc[recursive_summary["Variable"] == MAB].copy()
    linear["Model"] = "Linear VAR"
    knn = knn_comparison.rename(columns={
        "Mean batch mAb RMSE [mg/L]": "RMSE_mean",
        "SD batch mAb RMSE [mg/L]": "RMSE_std",
        "Mean batch mAb R2": "R2_mean",
    })
    knn["Model"] = knn["Neighbours"].map(lambda k: f"Recursive kNN k={k}")

    models, comparison, scores, trajectories = compare_direct_regression(data, metadata_train)
    metrics = ["Model", "RMSE_mean", "RMSE_std", "R2_mean"]
    model_comparison = pd.concat(
        [comparison[metrics], linear[metrics], knn[metrics]], ignore_index=True,
    ).sort_values("RMSE_mean")
    save_table(model_comparison, "model_comparison.csv")
    save_table(check_polynomial_design(data, metadata_train), "polynomial_design_checks.csv")

    # Compare harvest decisions as well as concentration errors.
    harvest_tables, harvest_rows = [], []
    for name in comparison["Model"]:
        results = evaluate_harvest_decisions(trajectories, data, model_name=name)
        results["Model"] = name
        harvest_tables.append(results)
        harvest_rows.append({"Model": name, **harvest_metrics(results)})
    harvest_comparison = pd.DataFrame(harvest_rows).merge(comparison, on="Model", validate="one_to_one")
    harvest_comparison = harvest_comparison.sort_values("Mean STP gap")
    save_table(harvest_comparison, "harvest_model_comparison.csv")
    save_table(pd.concat(harvest_tables, ignore_index=True), "harvest_validation_batches.csv")
    save_table(scores, "validation_batch_scores.csv")
    # Only the selected model's trajectories are needed for residual intervals and inspection.
    save_table(trajectories.loc[trajectories["Model"] == SELECTED_MODEL], "selected_validation_trajectories.csv")

    # This choice is fixed from the completed development analysis.
    # Previously examined test conditions provide follow-up evidence, not a new independent test.
    model = models[SELECTED_MODEL]
    followup_scores, followup_summary, followup_trajectories, followup_harvest = evaluate_direct_test(
        model, data, metadata_test, model_name=SELECTED_MODEL,
    )
    followup_summary = followup_summary.rename(columns={"Test batches": "Follow-up batches"})
    for name, value in harvest_metrics(followup_harvest).items():
        followup_summary[name] = value
    save_table(followup_summary, "reduced_followup_summary.csv")
    save_table(followup_scores, "reduced_followup_batches.csv")
    save_table(followup_trajectories, "reduced_followup_trajectories.csv")
    save_table(followup_harvest, "reduced_followup_harvest.csv")

    # Search the supplied condition grid and positive recorded harvest times.
    ranking, excluded, count = search_operating_conditions(model, initial_conditions, data)
    recommendation = ranking.head(1)
    save_table(ranking, "predicted_operating_conditions_ranked.csv")
    observed = summarize_best_observed_harvest(data)
    save_table(observed, "best_observed_harvest_per_batch.csv")
    features = [INITIAL_GLUCOSE, INITIAL_GLUTAMINE, INITIAL_VCD, TIME_D]
    reference = data.loc[[data[SPACE_TIME_PROFIT].idxmax()], features].copy()
    candidates = pd.concat([
        recommendation[features].assign(Case="Model recommendation"),
        reference.assign(Case="Best observed operating point"),
    ], ignore_index=True)
    intervals, coverage, lower, upper = estimate_prediction_intervals(
        model, candidates, trajectories, followup_trajectories, model_name=SELECTED_MODEL,
    )
    save_table(intervals, "operating_prediction_intervals.csv")
    save_table(coverage, "prediction_interval_followup_coverage.csv")
    save_table(pd.DataFrame([{
        "Candidates evaluated": count, "Negative candidates excluded": excluded,
        "Residual lower 2.5% [mg/L]": lower, "Residual upper 97.5% [mg/L]": upper,
        "Follow-up mean batch coverage": coverage["Coverage"].mean(),
    }]), "search_and_interval_summary.csv")

    # Four figures cover the data, model fit, harvest choice, and input effects.
    save_figure(plot_mab_time_series(data), "mab_time_series.png")
    save_figure(plot_harvest_economics(model, recommendation, data, lower, upper),
                "recommended_harvest_economics.png")
    save_figure(plot_initial_condition_effects(model, recommendation, initial_conditions),
                "initial_condition_effects.png")
    save_figure(plot_followup_predictions(followup_trajectories, lower, upper),
                "reduced_followup_predictions.png")

    # Keep the terminal concise. Detailed comparison tables remain available as CSVs.
    chosen = harvest_comparison.loc[harvest_comparison["Model"] == SELECTED_MODEL].iloc[0]
    print(f"\nSelected model: {SELECTED_MODEL}")
    print(f"Validation: mean batch RMSE = {chosen['RMSE_mean']:.4f} mg/L; "
          f"mean STP gap = {chosen['Mean STP gap']:.3f} $/(L*d).")
    result = followup_summary.iloc[0]
    print(f"Follow-up: mean batch RMSE = {result['RMSE_mean']:.4f} mg/L; "
          f"mean STP gap = {result['Mean STP gap']:.3f} $/(L*d).")
    print("Follow-up batches were examined previously; this is not an independent final test.")
    print("\nRecommended operating point and approximate pointwise 95% intervals:")
    print(intervals.iloc[0].drop(labels=["Case", "Time [h]"]).round(4).to_string())
    print(f"\nFollow-up mean interval coverage: {coverage['Coverage'].mean():.1%}.")
    print("Intervals describe pointwise prediction errors, not uncertainty in the optimal conditions.")
    print("Saved result tables to tables/ and four figures to figures/.")


if __name__ == "__main__":
    main()
