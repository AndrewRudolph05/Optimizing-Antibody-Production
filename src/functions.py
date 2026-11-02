from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (StandardScaler, PolynomialFeatures, FunctionTransformer,)
from sklearn.neighbors import KNeighborsRegressor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .constants import (
    AMMONIA,
    BATCH_ID,
    COST,
    DATASETS_FOLDER,
    DOWNTIME_D,
    GLUCOSE,
    GLUTAMINE,
    INITIAL_GLUCOSE,
    INITIAL_GLUTAMINE,
    INITIAL_VCD,
    LACTATE,
    MAB,
    N_CV_SPLITS,
    N_EXPECTED_BATCHES,
    PROCESS_COLUMNS,
    PROFIT,
    RANDOM_STATE,
    SPACE_TIME_PROFIT,
    SPACE_TIME_YIELD,
    TCD,
    TEST_FRACTION,
    TIME_D,
    TIME_H,
    VALUE,
    VCD,
)

def build_one_step_dataset(data: pd.DataFrame):
    """Pair each process state with the next state in the same batch."""
    ordered = data.sort_values(
        [BATCH_ID, TIME_H]
    ).reset_index(drop=True)

    # Each input contains the seven current process measurements.
    current_values = ordered[PROCESS_COLUMNS]

    # shift(-1) moves the next measurement into the current row.
    # Grouping prevents pairs from crossing batch boundaries.
    next_values = ordered.groupby(BATCH_ID)[PROCESS_COLUMNS].shift(-1)
    next_times = ordered.groupby(BATCH_ID)[TIME_H].shift(-1)

    # The last row of each batch has no next measurement.
    # Exclude those rows from the training pairs.
    has_next = next_times.notna()

    x = current_values.loc[has_next].reset_index(drop=True)
    y = next_values.loc[has_next].reset_index(drop=True)

    # Keep identifying information separate from model inputs.
    # Batch IDs and times help us trace and validate each pair.
    metadata = ordered.loc[
        has_next,
        [
            BATCH_ID,
            TIME_H,
            INITIAL_GLUCOSE,
            INITIAL_GLUTAMINE,
            INITIAL_VCD,
        ]
    ].reset_index(drop=True)

    metadata = metadata.rename(columns={TIME_H: "Input time [h]"})
    metadata["Target time [h]"] = (
        next_times.loc[has_next].to_numpy()
    )

    # Replicates share one condition group. Later, grouped validation
    # can keep identical starting conditions in the same fold.
    condition_columns = [
        INITIAL_GLUCOSE,
        INITIAL_GLUTAMINE,
        INITIAL_VCD,
    ]

    metadata["Condition group"] = metadata.groupby(
        condition_columns, sort=True
    ).ngroup()

    return x, y, metadata

def require_columns(table: pd.DataFrame, required: list, name: str) -> None:
    """Stop with a clear message if expected columns are absent."""
    missing = [
        column for column in required
        if column not in table.columns
    ]

    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_initial_conditions() -> pd.DataFrame:
    """Load the table containing each batch's starting conditions."""
    initial_conditions = pd.read_csv(
        DATASETS_FOLDER / "initial_conditions.csv"
    )

    required = [BATCH_ID, GLUCOSE, GLUTAMINE, VCD]
    require_columns(initial_conditions, required, "Initial-conditions table")

    return initial_conditions[required].copy()


def read_batch(file: Path) -> pd.DataFrame:
    """Read one batch file and attach its batch ID."""
    batch_data = pd.read_csv(file)

    required = [TIME_H] + PROCESS_COLUMNS
    require_columns(batch_data, required, file.name)

    if batch_data.empty:
        raise ValueError(f"{file.name} contains no measurements.")

    batch_data = batch_data[required].copy()

    # Example: "batch_0001" becomes the integer 1.
    batch_data[BATCH_ID] = int(file.stem.split("_")[1])

    return batch_data


def load_batches() -> pd.DataFrame:
    """Combine the selected batch files into one time-ordered table."""
    batch_files = sorted(DATASETS_FOLDER.glob("batch_*.csv"))

    if len(batch_files) != N_EXPECTED_BATCHES:
        raise ValueError(
            f"Expected {N_EXPECTED_BATCHES} batch files, "
            f"found {len(batch_files)}."
        )

    # Read each file, then stack all measurements into one table.
    data = pd.concat(
        [read_batch(file) for file in batch_files],
        ignore_index=True
    )

    # Preserve time order within each batch for later lagged inputs.
    return data.sort_values(
        [BATCH_ID, TIME_H]
    ).reset_index(drop=True)


def attach_initial_conditions(
    data: pd.DataFrame,
    initial_conditions: pd.DataFrame
) -> pd.DataFrame:
    """Attach each batch's starting conditions to all its measurements."""
    missing_ids = sorted(
        set(data[BATCH_ID]) - set(initial_conditions[BATCH_ID])
    )

    if missing_ids:
        raise ValueError(
            f"Missing initial conditions for batches: {missing_ids}"
        )

    # Distinguish starting conditions from measurements during the batch.
    selected = initial_conditions.rename(columns={
        GLUCOSE: INITIAL_GLUCOSE,
        GLUTAMINE: INITIAL_GLUTAMINE,
        VCD: INITIAL_VCD,
    })

    # A left merge preserves the selected batches' measurements.
    # many_to_one requires one initial-condition row per batch ID.
    return data.merge(
        selected,
        on=BATCH_ID,
        how="left",
        validate="many_to_one"
    )


def check_data(data: pd.DataFrame) -> None:
    """Stop on invalid data; successful checks do not clutter the terminal."""

    if data[BATCH_ID].nunique() != N_EXPECTED_BATCHES:
        raise ValueError(
            f"The loaded data must contain "
            f"{N_EXPECTED_BATCHES} unique batches."
        )

    required_numeric = (
        [TIME_H] + PROCESS_COLUMNS
        + [INITIAL_GLUCOSE, INITIAL_GLUTAMINE, INITIAL_VCD]
    )

    # Check the original columns, rather than a converted temporary copy.
    nonnumeric = [
        column for column in required_numeric
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if nonnumeric:
        raise ValueError(
            f"Columns must contain numeric values: {nonnumeric}"
        )

    # Missing or infinite values must be addressed before modelling.
    bad_columns = [
        column for column in required_numeric
        if not np.isfinite(data[column].to_numpy()).all()
    ]

    if bad_columns:
        raise ValueError(
            f"Columns with missing or infinite values: {bad_columns}"
        )

    if (data[TIME_H] < 0).any():
        raise ValueError("Negative measurement times were found.")

    if (data.groupby(BATCH_ID).size() < 2).any():
        raise ValueError("Each batch needs at least two time points.")

    # Calculate time differences within each batch, never across batches.
    valid_steps = data.groupby(BATCH_ID)[TIME_H].diff().dropna()

    # Sorted duplicate batch/time pairs produce a time step of zero.
    if (valid_steps <= 0).any():
        raise ValueError(
            "Measurement times must increase within each batch "
            "(duplicate times are not allowed)."
        )

    # All models use a shared, fixed sampling grid starting at zero.
    grids = [batch[TIME_H].to_numpy() for _, batch in data.groupby(BATCH_ID)]
    reference = grids[0]
    if not np.isclose(reference[0], 0.0):
        raise ValueError("Batch measurements must start at time zero.")
    if any(len(grid) != len(reference) or not np.allclose(grid, reference)
           for grid in grids):
        raise ValueError("All batches must use the same measurement times.")
    if not np.allclose(valid_steps, valid_steps.iloc[0], rtol=1e-6, atol=1e-8):
        raise ValueError("Measurement intervals must be constant.")


def add_economics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate economics at every recorded possible harvest time."""
    data = data.copy()

    # Convert hours to days to match the assignment's economic equations.
    data[TIME_D] = data[TIME_H] / 24.0
    cycle_time_d = data[TIME_D] + DOWNTIME_D

    # Starting culture cost remains constant throughout each batch.
    data[COST] = (
        10 * data[INITIAL_GLUCOSE]
        + 20 * data[INITIAL_GLUTAMINE]
        + 0.0005 * data[INITIAL_VCD]
    )

    # Culture value changes with antibody concentration.
    data[VALUE] = 1000 * data[MAB]
    data[PROFIT] = data[VALUE] - data[COST]

    # Include downtime when calculating production per litre per day.
    data[SPACE_TIME_PROFIT] = data[PROFIT] / cycle_time_d
    data[SPACE_TIME_YIELD] = data[MAB] / cycle_time_d

    return data


def summarize_best_observed_harvest(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Find the most profitable recorded harvest time for each batch."""
    # idxmax returns the row index of each batch's highest observed profit rate.
    # With time-sorted data, ties select the earliest recorded time.
    best_indices = data.groupby(BATCH_ID)[SPACE_TIME_PROFIT].idxmax()

    columns = [
        BATCH_ID, INITIAL_GLUCOSE, INITIAL_GLUTAMINE, INITIAL_VCD,
        TIME_D, MAB, COST, VALUE, PROFIT,
        SPACE_TIME_PROFIT, SPACE_TIME_YIELD,
    ]

    # This summarizes observations. It does not optimize between
    # measurements or estimate prediction intervals.
    return data.loc[best_indices, columns].rename(columns={
        TIME_D: "Best observed harvest time [d]"
    }).reset_index(drop=True)


def plot_mab_time_series(data: pd.DataFrame) -> plt.Figure:
    """Plot batch trajectories with colour indicating initial glucose."""
    fig, ax = plt.subplots(figsize=(9, 5))

    colormap = plt.cm.viridis
    normalizer = plt.Normalize(
        data[INITIAL_GLUCOSE].min(),
        data[INITIAL_GLUCOSE].max()
    )

    for _, batch_data in data.groupby(BATCH_ID):
        glucose = batch_data[INITIAL_GLUCOSE].iloc[0]

        ax.plot(
            batch_data[TIME_D],
            batch_data[MAB],
            color=colormap(normalizer(glucose)),
            alpha=0.7
        )

    fig.colorbar(
        plt.cm.ScalarMappable(norm=normalizer, cmap=colormap),
        ax=ax,
        label="Initial glucose [mM]"
    )

    # Glutamine and VCD also vary, so this plot does not isolate
    # glucose's effect on antibody production.
    ax.set_xlabel("Time [d]")
    ax.set_ylabel("mAb concentration [mg/L]")
    ax.set_title("Antibody production across the selected batches")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig

def split_by_condition(x, y, metadata):
    """Reserve complete starting-condition groups for final testing."""
    groups = metadata["Condition group"]

    # Replicate batches with identical starting conditions stay together.
    # test_size applies to the number of groups, not measurement rows.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(x, y, groups=groups)
    )

    x_train = x.iloc[train_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    metadata_train = metadata.iloc[train_indices].reset_index(drop=True)

    x_test = x.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    metadata_test = metadata.iloc[test_indices].reset_index(drop=True)

    return (
        x_train, x_test,
        y_train, y_test,
        metadata_train, metadata_test
    )


def forecast_batch(model, batch_data):
    """Forecast a complete batch from its stated initial conditions."""
    batch_data = batch_data.sort_values(TIME_H)
    first_row = batch_data.iloc[0]

    if not np.isclose(first_row[TIME_H], 0.0):
        raise ValueError("Full-batch evaluation requires a recorded t=0 row.")

    # Start from the conditions specified in initial_conditions.csv.
    # Lactate, ammonia and mAb initially equal zero; TCD equals VCD.
    initial_state = {
        GLUCOSE: first_row[INITIAL_GLUCOSE],
        GLUTAMINE: first_row[INITIAL_GLUTAMINE],
        VCD: first_row[INITIAL_VCD],
        TCD: first_row[INITIAL_VCD],
        LACTATE: 0.0,
        AMMONIA: 0.0,
        MAB: 0.0,
    }

    state = np.array(
        [initial_state[column] for column in PROCESS_COLUMNS],
        dtype=float
    )

    predictions = [state.copy()]

    # After initialization, use only predictions as future inputs.
    # Each iteration advances the forecast by one measurement interval.
    for _ in range(len(batch_data) - 1):
        model_input = pd.DataFrame(
            [state], columns=PROCESS_COLUMNS
        )
        state = model.predict(model_input)[0]

        if not np.isfinite(state).all():
            raise ValueError(
                "The recursive forecast became nonfinite. "
                "The model is unstable over this forecast horizon."
            )

        predictions.append(state.copy())

    return np.asarray(predictions)


def evaluate_recursive_var(x_train, y_train, metadata_train, data, model_template=None):
    """Evaluate full-batch forecasts using grouped cross-validation."""
    # Use linear VAR by default, or evaluate a supplied alternative.
    if model_template is None:
        model_template = make_pipeline(
            StandardScaler(),
            LinearRegression()
        )

    cv = GroupKFold(n_splits=N_CV_SPLITS)
    groups = metadata_train["Condition group"]

    score_records = []
    trajectory_tables = []

    for fold, (fit_indices, valid_indices) in enumerate(
        cv.split(x_train, y_train, groups=groups),
        start=1
    ):
        fold_model = clone(model_template)
        fold_model.fit(
            x_train.iloc[fit_indices],
            y_train.iloc[fit_indices]
        )

        # Evaluate only batches belonging to this validation fold.
        valid_batch_ids = metadata_train.iloc[
            valid_indices
        ][BATCH_ID].unique()

        for batch_id in valid_batch_ids:
            batch_data = (
                data.loc[data[BATCH_ID] == batch_id]
                .sort_values(TIME_H)
            )

            predicted = forecast_batch(fold_model, batch_data)
            observed = batch_data[PROCESS_COLUMNS].to_numpy()

            # Store trajectories for inspection and plotting.
            trajectory = pd.DataFrame({
                BATCH_ID: batch_id,
                "Fold": fold,
                TIME_H: batch_data[TIME_H].to_numpy(),
            })

            for column_index, variable in enumerate(PROCESS_COLUMNS):
                trajectory[f"Observed {variable}"] = (
                    observed[:, column_index]
                )
                trajectory[f"Predicted {variable}"] = (
                    predicted[:, column_index]
                )

                # Exclude t=0: this is supplied, rather than forecast.
                actual = observed[1:, column_index]
                forecast = predicted[1:, column_index]

                score_records.append({
                    "Fold": fold,
                    BATCH_ID: batch_id,
                    "Variable": variable,
                    "RMSE": np.sqrt(
                        mean_squared_error(actual, forecast)
                    ),
                    "R2": r2_score(actual, forecast),
                    "Negative predictions": int(
                        (forecast < 0).sum()
                    ),
                })

            trajectory_tables.append(trajectory)

    scores = pd.DataFrame(score_records)
    trajectories = pd.concat(trajectory_tables, ignore_index=True)

    # Each validation batch contributes equally to this summary.
    summary = (
        scores.groupby("Variable", sort=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            Negative_predictions=("Negative predictions", "sum"),
        )
        .reset_index()
    )

    return scores, summary, trajectories



def compare_knn_nvar(x_train, y_train, metadata_train, data):
    """Compare kNN settings using complete validation-batch forecasts."""
    comparison_records = []
    best_score = np.inf
    best_pipeline = None
    best_trajectories = None

    # A small starting grid keeps the comparison manageable.
    # Uniform weights average neighbouring next-step process states.
    for neighbours in [5, 15, 30]:

        # Scaling is essential because kNN uses distances and our
        # process variables have very different numerical ranges.
        pipeline = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=neighbours,
                weights="uniform"
            )
        )

        scores, summary, trajectories = evaluate_recursive_var(
            x_train,
            y_train,
            metadata_train,
            data,
            model_template=pipeline
        )

        # Select settings using mAb because it directly determines
        # the assignment's culture value and profit.
        mab_scores = scores.loc[scores["Variable"] == MAB]
        mean_rmse = mab_scores["RMSE"].mean()

        comparison_records.append({
            "Neighbours": neighbours,
            "Mean batch mAb RMSE [mg/L]": mean_rmse,
            "SD batch mAb RMSE [mg/L]": mab_scores["RMSE"].std(),
            "Mean batch mAb R2": mab_scores["R2"].mean(),
        })

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_pipeline = clone(pipeline)
            best_trajectories = trajectories

    comparison = pd.DataFrame(comparison_records).sort_values(
        "Mean batch mAb RMSE [mg/L]"
    ).reset_index(drop=True)

    # Fit the selected kNN model on all development data.
    # Reserved test conditions remain untouched.
    best_pipeline.fit(x_train, y_train)

    return best_pipeline, comparison, best_trajectories

def reduced_polynomial_features(x):
    """Create quadratic features plus a cubic term for time only."""
    values = np.asarray(x, dtype=float)

    # Inputs have already been scaled by the pipeline.
    # Keep all linear terms, squares and pairwise interactions.
    quadratic = PolynomialFeatures(
        degree=2,
        include_bias=False,
    ).fit_transform(values)

    # Time is the fourth input column.
    # Add time cubed without introducing cubic condition terms.
    time_cubed = values[:, 3] ** 3

    return np.column_stack([quadratic, time_cubed])

def compare_direct_regression(data, metadata_train):
    """Compare direct mAb models using the existing validation folds."""
    features = [
        INITIAL_GLUCOSE,
        INITIAL_GLUTAMINE,
        INITIAL_VCD,
        TIME_D,
    ]

    # Restrict all fitting and validation to development batches.
    development = data.loc[
        data[BATCH_ID].isin(metadata_train[BATCH_ID].unique())
    ].copy()

    # Degree 1 is ordinary linear regression. Degree 2 also includes
    # squared terms and interactions between conditions and time.
    templates = {
        f"Direct degree {degree}": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        )
        for degree in [1, 2, 3]
    }

    # Retain cubic flexibility in time while reducing the number
    # of coefficients describing initial-condition effects.
    templates["Reduced polynomial"] = make_pipeline(
        StandardScaler(),
        FunctionTransformer(
            reduced_polynomial_features,
            validate=False,
        ),
        LinearRegression(),
    )

    # Direct kNN predicts mAb from initial conditions and time.
    # Scaling prevents VCD's large numerical values from dominating distances.
    for neighbours in [5, 15, 30]:
        for weights in ["uniform", "distance"]:
            name = f"Direct kNN k={neighbours}, {weights}"
            templates[name] = make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(
                    n_neighbors=neighbours,
                    weights=weights,
                ),
            )

    cv = GroupKFold(n_splits=N_CV_SPLITS)
    score_records = []
    trajectory_tables = []

    # Split the ORIGINAL metadata to reproduce the VAR validation folds.
    # Then select complete batch trajectories for direct regression.
    for fold, (fit_indices, valid_indices) in enumerate(
        cv.split(
            metadata_train,
            groups=metadata_train["Condition group"],
        ),
        start=1,
    ):
        fit_ids = metadata_train.iloc[fit_indices][BATCH_ID].unique()
        valid_ids = metadata_train.iloc[valid_indices][BATCH_ID].unique()

        fit_data = development.loc[development[BATCH_ID].isin(fit_ids)]
        valid_data = development.loc[
            development[BATCH_ID].isin(valid_ids)
        ].sort_values([BATCH_ID, TIME_H])

        # Reference: use the training batches' average mAb at each time.
        # It deliberately ignores differences in initial conditions.
        mean_trajectory = fit_data.groupby(TIME_H)[MAB].mean()
        reference = valid_data[TIME_H].map(mean_trajectory)

        if reference.isna().any():
            raise ValueError(
                "The mean-trajectory reference requires matching time grids."
            )

        predictions = {"Mean trajectory": reference.to_numpy()}

        for name, template in templates.items():
            model = clone(template)

            # The pipeline learns scaling and regression coefficients
            # from this fold's training batches only.
            model.fit(fit_data[features], fit_data[MAB])
            predictions[name] = model.predict(valid_data[features])

        for name, predicted in predictions.items():
            if not np.isfinite(predicted).all():
                raise ValueError(f"{name} produced nonfinite predictions.")

            trajectory = valid_data[[BATCH_ID, TIME_H]].copy()
            trajectory["Fold"] = fold
            trajectory["Model"] = name
            trajectory[f"Observed {MAB}"] = valid_data[MAB].to_numpy()
            trajectory[f"Predicted {MAB}"] = predicted
            trajectory_tables.append(trajectory)

            for batch_id, batch in trajectory.groupby(BATCH_ID):
                # Match recursive evaluation: exclude the initial row.
                # Predictions remain unmodified, including negative ones.
                scored = batch.iloc[1:]
                actual = scored[f"Observed {MAB}"]
                forecast = scored[f"Predicted {MAB}"]

                score_records.append({
                    "Model": name,
                    "Fold": fold,
                    BATCH_ID: batch_id,
                    "RMSE": np.sqrt(mean_squared_error(actual, forecast)),
                    "R2": r2_score(actual, forecast),
                    "Negative predictions": int((forecast < 0).sum()),
                })

    scores = pd.DataFrame(score_records)
    trajectories = pd.concat(trajectory_tables, ignore_index=True)

    # Give each validation batch equal weight, as in the VAR comparison.
    comparison = (
        scores.groupby("Model", sort=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            Negative_predictions=("Negative predictions", "sum"),
        )
        .reset_index()
        .sort_values("RMSE_mean")
        .reset_index(drop=True)
    )

    # Fit candidate models on development data only.
    # This does not select a final model or evaluate the test set.
    fitted_models = {}
    for name, template in templates.items():
        model = clone(template)
        model.fit(development[features], development[MAB])
        fitted_models[name] = model

    return fitted_models, comparison, scores, trajectories

def evaluate_harvest_decisions(
    trajectories,
    data,
    model_name="Direct degree 2",
):
    """Evaluate harvest decisions using held-out validation predictions."""
    selected = trajectories.loc[
        trajectories["Model"] == model_name
    ].copy()

    if selected.empty:
        raise ValueError(f"No validation predictions found for {model_name}.")

    # Attach the known starting cost without using additional measurements.
    costs = data[[BATCH_ID, COST]].drop_duplicates()

    selected = selected.merge(
        costs,
        on=BATCH_ID,
        how="left",
        validate="many_to_one",
    )

    if selected[COST].isna().any():
        raise ValueError("Starting costs are missing for some batches.")

    selected[TIME_D] = selected[TIME_H] / 24.0

    # Consider recorded harvest times after startup.
    selected = selected.loc[selected[TIME_D] > 0].copy()
    cycle_time = selected[TIME_D] + DOWNTIME_D

    observed_profit = "Observed space-time profit [$/(L*d)]"
    predicted_profit = "Predicted space-time profit [$/(L*d)]"

    selected[observed_profit] = (
        1000 * selected[f"Observed {MAB}"] - selected[COST]
    ) / cycle_time

    selected[predicted_profit] = (
        1000 * selected[f"Predicted {MAB}"] - selected[COST]
    ) / cycle_time

    records = []

    for batch_id, batch in selected.groupby(BATCH_ID):
        batch = batch.sort_values(TIME_H)

        # Choose a harvest using predictions only.
        # With sorted times, ties select the earliest measurement.
        chosen = batch.loc[batch[predicted_profit].idxmax()]

        # Observations provide a retrospective reference for evaluation.
        observed_best = batch.loc[batch[observed_profit].idxmax()]

        records.append({
            BATCH_ID: batch_id,
            "Fold": int(chosen["Fold"]),
            "Model-selected harvest [d]": chosen[TIME_D],
            "Best observed harvest [d]": observed_best[TIME_D],
            "Predicted STP at selected harvest": chosen[predicted_profit],
            "Observed STP at selected harvest": chosen[observed_profit],
            "Best observed STP": observed_best[observed_profit],
            # Positive error means the model overestimated profitability.
            "STP prediction error": (
                chosen[predicted_profit] - chosen[observed_profit]
            ),
            # Observed opportunity lost by selecting a different time.
            "Observed STP gap": (
                observed_best[observed_profit] - chosen[observed_profit]
            ),
        })

    return pd.DataFrame(records).sort_values(
        "Observed STP gap",
        ascending=False,
    ).reset_index(drop=True)

def evaluate_direct_test(model, data, metadata_test, model_name="Direct degree 3"):
    """Assess a fixed model on batches excluded from its fit.

    In this workflow these are previously examined follow-up batches,
    so their scores must not be presented as an independent final test.
    """
    features = [
        INITIAL_GLUCOSE,
        INITIAL_GLUTAMINE,
        INITIAL_VCD,
        TIME_D,
    ]

    # Select complete trajectories for the reserved batches.
    test_data = (
        data.loc[
            data[BATCH_ID].isin(metadata_test[BATCH_ID].unique())
        ]
        .sort_values([BATCH_ID, TIME_H])
        .reset_index(drop=True)
    )

    if test_data.empty:
        raise ValueError("No reserved test batches were found.")

    # Predict using the model already fitted on development data.
    predicted = model.predict(test_data[features])

    if not np.isfinite(predicted).all():
        raise ValueError("The test predictions contain nonfinite values.")

    trajectories = test_data[[BATCH_ID, TIME_H]].copy()
    trajectories["Model"] = model_name

    # The harvest evaluator expects a Fold column.
    # Zero identifies the final test evaluation rather than a CV fold.
    trajectories["Fold"] = 0
    trajectories[f"Observed {MAB}"] = test_data[MAB].to_numpy()
    trajectories[f"Predicted {MAB}"] = predicted

    records = []

    for batch_id, batch in trajectories.groupby(BATCH_ID):
        # Match validation scoring by excluding the startup measurement.
        scored = batch.iloc[1:]
        actual = scored[f"Observed {MAB}"]
        forecast = scored[f"Predicted {MAB}"]

        records.append({
            BATCH_ID: batch_id,
            "RMSE": np.sqrt(mean_squared_error(actual, forecast)),
            "R2": r2_score(actual, forecast),
            "Negative predictions": int((forecast < 0).sum()),
        })

    scores = pd.DataFrame(records)

    # Summarize with equal weight for each batch, matching validation.
    summary = pd.DataFrame([{
        "Model": model_name,
        "Test batches": len(scores),
        "RMSE_mean": scores["RMSE"].mean(),
        "RMSE_std": scores["RMSE"].std(),
        "R2_mean": scores["R2"].mean(),
        "Negative predictions": scores["Negative predictions"].sum(),
    }])

    harvest_results = evaluate_harvest_decisions(
        trajectories,
        data,
        model_name=model_name,
    )

    return scores, summary, trajectories, harvest_results

def check_polynomial_design(data, metadata_train):
    """Check whether polynomial coefficients can be identified."""
    features = [
        INITIAL_GLUCOSE,
        INITIAL_GLUTAMINE,
        INITIAL_VCD,
        TIME_D,
    ]

    # Reproduce the same training subsets used during validation.
    cv = GroupKFold(n_splits=N_CV_SPLITS)
    subsets = {
        "All development": metadata_train[BATCH_ID].unique()
    }

    for fold, (fit_indices, _) in enumerate(
        cv.split(
            metadata_train,
            groups=metadata_train["Condition group"],
        ),
        start=1,
    ):
        subsets[f"Fold {fold} training"] = (
            metadata_train.iloc[fit_indices][BATCH_ID].unique()
        )

    records = []

    for subset_name, batch_ids in subsets.items():
        inputs = data.loc[
            data[BATCH_ID].isin(batch_ids), features
        ]

        # Fit scaling separately within each training subset.
        scaled = StandardScaler().fit_transform(inputs)

        for degree in [2, 3, "Reduced"]:
            # Include the intercept when checking identifiability.
            if degree == "Reduced":
                expanded = reduced_polynomial_features(scaled)
                design = np.column_stack([
                    np.ones(len(scaled)),
                    expanded,
                ])
            else:
                design = PolynomialFeatures(
                    degree=degree,
                    include_bias=True,
                ).fit_transform(scaled)

            # Singular values measure independent information in the
            # feature matrix. Near-zero values indicate redundant terms.
            singular_values = np.linalg.svd(
                design, compute_uv=False
            )
            tolerance = (
                singular_values[0]
                * max(design.shape)
                * np.finfo(float).eps
            )

            rank = int((singular_values > tolerance).sum())
            coefficients = design.shape[1]

            # A rank-deficient matrix has no unique coefficient solution.
            condition_number = (
                singular_values[0] / singular_values[-1]
                if rank == coefficients
                else np.inf
            )

            records.append({
                "Subset": subset_name,
                "Degree": degree,
                "Distinct conditions": (
                    inputs[features[:3]].drop_duplicates().shape[0]
                ),
                "Coefficients": coefficients,
                "Rank": rank,
                "Rank deficient": rank < coefficients,
                "Condition number": condition_number,
            })

    return pd.DataFrame(records)

def search_operating_conditions(model, initial_conditions, data):
    """Rank candidate conditions and harvest times by predicted profit."""
    condition_columns = [
        INITIAL_GLUCOSE,
        INITIAL_GLUTAMINE,
        INITIAL_VCD,
    ]
    features = condition_columns + [TIME_D]

    # Use the supplied starting-condition combinations.
    # Reading their input values does not require downloading more batches.
    conditions = (
        initial_conditions[[GLUCOSE, GLUTAMINE, VCD]]
        .drop_duplicates()
        .rename(columns={
            GLUCOSE: INITIAL_GLUCOSE,
            GLUTAMINE: INITIAL_GLUTAMINE,
            VCD: INITIAL_VCD,
        })
        .sort_values(condition_columns)
        .reset_index(drop=True)
    )

    # Stay within the measured time horizon and sampling grid.
    # Exclude startup because it is not a production harvest.
    harvest_hours = np.sort(
        data.loc[data[TIME_H] > 0, TIME_H].unique()
    )

    # Pair every initial-condition combination with every harvest time.
    candidates = conditions.loc[
        conditions.index.repeat(len(harvest_hours))
    ].reset_index(drop=True)

    candidates[TIME_H] = np.tile(
        harvest_hours, len(conditions)
    )
    candidates[TIME_D] = candidates[TIME_H] / 24.0

    # Predict directly from nominal starting conditions and harvest time.
    candidates[MAB] = model.predict(candidates[features])

    if not np.isfinite(candidates[MAB]).all():
        raise ValueError("The operating search produced nonfinite predictions.")

    # Negative concentrations are physically invalid.
    # Exclude them from recommendations without changing their values.
    negative_count = int((candidates[MAB] < 0).sum())
    valid = candidates.loc[candidates[MAB] >= 0].copy()

    if valid.empty:
        raise ValueError("No candidates have nonnegative predicted mAb.")

    # Reuse the same economic equations as the observed-data analysis.
    valid = add_economics(valid)

    # Retain the best recorded harvest time for each condition.
    # Sorting first makes ties select the earlier harvest.
    valid = valid.sort_values(condition_columns + [TIME_H])
    best_indices = valid.groupby(
        condition_columns
    )[SPACE_TIME_PROFIT].idxmax()

    columns = condition_columns + [
        TIME_D,
        MAB,
        COST,
        VALUE,
        PROFIT,
        SPACE_TIME_PROFIT,
        SPACE_TIME_YIELD,
    ]

    ranked = (
        valid.loc[best_indices, columns]
        .sort_values(
            [SPACE_TIME_PROFIT, TIME_D],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return ranked, negative_count, len(candidates)

def estimate_prediction_intervals(
    model,
    candidates,
    validation_trajectories,
    followup_trajectories,
    model_name="Reduced polynomial",
):
    """Estimate approximate prediction intervals from validation errors."""
    features = [
        INITIAL_GLUCOSE,
        INITIAL_GLUTAMINE,
        INITIAL_VCD,
        TIME_D,
    ]
    observed_column = f"Observed {MAB}"
    predicted_column = f"Predicted {MAB}"

    # Use out-of-fold development predictions rather than training errors.
    # Exclude startup to match the model-performance calculations.
    validation = validation_trajectories.loc[
        (validation_trajectories["Model"] == model_name)
        & (validation_trajectories[TIME_H] > 0)
    ].copy()

    if validation.empty:
        raise ValueError("No validation residuals were found.")

    # Observed minus predicted: add these errors to future predictions.
    residuals = (
        validation[observed_column] - validation[predicted_column]
    ).to_numpy()

    if not np.isfinite(residuals).all():
        raise ValueError("Validation residuals must be finite.")

    # Approximate pointwise intervals: correlated rows and prior model selection
    # preclude a guaranteed 95% coverage claim or uncertainty for the optimum.
    # Empirical endpoints of the central 95% residual distribution.
    # These describe prediction errors, not uncertainty in the mean.
    lower_error, upper_error = np.quantile(
        residuals, [0.025, 0.975]
    )

    results = candidates.copy()

    # The model uses days; add_economics expects hours.
    results[TIME_H] = results[TIME_D] * 24.0

    results[MAB] = model.predict(results[features])
    results = add_economics(results)

    lower_mab = results[MAB] + lower_error
    upper_mab = results[MAB] + upper_error

    # Preserve raw bounds so any physically impossible bounds remain visible.
    results["mAb lower 95% [mg/L]"] = lower_mab
    results["mAb upper 95% [mg/L]"] = upper_mab

    cycle_time = results[TIME_D] + DOWNTIME_D

    # Cost is fixed by the specified initial conditions.
    # The economic equations are increasing linear functions of mAb,
    # so transform both concentration endpoints directly.
    results["Value lower 95% [$/L]"] = 1000 * lower_mab
    results["Value upper 95% [$/L]"] = 1000 * upper_mab

    results["STP lower 95% [$/(L*d)]"] = (
        1000 * lower_mab - results[COST]
    ) / cycle_time
    results["STP upper 95% [$/(L*d)]"] = (
        1000 * upper_mab - results[COST]
    ) / cycle_time

    results["STY lower 95% [mg/(L*d)]"] = lower_mab / cycle_time
    results["STY upper 95% [mg/(L*d)]"] = upper_mab / cycle_time

    # Assess the unchanged interval rule on follow-up trajectories.
    # These batches are not used to calculate the residual endpoints.
    followup = followup_trajectories.loc[
        (followup_trajectories["Model"] == model_name)
        & (followup_trajectories[TIME_H] > 0)
    ].copy()

    if followup.empty:
        raise ValueError("No follow-up trajectories were found.")

    followup["Lower 95% [mg/L]"] = (
        followup[predicted_column] + lower_error
    )
    followup["Upper 95% [mg/L]"] = (
        followup[predicted_column] + upper_error
    )
    followup["Covered"] = (
        (followup[observed_column] >= followup["Lower 95% [mg/L]"])
        & (followup[observed_column] <= followup["Upper 95% [mg/L]"])
    )

    # Report coverage per batch because measurements within a batch
    # are related; they are not independent experimental runs.
    coverage = (
        followup.groupby(BATCH_ID)
        .agg(
            Measurements=("Covered", "size"),
            Coverage=("Covered", "mean"),
        )
        .reset_index()
    )

    return results, coverage, lower_error, upper_error

def plot_harvest_economics(
    model, recommendation, data, lower_error, upper_error
):
    """Plot harvest economics at the recommended initial conditions."""
    features = [
        INITIAL_GLUCOSE,
        INITIAL_GLUTAMINE,
        INITIAL_VCD,
        TIME_D,
    ]
    best = recommendation.iloc[0]

    # Use the same positive harvest times as the operating search.
    times_h = np.sort(data.loc[data[TIME_H] > 0, TIME_H].unique())
    curve = pd.DataFrame({
        INITIAL_GLUCOSE: best[INITIAL_GLUCOSE],
        INITIAL_GLUTAMINE: best[INITIAL_GLUTAMINE],
        INITIAL_VCD: best[INITIAL_VCD],
        TIME_H: times_h,
    })
    curve[TIME_D] = curve[TIME_H] / 24.0
    curve[MAB] = model.predict(curve[features])
    curve = add_economics(curve)

    cycle_time = curve[TIME_D].to_numpy() + DOWNTIME_D
    predicted_stp = curve[SPACE_TIME_PROFIT].to_numpy()
    lower_stp = predicted_stp + 1000 * lower_error / cycle_time
    upper_stp = predicted_stp + 1000 * upper_error / cycle_time
    days = curve[TIME_D].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(days, predicted_stp, label="Predicted space-time profit")
    ax.fill_between(
        days, lower_stp, upper_stp,
        alpha=0.2,
        label="Approximate pointwise 95% prediction interval",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(
        best[TIME_D], color="tab:red", linestyle="--",
        label=f"Selected harvest: {best[TIME_D]:g} d",
    )

    ax.set_xlabel("Harvest time [d]")
    ax.set_ylabel("Space-time profit [$/L/day]")
    ax.set_title(
        "Harvest economics at recommended initial conditions\n"
        f"Glucose {best[INITIAL_GLUCOSE]:g} mM; "
        f"glutamine {best[INITIAL_GLUTAMINE]:g} mM; "
        f"VCD {best[INITIAL_VCD]:,.0f} cells/mL"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_initial_condition_effects(model, recommendation, initial_conditions):
    """Vary one initial condition at a time at a fixed harvest time."""
    condition_columns = [
        INITIAL_GLUCOSE,
        INITIAL_GLUTAMINE,
        INITIAL_VCD,
    ]
    features = condition_columns + [TIME_D]
    source_columns = [GLUCOSE, GLUTAMINE, VCD]
    best = recommendation.iloc[0]

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 7), sharex="col"
    )

    for column_index, (condition, source) in enumerate(
        zip(condition_columns, source_columns)
    ):
        # Use supplied input levels; hold other conditions fixed.
        levels = np.sort(initial_conditions[source].unique())
        inputs = pd.DataFrame({
            feature: np.full(len(levels), best[feature])
            for feature in features
        })
        inputs[condition] = levels
        inputs[TIME_H] = inputs[TIME_D] * 24.0
        inputs[MAB] = model.predict(inputs[features])
        results = add_economics(inputs)

        # Display cell density in millions for readable axis labels.
        scale = 1_000_000.0 if condition == INITIAL_VCD else 1.0
        x_values = levels / scale
        label = (
            "Initial VCD [million cells/mL]"
            if condition == INITIAL_VCD else condition
        )

        for row_index, metric in enumerate(
            [SPACE_TIME_PROFIT, SPACE_TIME_YIELD]
        ):
            ax = axes[row_index, column_index]
            ax.plot(x_values, results[metric], marker="o")
            ax.axvline(
                best[condition] / scale,
                color="tab:red", linestyle="--", alpha=0.7,
            )
            ax.grid(alpha=0.3)

        axes[1, column_index].set_xlabel(label)

    axes[0, 0].set_ylabel("Predicted space-time profit [$/L/day]")
    axes[1, 0].set_ylabel("Predicted space-time yield [mg/L/day]")
    fig.suptitle(
        f"Model sensitivity at a fixed {best[TIME_D]:g}-day harvest\n"
        "One input varies; others stay fixed. Red dashed lines: recommendation"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_followup_predictions(trajectories, lower_error, upper_error):
    """Show observed and predicted mAb for every follow-up batch."""
    batch_ids = sorted(trajectories[BATCH_ID].unique())
    ncols = 2
    nrows = (len(batch_ids) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(11, 2.7 * nrows),
        squeeze=False,
    )

    for ax, batch_id in zip(axes.flat, batch_ids):
        batch = trajectories.loc[
            trajectories[BATCH_ID] == batch_id
        ].sort_values(TIME_H)

        days = batch[TIME_H].to_numpy() / 24.0
        predicted = batch[f"Predicted {MAB}"].to_numpy()

        ax.scatter(
            days, batch[f"Observed {MAB}"],
            s=10, alpha=0.65, label="Observed",
        )
        ax.plot(days, predicted, color="tab:orange", label="Predicted")

        # Exclude startup from the residual-based prediction band,
        # matching the measurements used to estimate its endpoints.
        positive = days > 0
        ax.fill_between(
            days[positive],
            predicted[positive] + lower_error,
            predicted[positive] + upper_error,
            color="tab:orange", alpha=0.2,
            label="Approximate pointwise 95% PI",
        )

        ax.set_title(f"Batch {batch_id}")
        ax.set_xlabel("Time [d]")
        ax.set_ylabel("mAb [mg/L]")
        ax.grid(alpha=0.3)

    for ax in list(axes.flat)[len(batch_ids):]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=3, fontsize=8,
    )
    fig.suptitle("Reduced polynomial: follow-up trajectories")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig