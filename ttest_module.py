import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from a import (
    METRICS_TTEST_THRESHOLD_GAN,
    METRICS_TTEST_THRESHOLD_GHOST,
    METRICS_TTEST_THRESHOLD_GHOST_BG_SEG_DIRT,
    METRICS_TTEST_THRESHOLD_SHRP_CWSSIM,
    MetricSetting,
)
from b import EvalType
from c import EvalSetup
from d import get_plot_headers
from e import (
    PlotSettings,
    add_ttest_vlines,
    get_hue,
    get_metric_plot_params,
    plot_ttest_stats,
)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats


def get_ttest_metric_plot_data(
    eval_setup: EvalSetup, eval_type: EvalType
) -> Tuple[Dict[str, Dict[str, any]], Tuple[MetricSetting]]:

    metric_plot_data = get_metric_plot_params(
        metrics=eval_setup.metrics, data_range=eval_setup.data_range
    )
    XLIM = 0.75
    thresholds = {
        MetricSetting.SHARPNESS: METRICS_TTEST_THRESHOLD_SHRP_CWSSIM,
        MetricSetting.CWSSIM: METRICS_TTEST_THRESHOLD_SHRP_CWSSIM,
    }

    value_range_settings = {
        MetricSetting.SHARPNESS: (-XLIM, XLIM),
        MetricSetting.CWSSIM: (-XLIM, XLIM),
    }

    # Define metrics based on eval_type
    metric_candidates = {
        EvalType.B: (
            MetricSetting.SHARPNESS,
            MetricSetting.CWSSIM,
        ),
        EvalType.A: (
            MetricSetting.SHARPNESS,
            MetricSetting.CWSSIM,
        ),
    }

    # Check for each metric if it exists in eval_setup.metrics and include it accordingly
    if eval_type in metric_candidates:
        metrics_to_include = tuple(
            metric
            for metric in metric_candidates[eval_type]
            if metric in eval_setup.metrics
        )
    else:
        raise ValueError(f"EvalType {eval_type} not implemented")

    # Update thresholds and settings if the metric exists
    for metric, threshold in thresholds.items():
        if metric in eval_setup.metrics:
            metric_plot_data[metric]["threshold"] = threshold

    for metric, value_range in value_range_settings.items():
        if metric in eval_setup.metrics:
            metric_plot_data[metric]["value_range"] = value_range

    # Set "equal_to_ref" for specific metrics
    for metric in [MetricSetting.CWSSIM, MetricSetting.CWSSIM_GHOST]:
        if metric in eval_setup.metrics:
            metric_plot_data[metric]["equal_to_ref"] = 0

    return metric_plot_data, metrics_to_include


def calculate_delta_metrics(
    metric_table: pd.DataFrame,
    metric_plot_data: Dict[str, Dict[str, any]],
    reference_dataset: str,
    comparand_dataset: str,
    reference_model: str,
    comparand_model: str,
    metrics_to_include: Tuple[MetricSetting],
    confidence_level: float,
    column_type: str,
    ttest_alternative: str,
):
    logging.info(f"Calculating T-test statistics")
    metrics = pd.DataFrame(columns=metric_table.columns)
    metric_types = [metric_type for metric_type in metric_table[column_type].unique()]
    for metric_type in metric_types:
        threshold_delta = metric_plot_data[metric_type]["threshold"]
        metric_table_results, metrics_delta = _calculate_metric_stats(
            metric_table=metric_table,
            metric_type=metric_type,
            reference_dataset=reference_dataset,
            comparand_dataset=comparand_dataset,
            reference_model=reference_model,
            comparand_model=comparand_model,
        )
        metrics = metrics.append(metric_table_results)
        # Skip t-test if metric type is not in the metrics_to_include
        if metric_type not in metrics_to_include:
            continue
        # GAN artifact handled on image metric instead of pairwise images delta metrics. Hence it needs to be greater than
        if metric_type == MetricSetting.GAN_ARTIFACT_2:
            ttest_alternative_metric = "greater"
        else:
            ttest_alternative_metric = ttest_alternative

        p_value, confidence_interval = _ttest(
            metrics_delta=metrics_delta,
            thrs=threshold_delta,
            alternative=ttest_alternative_metric,
            confidence_level=confidence_level,
        )

        metric_plot_data[metric_type]["p_value"] = p_value
        if p_value < 1 - confidence_level:
            metric_plot_data[metric_type]["passed"] = False
            logging.warning(
                f"The {metric_type} metric test failed! The root cause needs to be investigated. (T-test used stats.ttest_1samp(popmean={threshold_delta},alt={ttest_alternative}) with {100*confidence_level}% confidence, distribution mean={np.mean(confidence_interval):.3f}. p={p_value:.3f})"
            )
        else:
            metric_plot_data[metric_type]["passed"] = True
        metric_plot_data[metric_type]["confidence_interval"] = confidence_interval
        metric_plot_data[metric_type]["failed"] = len(
            metric_table_results[
                (
                    metric_table_results["metric_value"] < threshold_delta
                    if metric_type != MetricSetting.GAN_ARTIFACT_2
                    else metric_table_results["metric_value"] > threshold_delta
                )
            ]
        )

    metric_plot_data["matched_samples"] = len(metric_table_results)

    return metrics, metric_plot_data


def _get_filtered_df(table, dataset_name: str, model_name: str):
    key = (table["dataset"] == dataset_name) & (table["model"] == model_name)
    return table[key]


def _calculate_metric_stats(
    metric_table: pd.DataFrame,
    metric_type: str,
    reference_dataset: str,
    comparand_dataset: str,
    reference_model: str,
    comparand_model: str,
):
    """
    Calculate the difference between the datasets for the given metric_type. To be used in t-test calculations.

    Args:
    - metric_table (pd.DataFrame): DataFrame containing the datasets, metric types, and metric values etc.
    - metric_type (str): Metric type column name.

    Returns:
    - metric_table_results (pd.DataFrame): DataFrame containing results of the difference between the datasets for the given metric_type
    - metrics_delta (np.array): Differences between the datasets for the given metric_type
    """
    filtered_table = metric_table[metric_table["metric_type"] == metric_type]

    # Filter for comparand dataset and reference dataset
    reference_data = _get_filtered_df(
        filtered_table,
        dataset_name=reference_dataset,
        model_name=reference_model,
    )

    comparand_data = _get_filtered_df(
        filtered_table,
        dataset_name=comparand_dataset,
        model_name=comparand_model,
    )

    reference_name, comparand_name = get_reference_comparand(
        reference_dataset=reference_dataset,
        comparand_dataset=comparand_dataset,
        reference_model=reference_model,
        comparand_model=comparand_model,
    )

    # Find the common "slide_set", "slide_id" and "sample_id" between both datasets
    common_keys = pd.merge(
        comparand_data[["slide_set", "slide_id", "sample_id"]],
        reference_data[["slide_set", "slide_id", "sample_id"]],
        on=["slide_set", "slide_id", "sample_id"],
        how="inner",
    )

    # Filter the original table to keep only rows with matching "slide_set", "slide_id" and "sample_id"
    merged_table = filtered_table.merge(
        common_keys, on=["slide_set", "slide_id", "sample_id"], how="inner"
    )

    filtered_table = merged_table.sort_values(by=["slide_set", "sample_id", "slide_id"])
    metric_table_results = pd.DataFrame(columns=metric_table.columns)

    # Special case: GAN artifact handled on images metric instead of pairwise images delta metrics
    # only plots it in the case of dataset comparison
    if metric_type == MetricSetting.GAN_ARTIFACT_2 and not filtered_table.empty:
        metric_table_results = filtered_table[
            filtered_table["dataset"] == comparand_name
        ]
        metrics_delta = metric_table_results["metric_value"].to_list()

    elif (
        filtered_table.empty
    ):  # special case: repeatability study, e.g. 2 datasets and one with 1 slide scanned 5 times
        filtered_table = filtered_table.sort_values(by=["sample_id", "slide_id"])
        dataset_counts = filtered_table["slide_id"].value_counts()
        # TODO piag we need to fix this because now it is 2 models too
        dataset_1, dataset_2 = filtered_table["dataset"].unique()

        slide_1 = dataset_counts.idxmax()
        slide_2 = dataset_counts.idxmin()

        if reference_dataset == slide_1:
            slide_1, slide_2 = slide_2, slide_1
        df_most_used = filtered_table[filtered_table["slide_id"] == slide_1]
        df_rest = filtered_table[filtered_table["slide_id"] != slide_1]

        metrics_1 = df_most_used.groupby("sample_id")["metric_value"].mean().tolist()
        metrics_2 = df_rest.groupby("sample_id")["metric_value"].mean().tolist()
        metrics_delta = np.subtract(metrics_1, metrics_2)
        dataset_key = f"Subtract({comparand_name}, {reference_name})"

        metric_table_results = filtered_table[
            filtered_table["slide_id"] == slide_2
        ].copy()
        metric_table_results.loc[:, ("dataset")] = dataset_key
        metric_table_results.loc[:, ("slide_id")] = slide_1
        metric_table_results.loc[:, ("metric_value")] = metrics_delta

    else:
        reference_data = _get_filtered_df(
            filtered_table,
            dataset_name=reference_dataset,
            model_name=reference_model,
        )
        metrics_reference = reference_data["metric_value"].tolist()
        comparand_data = _get_filtered_df(
            filtered_table,
            dataset_name=comparand_dataset,
            model_name=comparand_model,
        )
        metrics_instrument = comparand_data["metric_value"].tolist()

        metrics_delta = np.subtract(metrics_instrument, metrics_reference)
        dataset_key = f"Subtract({comparand_name}, {reference_name})"
        metric_table_results = comparand_data.copy()
        metric_table_results.loc[:, ("dataset")] = dataset_key
        metric_table_results.loc[:, ("metric_value")] = metrics_delta

    return metric_table_results, metrics_delta


def _ttest(
    metrics_delta,
    thrs: float = 0.0,
    confidence_level: float = 0.95,
    alternative: str = "less",
):
    """Pairwise t-test to find the metric value where two datasets are diverging with 95% confidence for a specific metric type."""

    t_stat, p_value = stats.ttest_1samp(
        metrics_delta, popmean=thrs, alternative=alternative
    )
    confidence_interval = stats.t.interval(
        confidence_level,
        df=len(metrics_delta) - 1,
        loc=np.mean(metrics_delta),
        scale=stats.sem(metrics_delta),
    )
    return p_value, confidence_interval


def convert_metrics_to_dict(metrics: pd.DataFrame) -> List[Dict[str, any]]:
    df = metrics.drop(columns=["dataset", "model"])
    df_pivoted = df.pivot_table(
        index=["slide_set", "slide_id", "sample_id"],
        columns="metric_type",
        values="metric_value",
        aggfunc="mean",
    ).reset_index()
    df_pivoted.columns.name = None
    # Move metric_type columns to the start
    metric_columns = [
        col
        for col in df_pivoted.columns
        if col not in ["slide_set", "slide_id", "sample_id"]
    ]
    other_columns = ["slide_set", "slide_id", "sample_id"]
    new_order = metric_columns + other_columns
    all_metrics_results = df_pivoted[new_order].to_dict(orient="records")
    return all_metrics_results


def plot_ttest_results(
    save_path: Path,
    metric_table: pd.DataFrame,
    metric_plot_data: Dict[str, Dict[str, any]],
    metrics_to_include: Tuple[MetricSetting],
    reference_dataset: str,
    comparand_dataset: str,
    reference_model: str,
    comparand_model: str,
    column_type: str,
    confidence_level: float,
    zoomed_quantiles: List[float],
    save_suffix_zoomed: str,
    save_image_extension: str,
    plot_settings: PlotSettings,
) -> None:

    plot_suffix = get_plot_suffix(
        reference_dataset=reference_dataset,
        comparand_dataset=comparand_dataset,
        reference_model=reference_model,
        comparand_model=comparand_model,
    )

    logging.info("Creating zoomed all in one histograms")
    _plot_metrics_grid(
        save_path=save_path,
        metric_table=metric_table,
        metric_plot_data=metric_plot_data,
        metrics_to_include=metrics_to_include,
        reference_dataset=reference_dataset,
        comparand_dataset=comparand_dataset,
        reference_model=reference_model,
        comparand_model=comparand_model,
        confidence_level=confidence_level,
        column_type=column_type,
        row_type=None,
        hue_type=["model", "dataset"],
        save_suffix=plot_suffix + save_suffix_zoomed,
        save_image_extension=save_image_extension,
        plot_settings=plot_settings,
        zoomed_quantiles=zoomed_quantiles,
    )
    logging.info("Creating all in one histograms")
    _plot_metrics_grid(
        save_path=save_path,
        metric_table=metric_table,
        metric_plot_data=metric_plot_data,
        metrics_to_include=metrics_to_include,
        reference_dataset=reference_dataset,
        comparand_dataset=comparand_dataset,
        reference_model=reference_model,
        comparand_model=comparand_model,
        confidence_level=confidence_level,
        column_type=column_type,
        row_type=None,
        hue_type=["model", "dataset"],
        save_suffix=plot_suffix,
        save_image_extension=save_image_extension,
        plot_settings=plot_settings,
    )


def _plot_metrics_grid(
    save_path: Path,
    metric_table: pd.DataFrame,
    metric_plot_data: Dict[str, Dict[str, any]],
    metrics_to_include: Tuple[MetricSetting],
    reference_dataset: str,
    comparand_dataset: str,
    reference_model: str,
    comparand_model: str,
    confidence_level: float,
    column_type: str,
    row_type: str,
    hue_type: str,
    save_suffix: str,
    save_image_extension: str,
    plot_settings: PlotSettings,
    zoomed_quantiles=None,  # None when on zoomed
) -> None:
    metrics = metric_table.copy()

    col_names = pd.unique(metrics[column_type])
    col_names = np.array(
        [metric_type for metric_type in col_names if metric_type in metrics_to_include]
    )

    nbr_columns = col_names.shape[0]
    row_mult = math.ceil(nbr_columns / plot_settings.max_nbr_columns)
    row_names = (
        pd.unique(metrics[row_type]) if row_type is not None else pd.Series(["all"])
    )
    nbr_rows = row_names.shape[0]
    plot_rows = nbr_rows * row_mult

    plot_columns = min(plot_settings.max_nbr_columns, nbr_columns)
    fig = plt.figure(
        figsize=[
            plot_columns * plot_settings.width_height_subplot,
            plot_settings.width_height_subplot * nbr_rows * row_mult,
        ]
    )

    top = (
        1 - (0.3 / plot_rows) if plot_rows > 2 else 1 - (0.2 / plot_rows)
    )  # where gridspace starts
    gs = GridSpec(plot_rows, plot_columns, hspace=0.3, wspace=0.3, top=top)

    show_legend = False

    for col, col_name in enumerate(col_names):  # iterate over cols
        plot_data_col = metrics.loc[metrics[column_type] == col_name]

        for row, row_name in enumerate(row_names):
            plot_data = (
                plot_data_col.loc[metrics[row_type] == row_name]
                if row_type is not None
                else plot_data_col
            )

            hue = get_hue(hue_type, plot_data)

            if col == col_names.shape[0] - 1 and row == 0:
                show_legend = True
            else:
                show_legend = False
            ax = fig.add_subplot(
                gs[
                    (row * row_mult + col // plot_settings.max_nbr_columns),
                    col % plot_settings.max_nbr_columns,
                ]
            )
            seaborn_plot = sns.histplot(
                data=plot_data,
                x="metric_value",
                hue=hue,
                element="step",
                stat="probability",
                common_norm=False,
                fill=True,
                legend=show_legend,
            )

            metric_type = col_name
            # set row label at the start of each row
            ax.set_title(col_name)
            ax.set(xlabel=None)

            if col == 0:
                dataset = metric_table["dataset"].unique().tolist()
                ax.set(ylabel=f"T-test: {dataset}")
            else:
                ax.set(ylabel=None)

            if metric_type in metric_plot_data.keys():
                add_ttest_vlines(
                    metric_plot_data=metric_plot_data,
                    metric_type=metric_type,
                    ax=ax,
                )
                ax.legend(
                    loc="upper right", bbox_to_anchor=(1.02, 0.92), prop={"size": 7}
                )
                reference, comparand = get_reference_comparand(
                    reference_dataset=reference_dataset,
                    comparand_dataset=comparand_dataset,
                    reference_model=reference_model,
                    comparand_model=comparand_model,
                )
                plot_ttest_stats(
                    ax=ax,
                    metric_plot_data=metric_plot_data,
                    metric_type=metric_type,
                    confidence_level=confidence_level,
                    reference=reference,
                    comparand=comparand,
                )

            # this will zoom in on the values inside the xlim_quantiles
            if zoomed_quantiles is not None:
                xlims = list(plot_data["metric_value"].quantile(zoomed_quantiles))
                xlims.sort()
                xlims = [
                    xlims[0]
                    - 0.0005,  # the left xlim point is not included, need to expand range
                    xlims[-1] + 0.05,
                ]
                ax.set_xlim(xlims)
            # this will adjust the limits to match the range of the metric
            elif metric_plot_data[metric_type]["value_range"] is not None:
                ax.set_xlim(metric_plot_data[metric_type]["value_range"])

    fig.suptitle(plot_settings.plot_title)

    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path / f"histograms_{save_suffix}{save_image_extension}",
        bbox_inches="tight",
        dpi=plot_settings.figure_dpi,
    )


def get_plot_suffix(
    reference_dataset: str,
    comparand_dataset: str,
    reference_model: str,
    comparand_model: str,
) -> str:
    if comparand_dataset == reference_dataset:
        return f"model_comparison_{reference_model}_{comparand_model}_{reference_dataset}_allinone"

    elif comparand_model == reference_model:
        return f"dataset_comparison_{reference_dataset}_{comparand_dataset}_{reference_model}_allinone"

    return "comparison_invalid"


def get_reference_comparand(
    reference_dataset: str,
    comparand_dataset: str,
    reference_model: str,
    comparand_model: str,
) -> tuple:
    """
    Returns the reference and comparand based on whether the datasets or models are being compared.

    If the datasets are the same, returns the models as (reference, comparand).
    If the models are the same, returns the datasets as (reference, comparand).

    Parameters:
    reference_dataset (str): Name of the reference dataset.
    comparand_dataset (str): Name of the comparand dataset.
    reference_model (str): Name of the reference model.
    comparand_model (str): Name of the comparand model.

    Returns:
    tuple: A tuple containing (reference, comparand) based on the comparison type.
    """
    if comparand_dataset == reference_dataset:
        return reference_model, comparand_model

    return reference_dataset, comparand_dataset
