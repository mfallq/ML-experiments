import logging
from dataclasses import dataclass, field
from typing import List

from a import EvalType
from z import EvalSetup
from y import IEvalStep
from x import (
    calculate_delta_metrics,
    convert_metrics_to_dict,
    get_ttest_metric_plot_data,
    plot_ttest_results,
)
from b import write_metrics_to_csv
from c import PlotSettings


@dataclass(frozen=True)
class CompareModelImagesPairwise(IEvalStep):
    """
    CompareModelImagesPairwise is a class used for pairwise comparison of a dataset matching images from different models, based on various metric types.
    The output is a set of histograms with t-test statistics, saved for further analysis.

    In cases where test results indicate a significant difference between models, additional analysis is needed to determine the cause of the variation.

    The produce_result adds a fake model to the eval_setup in the end in order to generate and save plots to the correct folder.

    Input:
    - `eval_type` (EvalType): The type of evaluation being conducted

    Output:
    - saved histograms of resulting t-tests
    """

    eval_type: EvalType = EvalType.B
    zoomed_quantiles: List[float] = field(default_factory=lambda: [0.005, 0.995])

    plot_settings = PlotSettings(
        plot_title=(
            "Evaluation of model differences in paired T-test with empirical thresholds for acceptable deviations"
        ),
        max_nbr_columns=4,
        width_height_subplot=3,
        figure_dpi=300,
    )

    # metrics params
    column_type: str = "metric_type"
    ttest_alternative: str = "less"
    confidence_level: float = 0.95

    # saving params
    save_folder: str = "histograms_compare_imgs_pairwise"
    save_image_extension: str = ".png"
    save_suffix_zoomed: str = "_zoomed"

    def produce_results(self, eval_setup: EvalSetup) -> None:
        logging.info("Running compare_model_image_pairwise.py")
        if len(eval_setup.models) < 1:
            logging.info(
                "Can't compare models per dataset if there is only one model in the eval. Exiting step!"
            )
            return
        reference_model = eval_setup.get_reference_model()
        if not reference_model:
            logging.warning(
                "Can't run the comparison between models if there is no reference model, please select one"
            )
            return

        metric_table = eval_setup.load_metrics_data()

        metric_plot_data, metrics_to_include = get_ttest_metric_plot_data(
            eval_setup=eval_setup, eval_type=self.eval_type
        )

        for comparand_model in eval_setup.models:
            if comparand_model.name == reference_model.name:
                continue  # Skip when comparing the reference model with itself
            reference_datasets = eval_setup.get_dataset_names_for_model(
                model_name=reference_model.name
            )
            comparand_datasets = eval_setup.get_dataset_names_for_model(
                model_name=comparand_model.name
            )
            common_datasets = list(
                set(reference_datasets).intersection(comparand_datasets)
            )
            if not common_datasets:
                logging.warning(
                    f"There are no matching datasets between the reference model {reference_model.name} and the comparand model {comparand_model.name}."
                )
                continue
            for dataset in common_datasets:
                logging.info(
                    f"Calculating T-test statistics for dataset {dataset} and reference model {reference_model.name} and the comparand model {comparand_model.name}"
                )
                metrics, metric_delta_plot_data = calculate_delta_metrics(
                    metric_table=metric_table,
                    metric_plot_data=metric_plot_data,
                    reference_dataset=dataset,
                    comparand_dataset=dataset,
                    reference_model=reference_model.name,
                    comparand_model=comparand_model.name,
                    metrics_to_include=metrics_to_include,
                    confidence_level=self.confidence_level,
                    column_type=self.column_type,
                    ttest_alternative=self.ttest_alternative,
                )
                if len(metrics) > 0:
                    logging.info(f"Plotting T-test results")
                    plot_ttest_results(
                        save_path=eval_setup.save_path / self.save_folder,
                        metric_table=metrics,
                        metric_plot_data=metric_delta_plot_data,
                        metrics_to_include=metrics_to_include,
                        reference_dataset=dataset,
                        comparand_dataset=dataset,
                        reference_model=reference_model.name,
                        comparand_model=comparand_model.name,
                        zoomed_quantiles=self.zoomed_quantiles,
                        confidence_level=self.confidence_level,
                        column_type=self.column_type,
                        save_suffix_zoomed=self.save_suffix_zoomed,
                        save_image_extension=self.save_image_extension,
                        plot_settings=self.plot_settings,
                    )
                    logging.info(f"Writing T-test csv results")
                    all_metrics_results = convert_metrics_to_dict(metrics)
                    folder_name = eval_setup.get_folder_name(
                        model_name=f"{reference_model.name}_{comparand_model.name}",
                        dataset_name=dataset,
                    )
                    metrics_folder = eval_setup.get_results_folder() / folder_name
                    write_metrics_to_csv(
                        metrics_values_csv_name=eval_setup.metrics_values_csv_name,
                        save_path=metrics_folder,
                        all_metric_results=all_metrics_results,
                    )
