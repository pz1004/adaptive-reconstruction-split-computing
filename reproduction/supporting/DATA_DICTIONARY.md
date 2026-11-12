# Data dictionary

Interface identifiers are preserved: early = higher-resolution, six channels at 16 x 16; current = lower-resolution, six channels at 8 x 8. Model seeds are 7, 42, 123, 2024, 2025. They index paired pipeline runs. Draw seeds 1701--1705 index repeated noise realizations within a run and are not independent replicates.

Task accuracy is in percent; differences are percentage points. SSIM, AUROC and balanced accuracy use their native dimensionless scales. Higher AUROC/balanced accuracy means more observed attribute predictability. Lower SSIM means less observed reconstruction fidelity.

Compute *_latency_seconds and summed_component_time_seconds are whole-batch seconds per call. Divide by batch_size for amortized per-item compute time. *_throughput_per_second is items/second (batch size divided by whole-batch time). *_bytes are bytes. Condition values are medians of accepted technical-block medians; attempt_count is the accepted attempt count. Empty defense-only fields mean not applicable for standard/AFD. RSS is nonnegative end-minus-start process RSS including initialization, not peak or isolated model RSS. Measurements exclude transfer, network, queueing, serialization, data loading and energy.

run_id and condition_id identify experimental aggregate conditions, not people or examples. source_sha256 commits to a retained private prediction bundle. selected_defense_value is the selected coefficient/scale/radius for the named method. record_passed is the historical validation result. origin identifies prospective versus inherited conditions. Auxiliary fractions are nested training-pool fractions; auxiliary_examples is the total prefix count including internal validation. model_seed and attacker_training_seed are coupled in the budget study. Fixed-victim restarts use model_seed=42 and attacker_training_seed in [42, 314159, 271828]. MSE is native pixel squared error, PSNR is dB, LPIPS is native perceptual distance.

The original semantic endpoint averages probabilities over five Laplace draws before AUROC and threshold-0.5 balanced accuracy. The single-release endpoint averages per-draw macro metrics within a model run. Macro values average all 39 non-Smiling attributes. Ties at 0.5 are classified positive. Attribute SD columns describe variation across attributes, not uncertainty over runs. utility_loss_vs_standard_percentage_points = standard minus candidate accuracy; test_utility_loss_exceeds_validation_limit indicates a loss greater than one percentage point. strongest_attacker selects the maximum observed SSIM; validation_selected_attacker records the validation choice.

## primary_seed_results.csv
Rows: 80. Fields: `run_id`, `dataset`, `split_point`, `seed`, `method`, `selected_defense_value`, `accuracy_percentage_points`, `utility_loss_vs_standard_percentage_points`, `test_utility_loss_exceeds_validation_limit`, `worst_case_ssim`, `strongest_attacker`, `validation_selected_attacker`, `record_passed`.

## semantic_seed_results.csv
Rows: 40. Fields: `run_id`, `dataset`, `split_point`, `seed`, `method`, `selected_defense_value`, `macro_auroc`, `macro_balanced_accuracy`, `attribute_auroc_sample_sd`, `attribute_balanced_accuracy_sample_sd`, `semantic_draw_count`, `semantic_draw_seeds`, `record_passed`.

## benchmark_budget_conditions.csv
Rows: 120. Fields: `run_id`, `dataset`, `interface`, `method`, `model_seed`, `attacker_training_seed`, `auxiliary_fraction`, `auxiliary_examples`, `strongest_attacker`, `worst_case_ssim`, `origin`.

## attacker_restart_rows.csv
Rows: 96. Fields: `run_id`, `dataset`, `interface`, `method`, `architecture`, `model_seed`, `attacker_training_seed`, `mse`, `psnr`, `ssim`, `lpips`, `origin`.

## compute_condition_metrics.csv
Rows: 160. Fields: `condition_id`, `dataset`, `interface`, `method`, `model_seed`, `batch_size`, `cpu_encoder_plus_defense_latency_seconds`, `cpu_encoder_plus_defense_throughput_per_second`, `defense_only_latency_seconds`, `defense_only_throughput_per_second`, `gpu_classifier_latency_seconds`, `gpu_classifier_throughput_per_second`, `summed_component_time_seconds`, `isolated_cpu_rss_delta_bytes`, `cuda_peak_allocation_bytes`, `static_edge_encoder_bytes`, `static_cloud_classifier_bytes`, `static_defense_bytes`, `static_total_bytes`, `attempt_count`.

## laplace_draw_metrics.csv
Rows: 50. Fields: `dataset`, `interface`, `model_seed`, `draw_seed`, `macro_auroc`, `macro_balanced_accuracy`, `source_sha256`.

## laplace_run_metrics.csv
Rows: 10. Fields: `dataset`, `interface`, `model_seed`, `draw_count`, `attribute_count`, `single_release_macro_auroc`, `five_realization_macro_auroc`, `single_release_macro_balanced_accuracy`, `five_realization_macro_balanced_accuracy`, `source_sha256`.

Expected statistics: mean_difference is comparison minus reference; sample_sd is the sample SD across five paired differences; ci95_lower/upper are unadjusted Student-t interval bounds; exact_sign_flip_pvalue is two-sided over all 32 sign patterns; Hedges gz uses J=0.8; Shapiro-Wilk values are diagnostics only. Original Holm families are 24 and 12; the post hoc family has eight contrasts. Compute summary suffixes n/mean/std/ci95_low/ci95_high describe five model-seed conditions; compute differences use seconds. Full-precision source pointers in the retained original supplement refer to historical analysis objects, not required filesystem dependencies.
