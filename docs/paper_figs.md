# Paper figures — source files

Mapping of each paper figure panel to the result file it is built from and the notebook/script that produces it. Kept in sync with paper_figs.xlsx.

Pure-schematic panels (no repo-generated file) are listed with an empty results file and marked "schematic" in the notes.

| paper figure | results file | notebook | notes |
| --- | --- | --- | --- |
| Fig 1c left | results/stim_05-08-23_example_images.pdf | plot_figures.ipynb |  |
| Fig 1c right | results/mouse_summary/TX109_05_30_2023_reliable_0.1thre_positions.pdf | plot_figures.ipynb | mouse 2 |
| Fig 1d | results/svm_0.001C_250avg.pdf | plot_figures.ipynb | full space |
| Fig 1d (stats) | results/svm_significance.csv | plot_figures.ipynb | p-values |
| Fig 1e | results/ori_0ref_250avg.pdf | plot_figures.ipynb | full space |
| Fig 1e (stats) | results/ori_significance_0ref.csv | plot_figures.ipynb | p-values |
| Fig 2 |  |  | all schematics; not repo-generated |
| Fig 3a |  |  | schematic: method; not repo-generated |
| Fig 3b | results/dim_250avg_20dim.pdf | plot_figures.ipynb | dim counts; dashed lines = the 'both' shuffle null of results_extended/null_bands_250avg_20dim.csv (plot_extended_figures.ipynb, plotting_validation.null_band_table) |
| Fig 3c top | results/svm_1C_250avg_20dim.pdf | plot_figures.ipynb | invariant subspace |
| Fig 3c top (stats) | results/svm_significance_20dim.csv | plot_figures.ipynb | p-values |
| Fig 3c top (stats) | results_extended/class2x_effect_sizes_20dim.csv | plot_figures.ipynb | effect sizes |
| Fig 3c bottom | results/ori_0ref_250avg_20dim.pdf | plot_figures.ipynb | tuned subspace |
| Fig 3c bottom (stats) | results/ori_significance_0ref_20dim.csv | plot_figures.ipynb | p-values |
| Fig 3c bottom (stats) | results_extended/ori2x_effect_sizes_20dim.csv | plot_figures.ipynb | effect sizes |
| Fig 4a (odd columns) | results/example_projs/{model_name}_{layer_name}_example_projs_data_250rand_20dim_{invariant\|tuned}.pdf | plot_figures.ipynb | mouse 2 V1, mouse 2 LM, mouse 2 AL, AlexNet conv1, AlexNet conv3, AlexNet conv5 |
| Fig 4a (even columns) | results/example_projs/{model_name}_{layer_name}_example_projs_model_250rand_20dim_{invariant\|tuned}.pdf | plot_figures.ipynb | same panels |
| Fig 4b | results/pcorr_250avg_20dim.pdf | plot_figures.ipynb |  |
| Fig 4b orange (stats) | results/pcorr_inv_significance.csv | plot_figures.ipynb | p-values |
| Fig 4b brown (stats) | results/pcorr_tuned_significance.csv | plot_figures.ipynb | p-values |

| ED Fig 1a | results/mouse_summary/{model_name}_reliable_0.1thre_positions.pdf | plot_figures.ipynb | retinotopy maps; one per mouse |
| ED Fig 1b | results/mouse_summary/{model_name}_reliable_0.1thre_CChalf.pdf | plot_figures.ipynb | one per mouse |
| ED Fig 1b | results/mouse_summary/{model_name}_reliable_0.1thre_dp.pdf | plot_figures.ipynb | one per mouse |
| ED Fig 1b | results/mouse_summary/{model_name}_reliable_0.1thre_OSI.pdf | plot_figures.ipynb | one per mouse |
| ED Fig 2a top | results_extended/svm_stim_0.001C_250avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2a bottom | results_extended/ori_stim_0ref_250avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2b top | results_extended/svm1_relori_.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2b bottom | results_extended/svm2_relori_.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2c top | results_extended/ori1_relori.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2c bottom | results_extended/ori2_relori.pdf | plot_extended_figures.ipynb |  |
| ED Fig 3a top | results/stim_05-08-23_all_classes.pdf | plot_figures.ipynb |  |
| ED Fig 3a bottom | results_extended/distances_avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 3b | results_extended/relcenter_avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 4a top | results_extended/svm1_0.001C_Navg.pdf | plot_extended_figures.ipynb | class decoding vs N |
| ED Fig 4a bottom | results_extended/svm2_0.001C_Navg.pdf | plot_extended_figures.ipynb | class decoding vs N |
| ED Fig 4b top | results_extended/ori1_Navg.pdf | plot_extended_figures.ipynb | rel-orientation decoding vs N |
| ED Fig 4b bottom | results_extended/ori2_Navg.pdf | plot_extended_figures.ipynb | rel-orientation decoding vs N |
| ED Fig 5 | results_extended/example_weights/TX109_05_30_2023_{layer}_example_decoder_250rand_comp_{i}{j}.pdf | plot_extended_figures.ipynb (plotting_functions.plot_example_decoder_weights) | mouse 2 (TX109); the files of every row of model_params.csv are written |
| ED Fig 6a-c |  |  | all schematics; not repo-generated |
| ED Fig 6d top | results_extended_anns/rotation_robustness.pdf | more_anns_analyses.ipynb |  |
| ED Fig 6d bottom | results_extended_anns/orientation_robustness.pdf | more_anns_analyses.ipynb |  |
| ED Fig 6e left | results_extended_anns/more_anns_inv_dim_by_model.pdf | more_anns_analyses.ipynb | invariant dims |
| ED Fig 6e right | results_extended_anns/more_anns_tuned_dim_by_model.pdf | more_anns_analyses.ipynb | tuned dims |
| ED Fig 6f left | results_extended_anns/more_anns_decoding_error_by_model.pdf | more_anns_analyses.ipynb | class decoding error |
| ED Fig 6f right | results_extended_anns/more_anns_orient_error_by_model.pdf | more_anns_analyses.ipynb | relative-orientation error |
| ED Fig 6g left | results_extended_anns/more_anns_inv_pcorr_by_model.pdf | more_anns_analyses.ipynb | |
| ED Fig 6g right | results_extended_anns/more_anns_tuned_pcorr_by_model.pdf | more_anns_analyses.ipynb | |
| ED Fig 7a-b |  |  | all schematics; not repo-generated |
| ED Fig 8a |  |  | schematic: simulation; not repo-generated |
| ED Fig 8b row 1 | results_extended/simulation/simulation_{fitdim,N,snr,xi,eqvfrac}_dims.pdf | src/simulations.ipynb (plot_simulation.plot_axis) | estimated d_inv / d_tun; columns = fit dim d̂, N, σ² (0/0.5/1/2/3.9), angular irregularity ξ (0 / 0.1 / 0.9 = cells reference / control_C2p_a100 / control_C2p_a1), equivariant fraction f; 10 seeds per cell |
| ED Fig 8b row 2 | results_extended/simulation/simulation_{fitdim,N,snr,xi,eqvfrac}_svm_err.pdf | src/simulations.ipynb (plot_simulation.plot_axis) | image-identity SVM error (lowest held-out error over C = 0.001…100, per space), full space + estimated invariant subspace overlaid |
| ED Fig 8b row 3 | results_extended/simulation/simulation_{fitdim,N,snr,xi,eqvfrac}_ori_err.pdf | src/simulations.ipynb (plot_simulation.plot_axis) | relative-orientation error, full space + estimated tuned subspace overlaid |
| ED Fig 8b row 4 | results_extended/simulation/simulation_{fitdim,N,snr,xi,eqvfrac}_cc_norm_equivariance_c.pdf | src/simulations.ipynb (plot_simulation.plot_axis) | centered median CC_norm of the equivariance model |
| ED Fig 8c row 1 | results_extended/simulation/simulation_{dims,epsilon,delta,snr_hetero,vtun,images,trials}_dims.pdf | src/simulations.ipynb (plot_simulation.plot_axis) | same 4 rows as 8b; columns = true dims, epsilon, delta, gain heterogeneity (snr_hetero), tuned variance (var_tun), n images, n trials |
| ED Fig 8c row 2 | results_extended/simulation/simulation_{dims,epsilon,delta,snr_hetero,vtun,images,trials}_svm_err.pdf | src/simulations.ipynb (plot_simulation.plot_axis) |  |
| ED Fig 8c row 3 | results_extended/simulation/simulation_{dims,epsilon,delta,snr_hetero,vtun,images,trials}_ori_err.pdf | src/simulations.ipynb (plot_simulation.plot_axis) |  |
| ED Fig 8c row 4 | results_extended/simulation/simulation_{dims,epsilon,delta,snr_hetero,vtun,images,trials}_cc_norm_equivariance_c.pdf | src/simulations.ipynb (plot_simulation.plot_axis) |  |
| ED Fig 8b (stats) | results_extended/simulation/simulation_effect_sizes.csv | src/simulations.ipynb (simulation.effect_size_table) | paired decoder effect sizes (mean(δ)/sd(δ), ROC AUC) of the N and σ² cells quoted in the Methods; cache/simulation_effect, 10 seeds |
| ED Fig 9a left | results_extended/ccnorm_pooled_vs_dim1_centered_250avg.pdf | plot_extended_figures.ipynb (plotting_validation.plot_ccnorm_pooled_vs_dim) | pooled centered CC_norm of the equivariance model vs the fitted d = 6 / 20 / 50 / 100 / 200 at N = 250 (pooled_ccnorm_vs_dim_all), pixels + mouse areas |
| ED Fig 9a right | results_extended/ccnorm_pooled_vs_dim2_centered_250avg.pdf | plot_extended_figures.ipynb (plotting_validation.plot_ccnorm_pooled_vs_dim) | AlexNet conv1-5 |
| ED Fig 9b | results_extended/cc_pooled_c1c2_centered_250avg_20dim.pdf | plot_extended_figures.ipynb (plotting_validation.plot_ccnorm_pooled_c1c2) | pooled centered CC_norm, equivariance model (left dot of each pair) vs the fitted per-rotation control model (right); pooled_ccnorm_all with 10 shuffles per kind. NB on the pixels the equivariance (R^k) point sits slightly ABOVE the per-rotation point (0.276 vs 0.264) although the latter is more flexible: pixels are exactly equivariant, so the 14 free maps only add estimation variance out of sample |
| ED Fig 9b (stats) | results_extended/ccnorm_pooled_250avg_20dim_table.csv | plot_extended_figures.ipynb (plotting_validation.pooled_ccnorm_table) | pooled CC_norm / FEVE table (covers ED Fig 9a-b) |
| ED Fig 9c | results_extended/calibration/ccnorm_vs_d_pixels.pdf | src/simulations.ipynb (plot_simulation.plot_pixels_vs_d) | pixels, median CC_norm vs fitted d at N = 250/1000/4000/8000 (light to dark); injected noise solid, noiseless dashed |
| ED Fig 9d left | results_extended/calibration/ccnorm_peak_vs_N_pixels.pdf | src/simulations.ipynb (plot_simulation.plot_pixels_peak_vs_n) | pixels, peak over d of the median CC_norm vs N; noisy solid, noiseless dashed |
| ED Fig 9d right | results_extended/calibration/ccnorm_peak_vs_N_pixel_simulations.pdf | src/simulations.ipynb (plot_simulation.plot_pixel_simulations_peak_vs_n) | simulated populations built from the pixels' equivariant structure (simulation.band_structure + run_matched_populations; cache/calib_pixels/rtruth_c0, rtruth_r500_c{0,1}, rtruth_r20_c{0,1,2}) at equivariant fraction f = 1 (dashed) / 0.5 (solid) / 0.1 (dotted) and equivariant dimension R = 1362 (black) / 500 (mid grey) / 20 (light grey); same N, noise (direct calibration, inv_snr 3.98) and pipeline as the pixels. The published panel was made from populations with the earlier per-stimulus noise calibration (inv_snr 3.83 / 3.71, now cache/calib_pixels/legacy_rtruth_c*); a regeneration with the public code lowers the f = 1 lines slightly at small N (0.22 instead of 0.27 at N = 250) |
| ED Fig 9e left | results_extended/calibration/ccnorm_vs_d_conv1.pdf | src/simulations.ipynb (plot_simulation.plot_conv1_vs_d) | AlexNet conv1 at N = 250/1000/5040 (its reliably responsive units), noisy solid, the same units noiseless dashed |
| ED Fig 9e right | results_extended/calibration/ccnorm_vs_d_conv_layers.pdf | src/simulations.ipynb (plot_simulation.plot_conv_layers_vs_d) | conv1–5 (light to dark) at their reliably responsive counts (5040/2374/927/539/303), noisy solid, the same units noiseless dashed (cache suffix _noiseless_rel, simulation.build_noiseless_rel_caches) |
| ED Fig 9f | results_extended/calibration/ccnorm_peak_vs_N_conv_layers.pdf | src/simulations.ipynb (plot_simulation.plot_conv_layers_peak_vs_n) | conv1–5 peak over d of the median CC_norm vs N up to each layer's reliable count; noisy solid with markers, noiseless (same units) dashed without |
| ED Fig 9g left | results_extended/calibration/ccnorm_vs_d_V1.pdf | src/simulations.ipynb (plot_simulation.plot_v1_vs_d) | mouse V1 pooled across the 3 mice (cache/mouse_pooled, simulation.build_pooled_caches), N = 250/1000/4000/8000 |
| ED Fig 9g right | results_extended/calibration/ccnorm_vs_d_areas.pdf | src/simulations.ipynb (plot_simulation.plot_areas_vs_d) | V1/LM/RL/AL pooled across the 3 mice, each at its largest N (8000/3000/1600/855) |
| ED Fig 9h | results_extended/calibration/ccnorm_peak_vs_N_areas.pdf | src/simulations.ipynb (plot_simulation.plot_areas_peak_vs_n) | V1/LM/RL/AL pooled across the 3 mice (10,698 / 3,100 / 1,696 / 855 reliable neurons), peak over d of the median CC_norm vs N |
| ED Fig 10a | results_extended/data_fit/datafit_{delta_over_r,snr_hetero,sigma_calib,f_calib}.pdf | src/simulations.ipynb (plot_simulation.plot_datafit_params) | per population: δ/r and the per-neuron SNR spread measured per neuron subsample (simulation.datafit_params → data_fit_params.csv; pixels and conv layers drawn at 0, uniform SNR by construction), the calibrated noise σ = √inv_snr per seed of the data-fit simulations, and f = 1 − iid from the (N, d) sweep (simulation.best_iid / fill_swept → params/datafit_swept_iid.csv: pixels 1, V1 0.5, LM 0.7, RL 0.5, AL 0.6, conv1–3 0.8, conv4 0.7, conv5 0.4) |
| ED Fig 10b | results_extended/data_fit/datafit_sim_{dim,svm1C,ori,cc}.pdf | src/simulations.ipynb (plot_simulation.plot_datafit_simulations) | the structure-matched populations at their swept f through the standard pipeline (N = 250, d = 20, 2 trials; simulation.run_target_cached → cache/simulation_datafit_sized, 10 seeds; datafit_table → data_fit_sim.csv) with the data overlaid; rows = invariant / tuned dimensions, image-identity error in the invariant subspace (C = 1), orientation error in the tuned subspace, median CC_norm |

| SI Fig 3a top | results_extended/ori1_refs.pdf | plot_extended_figures.ipynb |  |
| SI Fig 3a bottom | results_extended/ori2_refs.pdf | plot_extended_figures.ipynb |  |
| SI Fig 3b top | results_extended/ori1_refs_20dim.pdf | plot_extended_figures.ipynb |  |
| SI Fig 3b bottom | results_extended/ori2_refs_20dim.pdf | plot_extended_figures.ipynb |  |
| SI Fig 4a | results_extended/svm_Crange_250avg.pdf | plot_extended_figures.ipynb | class decoding vs C, full space (C=0.001 used in Fig 1d). Panel-letter order inferred — confirm against the assembled figure |
| SI Fig 4b | results_extended/svm_Crange_250avg_20dim.pdf | plot_extended_figures.ipynb | class decoding vs C, invariant subspace (C=1 used in Fig 3c) |
| SI Fig 5a | results_extended_gain/panel_a_noise_correlations.pdf | gain_model_analyses.ipynb | mouse 1 |
| SI Fig 5b left | results_extended_gain/panel_b_mean_vs_std_raw.pdf | gain_model_analyses.ipynb | mouse 1 |
| SI Fig 5b right | results_extended_gain/panel_b_mean_vs_std_normalized.pdf | gain_model_analyses.ipynb | mouse 1 |
| SI Fig 5c | results_extended_gain/panel_c_svd_explained_variance.pdf | gain_model_analyses.ipynb | mouse 1 |
| SI Fig 5d left | results_extended_gain/svm_0.001C_250avg_mult_gain.pdf | gain_model_analyses.ipynb |  |
| SI Fig 5d right | results_extended_gain/ori_0ref_250avg_mult_gain.pdf | gain_model_analyses.ipynb |  |
| SI Fig 5e left | results_extended_gain/dim_250avg_20dim_mult_gain.pdf | gain_model_analyses.ipynb |  |
| SI Fig 5e right | results_extended_gain/pcorr_250avg_20dim_mult_gain.pdf | gain_model_analyses.ipynb |  |
| SI Fig 5f | results_extended_gain/cc_pooled_equiv_centered_250avg_20dim_mult_gain.pdf | gain_model_analyses.ipynb (plotting_validation.plot_ccnorm_pooled_equiv) | pooled within-image-centered CC_norm of the equivariance model under the mult_gain noise model (pooled_ccnorm_all with the gain params, no shuffles) |
| SI Fig 6a |  |  | schematic: the three architecture variants (C1 / C15 / C15>C5>C1), built-in equivariance per layer; not repo-generated. NB panel-letter assignment (a = schematic, b = robustness) — confirm against the assembled figure |
| SI Fig 6b | results_extended_anns/eqv_rotation_robustness.pdf | more_anns_analyses.ipynb | eqv ANNs (c15 pink, c15r red, c1 black): classification error vs rotation angle |
| SI Fig 6c left | results_extended_anns/eqv_anns_inv_dim_by_model.pdf | more_anns_analyses.ipynb | |
| SI Fig 6c right | results_extended_anns/eqv_anns_tuned_dim_by_model.pdf | more_anns_analyses.ipynb | |
| SI Fig 6d left | results_extended_anns/eqv_anns_decoding_error_by_model.pdf | more_anns_analyses.ipynb | |
| SI Fig 6d right | results_extended_anns/eqv_anns_orient_error_by_model.pdf | more_anns_analyses.ipynb | |
| SI Fig 6e left | results_extended_anns/eqv_anns_inv_pcorr_by_model.pdf | more_anns_analyses.ipynb | |
| SI Fig 6e right | results_extended_anns/eqv_anns_tuned_pcorr_by_model.pdf | more_anns_analyses.ipynb | |
| SI Fig 6f left | results_extended_anns/ccnorm_peak_vs_N_eqv-c1.pdf | more_anns_analyses.ipynb (plot_simulation.plot_eqv_net_peak_vs_n) | C1: peak over d of the median CC_norm vs N (ED Fig 9f construction and axes), conv1–5 in the SI Fig 6 colour shaded light (conv1) to dark (conv5); noisy solid with markers, noiseless dashed for conv1 and conv5; N up to each layer's reliably responsive count (simulation.ccnorm_surface) |
| SI Fig 6f middle | results_extended_anns/ccnorm_peak_vs_N_eqv-c15.pdf | more_anns_analyses.ipynb (plot_simulation.plot_eqv_net_peak_vs_n) | C15 exact: same layout |
| SI Fig 6f right | results_extended_anns/ccnorm_peak_vs_N_eqv-c15r.pdf | more_anns_analyses.ipynb (plot_simulation.plot_eqv_net_peak_vs_n) | C15>C5>C1: same layout |
| SI Fig 7a | results_extended/simulation/simulation_calibration_ccnorm_vs_corr.pdf | src/simulations.ipynb (plot_simulation.plot_calibration) | centered CC_norm vs the true correlation between prediction and noiseless signal: the benchmark cells (light grey, mean over seeds) with the 10 structure-matched populations of ED Fig 10b (dark); dashed identity |
| SI Fig 7b | results_extended/simulation/simulation_calibration_ccnorm2_vs_r2.pdf | src/simulations.ipynb (plot_simulation.plot_calibration) | centered CC_norm² vs the true as-is R² (1 − MSE / var(signal)), same layout |
| SI Fig 7c | results_extended/simulation/simulation_calibration_spe_vs_r2.pdf | src/simulations.ipynb (plot_simulation.plot_calibration) | centered SPE (signal power explained, the 'feve' of helper.pooled_scores) vs the true as-is R², same layout |
| SI Fig 8a top | results/svm_0.001C_4000avg_overlay250.pdf | plot_extended_figures.ipynb (plotting_functions.plot_classification_errors, overlay_params_df = the 250-neuron params, ymin=0.5) | N = 4000 opaque diamonds over N = 250 semi-transparent, area means only |
| SI Fig 8a bottom | results/ori_0ref_4000avg_overlay250.pdf | plot_extended_figures.ipynb (plotting_functions.plot_orientation_errors, overlay_params_df, ymin=0) | same overlay convention |
| SI Fig 8b top | results/dim_4000avg_20dim_overlay250.pdf | plot_extended_figures.ipynb (plotting_functions.plot_equivariance_dim, overlay_params_df) | same overlay convention; per-subsample counting as Fig 3b |
| SI Fig 8b bottom | results/pcorr_4000avg_20dim_overlay250.pdf | plot_extended_figures.ipynb (plotting_functions.plot_equivariance_pcorr, overlay_params_df, ymax=1) | same overlay convention; variance-weighted degrees as Fig 4b |
| SI Fig 8c | results_extended/cc_pooled_equiv_overlay_4000_250avg_centered_20dim.pdf | plot_extended_figures.ipynb (plotting_validation.plot_ccnorm_pooled_equiv_overlay, n_list=(4000, 250)) | pooled within-image-centered CC_norm of the equivariance model, N = 4000 opaque over N = 250 at alpha 0.4, no whiskers, area means only; needs the pooled caches at N = 4000 (pooled_ccnorm_all with the *_all params, no shuffles) |
| SI Fig 9a top | results_extended/cc_pooled_equiv_centered_250avg_20dim_scenes.pdf | plot_extended_figures.ipynb (plotting_validation.plot_ccnorm_pooled_equiv, subset='scenes') | pooled centered CC_norm, scenes only (pooled_ccnorm_subset_all) |
| SI Fig 9a bottom | results/dim_250avg_20dim_scenes.pdf | plot_extended_figures.ipynb (plotting_functions.plot_equivariance_dim, subset='scenes') | mean dimension, scenes only (equivariance_summary_all(subset=)) |
| SI Fig 9b top | results_extended/cc_pooled_equiv_centered_250avg_20dim_textures.pdf | plot_extended_figures.ipynb (plotting_validation.plot_ccnorm_pooled_equiv, subset='textures') | pooled centered CC_norm, textures only |
| SI Fig 9b bottom | results/dim_250avg_20dim_textures.pdf | plot_extended_figures.ipynb (plotting_functions.plot_equivariance_dim, subset='textures') | mean dimension, textures only |
| SI Fig 10a | results_extended/angular_steps_TX108.pdf | plot_extended_figures.ipynb (angular_steps.run_all → summarize → plot_si_panels) | mouse 1: the angular step of each of the 15 rotation steps (the 15th closes the loop) read out from the fitted tuned planes (circular mean over images per plane, planes pooled within a 250-neuron subsample, mean ± sem across subsamples; cache/angular_steps), pixels + V1/LM/RL/AL in the ED Fig 2 grey ramp, dotted line = regular 24°, x = rotation angle reached; the step CV ξ per population (quoted in the SI) in results_extended/angular_steps_summary.csv |
| SI Fig 10b | results_extended/angular_steps_TX109.pdf | plot_extended_figures.ipynb (angular_steps.plot_si_panels) | mouse 2, same layout |
| SI Fig 10c | results_extended/angular_steps_TX111.pdf | plot_extended_figures.ipynb (angular_steps.plot_si_panels) | mouse 3, same layout |
| SI Fig 10d | results_extended/angular_steps_alexnet.pdf | plot_extended_figures.ipynb (angular_steps.plot_si_panels) | AlexNet conv1–5 (conv1 lightest to conv5 black), same layout, no pixels |
