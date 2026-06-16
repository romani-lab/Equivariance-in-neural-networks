# Paper figures — source files

Mapping of each paper figure panel to the result file it is built from and the notebook that produces it. Generated from paper_figs.xlsx.

| paper figure | results file | notebook | notes |
| --- | --- | --- | --- |
| Fig 1c left | results/stim_05-08-23_example_images.pdf | plot_figures.ipynb |  |
| Fig 1c right | results/mouse_summary/TX109_05_30_2023_reliable_0.1thre_positions.pdf | plot_figures.ipynb | mouse 2 |
| Fig 1d | results/svm_0.001C_250avg.pdf | plot_figures.ipynb | full space |
| Fig 1d (stats) | results/svm_significance.csv | plot_figures.ipynb | p-values |
| Fig 1e | results/ori_0ref_250avg.pdf | plot_figures.ipynb | full space |
| Fig 1e (stats) | results/ori_significance_0ref.csv | plot_figures.ipynb | p-values |
| Fig 3b top | results/example_evals/{model_name}_{layer_name}_example_evals_abs_250rand_20dim.pdf | plot_figures.ipynb | mouse 2 V1, mouse 2 AL, AlexNet conv1, AlexNet conv5 |
| Fig 3b bottom | results/dim_250avg_20dim.pdf | plot_figures.ipynb |  |
| Fig 3c top | results/svm_1C_250avg_20dim.pdf | plot_figures.ipynb | invariant subspace |
| Fig 3c top (stats) | results/svm_significance_20dim.csv | plot_figures.ipynb | p-values |
| Fig 3c top (stats) | results_extended/svm_effect_sizes_20dim.csv | plot_figures.ipynb | effect sizes |
| Fig 3c bottom | results/ori_0ref_250avg_20dim.pdf | plot_figures.ipynb | tuned subspace |
| Fig 3c bottom (stats) | results/ori_significance_0ref_20dim.csv | plot_figures.ipynb | p-values |
| Fig 3c bottom (stats) | results_extended/ori_effect_sizes_20dim.csv | plot_figures.ipynb | effect sizes |
| Fig 4a (odd columns) | results/example_projs/{model_name}_{layer_name}_example_projs_data_250rand_20dim_{invariant\|tuned}.pdf | plot_figures.ipynb | mouse 2 V1, mouse 2 LM, mouse 2 AL, AlexNet conv1, AlexNet conv3, AlexNet conv5 |
| Fig 4a (even columns) | results/example_projs/{model_name}_{layer_name}_example_projs_model_250rand_20dim_{invariant\|tuned}.pdf | plot_figures.ipynb | mouse 2 V1, mouse 2 LM, mouse 2 AL, AlexNet conv1, AlexNet conv3, AlexNet conv5 |
| Fig 4b | results/pcorr_250avg_20dim.pdf | plot_figures.ipynb |  |
| Fig 4b orange (stats) | results/pcorr_inv_significance.csv | plot_figures.ipynb | p-values |
| Fig 4b brown (stats) | results/pcorr_tuned_significance.csv | plot_figures.ipynb | p-values |
| ED Fig 1b top | results/mouse_summary/{model_name}_reliable_0.1thre_positions.pdf | plot_figures.ipynb | one per mouse |
| ED Fig 1b | results/mouse_summary/{model_name}_reliable_0.1thre_CChalf.pdf | plot_figures.ipynb | one per mouse |
| ED Fig 1b | results/mouse_summary/{model_name}_reliable_0.1thre_dp.pdf | plot_figures.ipynb | one per mouse |
| ED Fig 1b | results/mouse_summary/{model_name}_reliable_0.1thre_OSI.pdf | plot_figures.ipynb | one per mouse |
| ED Fig 2a top | results_extended/svm_stim_0.001C_250avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2a bottom | results_extended/ori_stim_0ref_250avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2b top | results_extended/svm1_relori_.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2b bottom | results_extended/svm2_relori_.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2c top | results_extended/ori1_relori.pdf | plot_extended_figures.ipynb |  |
| ED Fig 2c bottom | results_extended/ori2_relori.pdf | plot_extended_figures.ipynb |  |
| ED Fig 3a | results_extended/svm_Crange_250avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 3b top | results_extended/svm1_0.001C_Navg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 3b bottom | results_extended/svm2_0.001C_Navg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 3c top | results_extended/ori1_Navg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 3c bottom | results_extended/ori2_Navg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 4a top | results/svm_0.001C_4000avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 4a bottom | results/ori_0ref_4000avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 4b top | results/dim_4000avg_20dim.pdf | plot_extended_figures.ipynb |  |
| ED Fig 4b bottom | results/pcorr_4000avg_20dim.pdf | plot_extended_figures.ipynb |  |
| ED Fig 4c top | results_extended/cc_4000avg_20dim.pdf | plot_extended_figures.ipynb |  |
| ED Fig 4c bottom | results_extended/var_4000avg_20dim.pdf | plot_extended_figures.ipynb |  |
| ED Fig 5a top | results/stim_05-08-23_all_classes.pdf | plot_figures.ipynb |  |
| ED Fig 5a bottom | results_extended/distances_avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 5b | results_extended/relcenter_avg.pdf | plot_extended_figures.ipynb |  |
| ED Fig 6 | results/example_decoder/TX109_05_30_2023_{layer_name}_example_decoder_250rand_comp_{i}.pdf | gain_model_analyses.ipynb | mouse 1 |
| ED Fig 7a | results_extended_gain/panel_a_noise_correlations.pdf | gain_model_analyses.ipynb | mouse 1 |
| ED Fig 7b left | results_extended_gain/panel_b_mean_vs_std_raw.pdf | gain_model_analyses.ipynb | mouse 1 |
| ED Fig 7b right | results_extended_gain/panel_b_mean_vs_std_normalized.pdf | gain_model_analyses.ipynb | mouse 1 |
| ED Fig 7c | results_extended_gain/panel_c_svd_explained_variance.pdf | gain_model_analyses.ipynb | mouse 1 |
| ED Fig 7d left | results_extended_gain/svm_0.001C_250avg_mult_gain.pdf | gain_model_analyses.ipynb |  |
| ED Fig 7d right | results_extended_gain/ori_0ref_250avg_mult_gain.pdf | gain_model_analyses.ipynb |  |
| ED Fig 7e left | results_extended_gain/dim_250avg_20dim_mult_gain.pdf | gain_model_analyses.ipynb |  |
| ED Fig 7e right | results_extended_gain/pcorr_250avg_20dim_mult_gain.pdf | gain_model_analyses.ipynb |  |
| ED Fig 7f left | results_extended_gain/cc_250avg_20dim_mult_gain.pdf | gain_model_analyses.ipynb |  |
| ED Fig 7f right | results_extended_gain/var_250avg_20dim_mult_gain.pdf | gain_model_analyses.ipynb |  |
| ED Fig 8d top | results_extended_anns/rotation_robustness.pdf | more_anns_analyses.ipynb |  |
| ED Fig 8d bottom | results_extended_anns/orientation_robustness.pdf | more_anns_analyses.ipynb |  |
| ED Fig 8e left | results_extended_anns/more_anns_decoding_error_by_model.pdf | more_anns_analyses.ipynb |  |
| ED Fig 8e right | results_extended_anns/more_anns_orient_error_by_model.pdf | more_anns_analyses.ipynb |  |
| ED Fig 8f left | results_extended_anns/more_anns_inv_dim_by_model.pdf | more_anns_analyses.ipynb |  |
| ED Fig 8f right | results_extended_anns/more_anns_tuned_dim_by_model.pdf | more_anns_analyses.ipynb |  |
| ED Fig 8g left | results_extended_anns/more_anns_inv_pcorr_by_model.pdf | more_anns_analyses.ipynb |  |
| ED Fig 8g right | results_extended_anns/more_anns_tuned_pcorr_by_model.pdf | more_anns_analyses.ipynb |  |
| ED Fig 10a top | results_extended/cc_vs_dim1_250avg_20dim.pdf | plot_extended_figures.ipynb |  |
| ED Fig 10a bottom | results_extended/cc_vs_dim2_250avg_20dim.pdf | plot_extended_figures.ipynb |  |
| ED Fig 10b | results_extended/cc_250avg_20dim.pdf | plot_extended_figures.ipynb |  |
| ED Fig 10c | results_extended/var_250avg_20dim.pdf | plot_extended_figures.ipynb |  |
| SI Fig 3a top | results_extended/ori1_refs.pdf | plot_extended_figures.ipynb |  |
| SI Fig 3a bottom | results_extended/ori2_refs.pdf | plot_extended_figures.ipynb |  |
| SI Fig 3b top | results_extended/ori1_refs_20dim.pdf | plot_extended_figures.ipynb |  |
| SI Fig 3b bottom | results_extended/ori2_refs_20dim.pdf | plot_extended_figures.ipynb |  |
| SI Fig 4 | results_extended/svm_Crange_250avg_20dim.pdf | plot_extended_figures.ipynb |  |
