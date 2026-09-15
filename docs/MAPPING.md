# Where every number comes from

Each row of the paper points at the scripts that produced it and at the stored result files
the numbers were read from.  The table is generated from `meta/SOURCES.md`, which the build
extracts from the manuscript when the internal `*Source:*` lines are removed.

| paper table / section | scripts | stored results |
|---|---|---|
| 3.2 Compression–transformation compatibility | `229_star_existence_test.py` | `results/star_existence_test.json` |
| Table 3 | — | `results/deep_linear_transport.json`, `results/transport_theory.json` |
| 3.5 The piecewise-linear layer: what a real encoder does | — | `results/depth_map_summary.json` |
| Table 5 | — | `results/dense_continuous_v3.json`, `results/harmonic_controls.json`, `results/lattice_v4.json` |
| 4.6 When the measured carrier supports a linear action: a design condition | `259_carrier_optimality_identity.py` | `results/carrier_optimality_identity.json` |
| 4.6 When the measured carrier supports a linear action: a design condition | `261_real_site_closure.py` | `results/real_site_closure.json` |
| Table 6 | — | `results/heat_semigroup_pilot.json`, `results/multiattr_summary.json`, `results/rotation_pilot.json`, `results/second_attribute_chain.json` |
| Table 7 | — | `results/action_fragmentation.json`, `results/causal_cifar_summary.json`, `results/family_capacity.json`, `results/hue_prediction_demo.json` |
| Table 8 | — | — |
| Table 9 | — | `results/multipath_composition.json` |
| Table 10 | — | `results/depth_map_summary.json` |
| 6.2 The geometry of changing realisability | — | `results/4.json`, `results/7.json` |
| 6.3 The readout–transformation pairing | — | `results/equivariance_theory_check.json` |
| Table 11 | — | `results/quotientization.json` |
| Table 12 | — | `results/linear_dynamics_counterexample.json` |
| Table 13 | `263_two_term_heldout.py` | `results/two_term_heldout_resnet50.json` |
| 7.2 The interface: learned encoding, fixed action | `265_interface_action_comparison.py` | `results/interface_action_comparison.json` |
| Table 14 | — | `results/CNN_s0_d90.json`, `results/_d90.json`, `results/hue_probe_app.json` |
| Table 15 | — | `results/code3d_sv_lowext.json`, `results/detector_o8_multiseed.json`, `results/heat_tables.json`, `results/hue_probe_app.json`, `results/multipath_composition.json`, `results/region_rho_demo.json`, `results/usage_rule_scale_strat.json` |
| Table 16 | — | `results/boundary_covariates.json`, `results/boundary_independent.json`, `results/boundary_largescale.json` |
| 8. Discussion | — | `results/heat_semigroup_structure.json` |
| Table A.3 | — | `results/boundary_largescale.json` |
| Table B.1 | — | `results/deep_linear_transport.json`, `results/transport_theory.json` |
| Table B.1 | — | `results/depth_map_summary.json` |
| Table B.4 | — | — |
| Table C1 | `01_gen_synthetic.py`, `115_sv_lowext.py`, `30_gen_dense.py`, `lattice.py` | `results/suite.npz` |
| Table C1 | `188_depth_map_harness.py` | `results/heat_semigroup_pilot.json` |
| Table C2 | `188_depth_map_harness.py`, `2_dense.py`, `31_dense_features.py` | `results/depth_map_summary.json` |
| Table C3 | — | `results/depth_map_summary.json` |
| Table C4 | `188_depth_map_harness.py` | `results/depth_map_summary.json`, `results/heat_semigroup_pilot.json` |
| Table C4 | `136_causal_cifar.py`, `146_unseen_delta_cifar.py`, `194_composition_heat.py`, `231_multipath_composition.py` | `results/CNN_s0_stage1.json`, `results/composition_heat.json`, `results/multipath_composition.json` |
| Table C4 | `136_causal_cifar.py`, `198_ceiling_regularisation.py`, `218_estimator_control.py` | `results/ceiling_regularisation.json`, `results/estimator_control.json` |
| Table C5 | `188_depth_map_harness.py` | `results/_d90.json`, `results/depth_map_summary.json` |
| Table C6 | `109_region_rho_demo.py`, `112_usage_rule_strat.py`, `138_boundary_independent.py`, `154_boundary_covariates.py`, `169_boundary_largescale.py`, `234_fig_correspondence.py` | `results/boundary_covariates.json`, `results/boundary_independent.json`, `results/boundary_largescale.json`, `results/fig_correspondence.json`, `results/region_rho_demo.json`, `results/usage_rule_scale_strat.json` |
| Table C7 | — | `results/crossing_consumer_resnet50.json` |
| Table D.1 | — | `results/clip_vit_probe.json`, `results/dinov2_vitb14_probe.json`, `results/dinov2_vits14_probe.json`, `results/harmonic_controls.json`, `results/resnet50_probe.json`, `results/vitb16_supervised_probe.json` |
| Table D.2 | — | `results/dense_continuous_v3.json` |
| Table D.3 | — | `results/harmonic_controls.json`, `results/lattice_v4.json` |
| Table D.4 | — | `results/depth_map_summary.json` |
| Table D.4 | — | `results/_hue.json`, `results/depth_map_summary.json` |
| Table D.5 | — | `results/depth_map_summary.json` |
| Table D.5 | — | `results/_heat.json`, `results/depth_map_summary.json` |
| Table D.6 | — | `results/depth_map_summary.json` |
| Table D.6 | — | `results/depth_map_summary.json` |
| Table D.7 | — | `results/quotientization.json` |
| Table D.8 | — | `results/quotientization.json` |
| Table D.8 | — | `results/quotientization.json` |
| Table D.9 | — | `results/fig_composition.json`, `results/multipath_composition.json` |
| Table D.10 | — | `results/fig_composition.json`, `results/multipath_composition.json` |
| Table D.11 | — | `results/composition_heat.json` |
| Table D.11 | — | `results/composition_heat.json`, `results/heat_tables.json` |
| Table D.12 | — | `results/multipath_composition.json` |
| Table D.12 | — | `results/fig_composition.json`, `results/multipath_composition.json` |
| Table D.13 | — | `results/multiattr_summary.json` |
| Table D.13 | — | `results/multiattr_summary.json` |
| Table D.14 | `159_realmultiattr_intervention.py`, `161_multiattr_summary.py`, `252_update_real_multiattr_table.py` | `results/multiattr_summary.json` |
| Table D.15 | — | `results/second_attribute_chain.json` |
| Table D.16 | — | `results/second_attribute_chain.json` |
| Table D.16 | — | `results/second_attribute_chain.json` |
| Table D.17 | — | `results/CNN_s0.json`, `results/rotation_pilot.json` |
| Table D.18 | — | `results/boundary_independent.json`, `results/boundary_largescale.json` |
| Table D.19 | — | `results/boundary_covariates.json` |
| Table D.19 | — | `results/boundary_covariates.json` |
| Table D.19 | — | `results/crossing_consumer_resnet50.json` |
| Table D.20 | `260_two_term_decomposition.py` | `results/two_term_decomposition_resnet50.json` |
| Table D.21 | `261_real_site_closure.py` | `results/real_site_closure.json` |
| Table D.22 | `258_measure_lipschitz.py` | `results/lipschitz_resnet50.json` |
| Table D.23 | `229_star_existence_test.py` | `results/star_existence_test.json` |
| Appendix E: Method boxes | `07_locality_global_vs_local.py`, `15_closure_check.py`, `20c_fourier_fit.py` | `results/3.json`, `results/3_K3.json` |
| Appendix E: Method boxes | `115_sv_lowext.py`, `79_code3d_v3.py`, `95_code3d_sv_map.py`, `arm_anchor.py` | `results/code3d_sv_lowext.json`, `results/code3d_v3.json` |
| Appendix E: Method boxes | `140_gen_yolo_hue.py`, `141_train_yolo_hue.py`, `156_detector_o8_multiseed.py`, `157_multiattribute_intervention.py`, `159_realmultiattr_intervention.py`, `173_record_yolo_val.py`, `188_depth_map_harness.py` | `results/detector_o8_multiseed.json`, `results/yolo_hue_val.json` |
| Table E1 | `100_perf_reforward_vs_rho.py`, `109_region_rho_demo.py` | `results/perf_reforward_vs_rho.json`, `results/region_rho_demo.json` |
| Table E1 | — | `results/perf_reforward_vs_rho.json`, `results/region_rho_demo.json` |
| Table F.1 | — | `results/causal_cifar_summary.json` |
| Table F.1 | — | `results/causal_cifar_summary.json` |
| Table F.2 | `c10_routeB_full.py`, `c10_routeB_summary.py`, `c10_routeB_train.py` | `results/CNN_s0.json`, `results/c10_routeB/summary.json` |
| Table F.2 | — | `results/c10_routeB/summary.json` |
| Table F.3 | — | — |
| Table F.4 | — | `results/detector_o8_multiseed.json` |
| Table F.4 | — | `results/causal_intervention_hue.json`, `results/detector_o8_multiseed.json`, `results/fig_composition.json` |
| Table F.5 | — | `results/real_task_consumer.json` |
| Table F.5 | — | `results/real_task_consumer.json` |
| Table F.6 | — | `results/boundary_covariates.json`, `results/boundary_largescale.json` |
| Table F.6 | — | `results/boundary_covariates.json`, `results/boundary_largescale.json` |
| Table F.7 | — | `results/CNN_s0_d90.json`, `results/_d90.json`, `results/code3d_sv_lowext.json`, `results/code3d_v3.json`, `results/hue_probe_app.json`, `results/usage_rule_scale_strat.json` |
| Table F.8 | — | `results/detector_o8_multiseed.json`, `results/multipath_composition.json`, `results/region_rho_demo.json`, `results/usage_rule_scale_strat.json` |
| Table F.9 | — | `results/perf_reforward_vs_rho.json`, `results/region_rho_demo.json` |
| Table F.10 | `265_interface_action_comparison.py` | `results/interface_action_comparison.json` |
| Table G.1 | — | — |
| Table G.1 | — | — |

## Scripts in this release

104 scripts are shipped: the experiment scripts the paper cites, the twelve
figure scripts plus their caption builder, the four verification scripts, and everything
those import.  `scripts/repo_paths.py` resolves every filesystem root.

## Stored results

125 result files are shipped, all under `results/`.  They are exactly the files
the manuscript quotes, so `python tools/verify_release.py` can re-derive every number
