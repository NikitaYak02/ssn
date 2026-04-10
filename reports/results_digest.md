# Сводка результатов

Этот отчет собирает самые сильные и самые показательные результаты по основным типам артефактов.

## Лучшие интерактивные прогоны

| Путь                                                                                                                                                                                                                              | Final mIoU | Coverage | Precision | Scribbles | No progress |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------- | --------- | --------- | ----------- |
| artifacts/case_studies/_two_quarters/top_left/eval                                                                                                                                                                                | 0.8828     | 0.9888   | 0.9696    | 131       | 9           |
| artifacts/interactive_runs/_eval_train01_prop_default_500                                                                                                                                                                         | 0.8779     | 0.9965   | 0.9743    | 500       | 63          |
| artifacts/case_studies/ssn_s1_v2_halfsize/test_19                                                                                                                                                                                 | 0.8701     | 0.9968   | 0.9496    | 500       | 0           |
| artifacts/case_studies/_quarter_run/eval                                                                                                                                                                                          | 0.8676     | 0.9691   | 0.9698    | 100       | 4           |
| artifacts/case_studies/_quarter_run/eval_resume_120_noprop                                                                                                                                                                        | 0.8676     | 0.9691   | 0.9698    | 100       | 1           |
| artifacts/case_studies/_quarter_run/eval_resume_300                                                                                                                                                                               | 0.8603     | 0.9865   | 0.9639    | 120       | 5           |
| artifacts/case_studies/_two_quarters/top_left/eval_100                                                                                                                                                                            | 0.8587     | 0.9666   | 0.9677    | 100       | 9           |
| artifacts/refinement/_refine_ssn_s1v2_fullcover_x05/per_pair/ssn__ssn_color_scale_0p36__ssn_fdim_20__ssn_niter_9__ssn_nspix_700__ssn_pos_scale_1p45__ssn_weights_Users_nikitayakovlev_dev_diplom_ssn_best_modelppth/train_52/run  | 0.8464     | 0.9572   | 0.9820    | 100       | 1           |
| artifacts/refinement/_refine_ssn_s1v2_fullcover_x05/per_pair/ssn__ssn_color_scale_0p36__ssn_fdim_20__ssn_niter_9__ssn_nspix_1000__ssn_pos_scale_1p45__ssn_weights_Users_nikitayakovlev_dev_diplom_ssn_best_modelppth/train_52/run | 0.8463     | 0.9570   | 0.9820    | 100       | 1           |
| artifacts/case_studies/ssn_s1_v2_halfsize/test_13                                                                                                                                                                                 | 0.8457     | 0.9996   | 0.9695    | 500       | 0           |

## Лучшие строки из sweep/refinement summary

| Источник                                                                           | Конфигурация                                                                                                                                          | Метрика | Coverage | Precision |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | -------- | --------- |
| artifacts/sweeps/_single_image_input/sweeps/train01_full_x05_100_probe/summary.csv | ssn__ssn_color_scale_0p26__ssn_fdim_20__ssn_niter_5__ssn_nspix_500__ssn_pos_scale_2p5__ssn_weights_Users_nikitayakovlev_dev_diplom_ssn_best_modelppth | 0.7433  | 0.9295   | 0.9565    |
| artifacts/sweeps/_single_image_input/sweeps/train01_full_x05_100_probe/summary.csv | slic__compactness_20p0__n_segments_500__sigma_0p0                                                                                                     | 0.7064  | 0.8661   | 0.9604    |
| artifacts/refinement/_refine_slic_s1v2_fullcover_x05/summary.csv                   | slic__compactness_9p5__n_segments_750__sigma_0p52                                                                                                     | 0.6175  | 0.8242   | 0.9357    |
| artifacts/refinement/_refine_slic_s1v2_fullcover_x05/summary.csv                   | slic__compactness_9p5__n_segments_650__sigma_0p52                                                                                                     | 0.6172  | 0.8247   | 0.9358    |
| artifacts/refinement/_refine_slic_s1v2_fullcover_x05/summary.csv                   | slic__compactness_9p5__n_segments_850__sigma_0p52                                                                                                     | 0.6172  | 0.8247   | 0.9357    |
| artifacts/refinement/_refine_slic_s1v2_fullcover_x05/summary.csv                   | slic__compactness_9p5__n_segments_850__sigma_0p42                                                                                                     | 0.6146  | 0.8286   | 0.9315    |
| artifacts/refinement/_refine_slic_s1v2_fullcover_x05/summary.csv                   | slic__compactness_9p5__n_segments_750__sigma_0p42                                                                                                     | 0.6144  | 0.8287   | 0.9312    |
| artifacts/refinement/_refine_slic_s1v2_fullcover_x05/summary.csv                   | slic__compactness_9p5__n_segments_650__sigma_0p42                                                                                                     | 0.6133  | 0.8278   | 0.9314    |
| artifacts/refinement/_refine_slic_s1v2_fullcover_x05/summary.csv                   | slic__compactness_11p0__n_segments_850__sigma_0p52                                                                                                    | 0.6096  | 0.7968   | 0.9439    |
| artifacts/refinement/_refine_slic_s1v2_fullcover_x05/summary.csv                   | slic__compactness_11p0__n_segments_750__sigma_0p52                                                                                                    | 0.6087  | 0.7961   | 0.9438    |

## Лучшие стратегии из comparison summary

| Источник                                                             | Стратегия   | mIoU   | Delta vs baseline |
| -------------------------------------------------------------------- | ----------- | ------ | ----------------- |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_03_04 | 0.5796 | 0.019065          |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_02_04 | 0.5793 | 0.018822          |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_10_04 | 0.5791 | 0.018584          |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_13_04 | 0.5789 | 0.018354          |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_12_04 | 0.5785 | 0.018020          |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_11_04 | 0.5785 | 0.018011          |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_15_04 | 0.5785 | 0.017996          |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_17_04 | 0.5785 | 0.017966          |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_16_04 | 0.5785 | 0.017958          |
| artifacts/postprocessing/_tmp_novel100_smoke/comparison_summary.json | novel_14_04 | 0.5783 | 0.017834          |

## Batch-результаты

| Источник | Mean mIoU | Coverage | Precision |
| -------- | --------- | -------- | --------- |
