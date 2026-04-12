# Geological Image Segmentation Shortlist

Этот shortlist не интегрирован в код benchmark на текущем этапе. Он нужен как отдельный доменный блок для thin-section и связанных геологических изображений.

Локальный `petroscope/HRNet` путь в репозитории рассматривается как уже существующий доменный baseline и не входит в эту пятёрку.

| Method / Paper | Task | Modality | Prompt / Supervision | Code Availability | Integration Cost | Why Relevant for Petrographic Thin Sections |
| --- | --- | --- | --- | --- | --- | --- |
| Digital petrography: Mineralogy and porosity identification using machine learning algorithms in petrographic thin section images | Mineral / pore segmentation and classification | Petrographic thin sections | Fully supervised semantic segmentation | Paper-first, code not bundled here | Medium | Напрямую работает с thin-section domain и задаёт сильный domain prior для минералогии и пористости. |
| Petrographic Microscopy with Ray Tracing and Segmentation from Multi-Angle Polarisation Whole-Slide Images | Whole-slide mineral segmentation | Multi-angle polarized petrographic microscopy | Supervised segmentation with polarization stacks | Research-grade availability, external setup expected | High | Важен для случаев, где один RGB-срез недостаточен и нужны свойства минералов под разной поляризацией. |
| Improved DeepLabV3+ for multichannel sandstone thin sections | Sandstone component segmentation | Multichannel sandstone thin sections | Fully supervised semantic segmentation | Paper-level reproducibility, external implementation needed | Medium | Хороший ориентир для multichannel thin-section segmentation и сравнения с single-image pipelines. |
| Intelligent Classification and Segmentation of Sandstone Thin Section Image Using a Semi-Supervised Framework and GL-SLIC | Semi-supervised segmentation and classification | Sandstone thin sections | Semi-supervised + GL-SLIC priors | Method described, local implementation absent | Medium | Особенно релевантен, если в дипломе важен мост между superpixel priors и разметкой с ограниченным GT. |
| Deep mineralogical segmentation based on QEMSCAN maps | Mineralogical semantic segmentation | QEMSCAN / mineral maps | Supervised semantic segmentation | Recent research artifact, external code/weights needed | High | Полезен как современный reference для automated mineralogical labeling и transfer в petrographic workflows. |

## Notes

- Для thin-section задач особенно полезно отдельно отслеживать классовый дисбаланс, мелкие минералогические фазы и overlap между близкими по текстуре материалами.
- Если эта пятёрка позже пойдёт в исполнение, лучше делать отдельный domain benchmark, а не смешивать его с общей пятёркой интерактивных методов.
- При следующем шаге интеграции разумно сравнивать не только `mIoU`, но и per-class recall по редким минералам и porosity-related classes.
