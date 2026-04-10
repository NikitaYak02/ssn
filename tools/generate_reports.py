#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Markdown reports from inventory artifacts.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--generated-dir",
        default="reports/generated",
        help="Directory with inventory JSON/CSV outputs.",
    )
    parser.add_argument("--reports-dir", default="reports", help="Directory for Markdown reports.")
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def human_size(size_bytes: int | float | None) -> str:
    if size_bytes in (None, ""):
        return "-"
    value = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    return f"{value:.1f} {units[unit_index]}"


def md_table(rows: list[list[str]]) -> str:
    if not rows:
        return "_Нет данных._"
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)) + " |"

    header = render(rows[0])
    divider = "| " + " | ".join("-" * widths[index] for index in range(len(rows[0]))) + " |"
    body = [render(row) for row in rows[1:]]
    return "\n".join([header, divider, *body])


def take_top(items: list[dict[str, Any]], key: str, limit: int = 10) -> list[dict[str, Any]]:
    valid = [item for item in items if item.get(key) not in (None, "", "nan")]
    return sorted(valid, key=lambda item: float(item.get(key) or 0.0), reverse=True)[:limit]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_index_report(top_level: list[dict[str, Any]], runs: list[dict[str, Any]], anomalies: list[dict[str, Any]]) -> str:
    zone_counts = Counter(item["classification"] for item in top_level)
    family_counts = Counter(item.get("family", "unknown") for item in runs)
    return f"""# Отчеты по репозиторию

Этот раздел собран автоматически на основе полной инвентаризации файлов, прогонов и итоговых артефактов репозитория.

## Быстрый обзор

- Верхнеуровневых записей: **{len(top_level)}**
- Каталогов/сводок прогонов: **{len(runs)}**
- Аномалий и неполных мест для ручной проверки: **{len(anomalies)}**

## Карта отчетов

- [Структура репозитория](repo_structure.md)
- [Семейства прогонов](run_families.md)
- [Сводка результатов](results_digest.md)
- [Хранилище и очистка](storage_and_cleanup.md)

## Зоны репозитория

{md_table([["Зона", "Количество"], *[[name, str(count)] for name, count in sorted(zone_counts.items())]])}

## Семейства прогонов

{md_table([["Семейство", "Количество"], *[[name, str(count)] for name, count in sorted(family_counts.items())]])}
"""


def build_repo_structure_report(top_level: list[dict[str, Any]]) -> str:
    rows = [["Путь", "Тип", "Классификация", "Размер"]]
    for item in sorted(top_level, key=lambda row: float(row.get("size_bytes") or 0.0), reverse=True):
        rows.append(
            [
                item["path"],
                "dir" if item["is_dir"] else "file",
                item["classification"],
                human_size(item.get("size_bytes")),
            ]
        )

    explanation = """# Структура репозитория

Репозиторий разделен на стабильные зоны:

- `docs/` — живая документация, включая разделы по оптимизациям и запуску скриптов.
- `reports/` — автоматически собираемые отчеты и машинно-читаемые инвентари.
- `models/` — контрольные точки и веса.
- `artifacts/` — экспериментальные прогоны и вспомогательные данные.
- `tools/` — утилиты инвентаризации, миграции и генерации документации.
- top-level Python entrypoint'ы, `lib/`, `superpixel_annotator/`, `tests/` — рабочая кодовая поверхность проекта.

Ниже показано фактическое содержимое корня после миграции.
"""
    return explanation + "\n\n" + md_table(rows)


def build_run_families_report(runs: list[dict[str, Any]]) -> str:
    family_rows = [["Семейство", "Количество", "Статусы"]]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_family[run.get("family", "unknown")].append(run)
    for family, items in sorted(by_family.items()):
        statuses = Counter(item.get("status", "unknown") for item in items)
        family_rows.append([family, str(len(items)), ", ".join(f"{name}:{count}" for name, count in sorted(statuses.items()))])

    examples_rows = [["Семейство", "Пример пути", "Комментарий"]]
    comments = {
        "interactive-run": "Одиночные и серийные интерактивные прогоны с `metrics.csv` и часто с `run.log`.",
        "sweep": "Серии перебора параметров и компактные smoke-проверки.",
        "refinement": "Локальное доуточнение параметров на выбранных парах изображений.",
        "postprocessing": "Сравнение стратегий постобработки и их влияние на baseline.",
        "case-study": "Отдельные экспериментальные сценарии и демонстрации.",
        "training": "Каталоги обучения и fine-tuning со своими логами и checkpoint-ами.",
        "precomputed": "Заранее рассчитанные `spanno`-артефакты.",
        "debug": "Отладочные и временные каталоги, которые не стоит трактовать как основные результаты.",
        "uncategorized": "Каталоги, которые требуют ручной проверки.",
    }
    for family, items in sorted(by_family.items()):
        sample = sorted(items, key=lambda row: row["path"])[0]["path"]
        examples_rows.append([family, sample, comments.get(family, "Без дополнительного комментария.")])

    return f"""# Семейства прогонов

В отчетах используется следующая интерпретация статусов:

- `main` — основной или содержательный результат.
- `smoke` — быстрый sanity-check.
- `debug` — отладочный запуск.
- `case-study` — демонстрационный или исследовательский сценарий.
- `archive` — полезная, но уже не первичная серия экспериментов.
- `uncategorized` — требует ручной проверки.

## Распределение по семействам

{md_table(family_rows)}

## Примеры семейств

{md_table(examples_rows)}
"""


def build_results_digest_report(runs: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> str:
    interactive = [run for run in runs if run.get("kind") == "interactive_run"]
    top_interactive = take_top(interactive, "final_miou", limit=10)

    summary_rows = [row for row in aggregates if row.get("kind") == "summary_row"]
    top_summaries = take_top(summary_rows, "metric_value", limit=10)

    comparison_rows = [row for row in aggregates if row.get("kind") == "comparison_method"]
    top_comparisons = take_top(comparison_rows, "metric_value", limit=10)

    batch_rows = [row for row in aggregates if row.get("kind") == "batch_mean"]
    top_batches = take_top(batch_rows, "metric_value", limit=10)

    interactive_table = [["Путь", "Final mIoU", "Coverage", "Precision", "Scribbles", "No progress"]]
    for item in top_interactive:
        interactive_table.append(
            [
                item["path"],
                f"{float(item.get('final_miou') or 0.0):.4f}",
                f"{float(item.get('final_coverage') or 0.0):.4f}",
                f"{float(item.get('final_annotation_precision') or 0.0):.4f}",
                str(int(float(item.get("final_scribbles") or 0))),
                str(item.get("log_no_progress_count") or 0),
            ]
        )

    summary_table = [["Источник", "Конфигурация", "Метрика", "Coverage", "Precision"]]
    for item in top_summaries:
        summary_table.append(
            [
                item["source"],
                item.get("label") or "-",
                f"{float(item.get('metric_value') or 0.0):.4f}",
                f"{float(item.get('coverage') or 0.0):.4f}" if item.get("coverage") is not None else "-",
                f"{float(item.get('annotation_precision') or 0.0):.4f}" if item.get("annotation_precision") is not None else "-",
            ]
        )

    comparison_table = [["Источник", "Стратегия", "mIoU", "Delta vs baseline"]]
    for item in top_comparisons:
        comparison_table.append(
            [
                item["source"],
                item.get("label") or "-",
                f"{float(item.get('metric_value') or 0.0):.4f}",
                f"{float(item.get('delta_miou_vs_baseline') or 0.0):.6f}" if item.get("delta_miou_vs_baseline") is not None else "-",
            ]
        )

    batch_table = [["Источник", "Mean mIoU", "Coverage", "Precision"]]
    for item in top_batches:
        batch_table.append(
            [
                item["source"],
                f"{float(item.get('metric_value') or 0.0):.4f}",
                f"{float(item.get('coverage') or 0.0):.4f}" if item.get("coverage") is not None else "-",
                f"{float(item.get('annotation_precision') or 0.0):.4f}" if item.get("annotation_precision") is not None else "-",
            ]
        )

    return f"""# Сводка результатов

Этот отчет собирает самые сильные и самые показательные результаты по основным типам артефактов.

## Лучшие интерактивные прогоны

{md_table(interactive_table)}

## Лучшие строки из sweep/refinement summary

{md_table(summary_table)}

## Лучшие стратегии из comparison summary

{md_table(comparison_table)}

## Batch-результаты

{md_table(batch_table)}
"""


def build_storage_report(top_level: list[dict[str, Any]], runs: list[dict[str, Any]], anomalies: list[dict[str, Any]]) -> str:
    largest_top_level = sorted(top_level, key=lambda item: float(item.get("size_bytes") or 0.0), reverse=True)[:20]
    top_level_table = [["Путь", "Классификация", "Размер"]]
    for item in largest_top_level:
        top_level_table.append([item["path"], item["classification"], human_size(item.get("size_bytes"))])

    largest_runs = sorted(runs, key=lambda item: float(item.get("size_bytes") or 0.0), reverse=True)[:20]
    run_table = [["Путь", "Тип", "Семейство", "Размер"]]
    for item in largest_runs:
        run_table.append([item["path"], item.get("kind", "-"), item.get("family", "-"), human_size(item.get("size_bytes"))])

    anomaly_table = [["Путь", "Проблема"]]
    for item in anomalies[:50]:
        anomaly_table.append([item["path"], item["issue"]])

    cleanup_candidates = [
        "- каталоги со статусом `debug` и `smoke` просмотреть первыми, если нужно освободить место;",
        "- каталоги `artifacts/uncategorized/` держать коротким списком и разбирать вручную;",
        "- большие case-study каталоги хранить, только если они нужны для воспроизведения или визуализации;",
        "- precomputed/spanno артефакты можно архивировать отдельно от кода, если требуется облегчить рабочую копию.",
    ]

    return f"""# Хранилище и очистка

## Самые тяжелые верхнеуровневые элементы

{md_table(top_level_table)}

## Самые тяжелые каталоги прогонов

{md_table(run_table)}

## Кандидаты на ручную проверку

{md_table(anomaly_table)}

## Практические рекомендации по очистке

{chr(10).join(cleanup_candidates)}
"""


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.repo_root).resolve()
    generated_dir = (root / args.generated_dir).resolve()
    reports_dir = (root / args.reports_dir).resolve()

    repo_inventory = load_json(generated_dir / "repo_inventory.json")
    run_catalog = load_json(generated_dir / "run_catalog.json")
    aggregate_catalog = load_json(generated_dir / "aggregate_catalog.json")
    anomalies_payload = load_json(generated_dir / "anomalies.json")

    top_level = repo_inventory.get("top_level", [])
    runs = run_catalog.get("runs", [])
    aggregates = aggregate_catalog.get("aggregates", [])
    anomalies = anomalies_payload.get("anomalies", [])

    write_text(reports_dir / "index.md", build_index_report(top_level, runs, anomalies))
    write_text(reports_dir / "repo_structure.md", build_repo_structure_report(top_level))
    write_text(reports_dir / "run_families.md", build_run_families_report(runs))
    write_text(reports_dir / "results_digest.md", build_results_digest_report(runs, aggregates))
    write_text(reports_dir / "storage_and_cleanup.md", build_storage_report(top_level, runs, anomalies))

    print(f"Reports written to: {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
