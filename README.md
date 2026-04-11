# SSN Repository Map

Репозиторий приведен к стабильной структуре, где код, документация, модели, артефакты и отчеты разделены по отдельным зонам.

## Что где лежит

- `lib/`, `superpixel_annotator/`, top-level `*.py`, `tests/` — основной код и точки входа.
- `docs/optimization/` — исторические и текущие материалы по оптимизации пайплайна обучения.
- `docs/scripts/` — подробная документация по запуску всех CLI-скриптов.
- `docs/notebooks/` — ноутбуки и исследовательские заметки.
- `models/checkpoints/` — сохраненные checkpoint-файлы, которые используются в примерах и evaluation.
  В репозитории хранится `models/checkpoints/best_model.pth` для SSN-сценариев из документации.
- `artifacts/` — все прогоны, sweep/refinement, precomputed/spanno, постобработка, case-study и debug-артефакты.
- `reports/` — человекочитаемые отчеты по текущему состоянию репозитория.
- `reports/generated/` — машинно-читаемые инвентари и каталоги, из которых собираются отчеты.
- `tools/` — утилиты миграции структуры, инвентаризации, генерации отчетов и документации.

## С чего начать

- Общая карта скриптов: [docs/scripts/index.md](docs/scripts/index.md)
- Окружения и запуск: [docs/scripts/run_environments.md](docs/scripts/run_environments.md)
- Оптимизации обучения: [docs/optimization/README.md](docs/optimization/README.md)
- Отчеты по прогонам и результатам: [reports/index.md](reports/index.md)

## Полезные команды

```bash
python3 tools/build_all.py --repo-root .
```

Пересобирает:

- инвентарь файлов и прогонов;
- каталог CLI-скриптов и их документацию;
- Markdown-отчеты по структуре и результатам.

```bash
python3 tools/migrate_layout.py --repo-root .
```

Показывает dry-run план миграции в каноническую структуру.

```bash
python3 tools/migrate_layout.py --repo-root . --apply
```

Применяет переносы файлов и каталогов согласно правилам из `tools/repo_conventions.py`.
