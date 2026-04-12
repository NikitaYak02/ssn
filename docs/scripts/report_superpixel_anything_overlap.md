# report_superpixel_anything_overlap.py

- Тип сценария: `report`
- Прямой запуск: **да**
- Файл: `report_superpixel_anything_overlap.py`

## Назначение

Пишет markdown-отчет о пересечениях между уже реализованными в репозитории superpixel-методами и контекстом `Superpixel Anything`.

## Минимальный запуск

```bash
python3 report_superpixel_anything_overlap.py
```

## Выход

- По умолчанию: `reports/superpixel_anything_overlap.md`

## Что важно

- Скрипт проверяет не только exact method-id overlaps, но и архитектурные / lineage overlaps.
- Для текущего состояния репозитория проверка на "совпадений нет" не проходит.
