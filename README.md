# ccta-pcat

Репозиторий проекта по анализу перикоронарного жира (PCAT / pFAI) на КТ-ангиографии
коронарных артерий. Данные: **ImageCAS** (КТ) и **ImageCAS-X** (сегментные маски и
центрлинии коронарных артерий).

## Структура

```
generator/   генератор тестовых/обучающих данных (скан + 2 точки -> фрагмент сосуда)
docs/        (планируется) документация проекта
detector/    (планируется) детектор/сегментатор сосудов
```

## Компоненты

### `generator/`
Пайплайн генерации примеров «скан + две точки → фрагмент сосуда» для трёх
магистралей (LAD, LCx, RCA):

- `common.py` — чтение данных, граф и корневание центрлиний, атрибуция вокселей,
  радиус по сечению и EDT, экспорт NIfTI/TIFF, дисковый кэш;
- `01_cycles.py` — анализ циклов в центрлиниях;
- `02_overlay.py` — наложение маски сосудов и центрлиний на КТ (TIFF/NIfTI/PNG для
  ImageJ/Fiji);
- `03_qc.py` — QC (геометрия, метки, snap, оторванные компоненты, радиусы, длины);
- `04_generate.py` — генерация 3 фрагментов на скан.

Подробности — в [`generator/README.md`](generator/README.md) и
[`generator/SPEC.md`](generator/SPEC.md).

## Быстрый старт

```bash
cd generator

python 04_generate.py --ids 961     # 3 фрагмента (LAD, LCx, RCA) на скан
python 04_generate.py --split test  # весь тестовый сплит
```

## Данные

- КТ ImageCAS: `/srv/fast1/y.pchelitsev/datasets/ImageCAS/data`
- ImageCAS-X: `/srv/fast1/y.pchelitsev/datasets/ImageCAS-X`

Пути заданы в `generator/common.py`. Артефакты прогонов (`out/`, `*.nii.gz`,
`*.tif`) не версионируются — воспроизводятся командами выше.
