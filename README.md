# High Performance BVH Builder in CUDA

<img width="1614" height="951" alt="image" src="https://github.com/user-attachments/assets/404c9b6d-9141-4b27-9e80-0347fe481004" />

Репозиторий дипломной работы по теме:  
**«Реализация и оптимизация алгоритма параллельного построения BVH на GPU для рендеринга в реальном времени»**

## TODO

### Основные задачи

- [x] **Тестирующая система**
- [x] **Реализация LBVH**  
  Статья: [Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees, 2012](https://devblogs.nvidia.com/wp-content/uploads/2012/11/karras2012hpg_paper.pdf)
- [x] **Реализация H-PLOC**  
  Статья: [H-PLOC: Hierarchical Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction, 2024](https://gpuopen.com/download/HPLOC.pdf)
- [x] **Реализация Binary BVH to Wide BVH Conversion для H-PLOC**
- [ ] **Сравнение производительности**

## Датасеты

- [data/gnome](data/gnome) — простая модель гнома: **1 297 вершин**, **764 треугольника**. Уже добавлена в репозиторий.
- [data/powerplant](data/powerplant) — детальная модель угольной электростанции: **5 984 083 вершин**, **12 759 246 треугольников**.  
  Скачать можно [с Яндекс.Диска](https://disk.yandex.ru/d/u4ORSCvWdITAkw) или [отсюда](https://casual-effects.com/g3d/data10/research/model/powerplant/powerplant.zip).
- [data/san-miguel](data/san-miguel) — детальная модель местности в Сан-Мигель-де-Альенде, Мексика: **5 933 233 вершин**, **9 980 699 треугольников**.  
  Скачать можно [с Яндекс.Диска](https://disk.yandex.ru/d/mIP8q6V9nJiBLw) или [отсюда](https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip).

## Локальная сборка

Сборка через CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target run_experiments -j
```

При необходимости тесты топологии H-PLOC можно собрать так:

```bash
cmake --build build --target hploc_topology_tests -j
```

## Локальный запуск

Запускать `run_experiments` нужно из корня репозитория, чтобы относительные пути к `data/` и `results/` работали корректно.

Формат запуска:

```bash
./build/run_experiments \
  --bench_iters <N> \
  --warmup_iters <N> \
  --experiments <список_экспериментов> \
  --scenes <список_сцен> \
  [--device <id>]
```

Поддерживаемые эксперименты:

- `cpu_lbvh`
- `lbvh`
- `kitten_lbvh`
- `hploc`
- `hploc_bvh4`
- `hploc_bvh8`

Пример запуска:

```bash
./build/run_experiments \
  --bench_iters 10 \
  --warmup_iters 10 \
  --experiments hploc,hploc_bvh4,lbvh \
  --scenes data/gnome/gnome.ply,data/powerplant/powerplant.obj
```

Пример запуска одного эксперимента на одной сцене:

```bash
./build/run_experiments \
  --bench_iters 1 \
  --warmup_iters 0 \
  --experiments hploc_bvh4 \
  --scenes data/hairball/hairball.obj \
  --device 0
```
