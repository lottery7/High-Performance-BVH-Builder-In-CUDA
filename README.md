# High Performance BVH Builder in CUDA

<img width="1614" height="951" alt="image" src="https://github.com/user-attachments/assets/404c9b6d-9141-4b27-9e80-0347fe481004" />

Репозиторий дипломной работы по теме:  
**«Реализация и оптимизация алгоритма параллельного построения BVH на GPU для рендеринга в реальном времени»**

## Todo

- [x] **Тестирующая система**
- [x] **Реализация LBVH**  
  Статья: [Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees, 2012](https://devblogs.nvidia.com/wp-content/uploads/2012/11/karras2012hpg_paper.pdf)
- [x] **Реализация H-PLOC**  
  Статья: [H-PLOC: Hierarchical Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction, 2024](https://gpuopen.com/download/HPLOC.pdf)
- [x] **Реализация Binary BVH to Wide BVH Conversion для H-PLOC**
- [ ] **Сравнение производительности**

## Датасеты

Все недостающие модели можно скачать [отсюда](https://casual-effects.com/g3d/data10/)

- Gnome - простая модель гнома: **1 297 вершин**, **764 треугольника**. Уже добавлена в репозиторий.
- Sponza - модель атриума дворца Sponza в версии Crytek: **184 330 вершин**, **262 267 треугольников**.
- Chinese Dragon - модель китайского дракона из Stanford Scan: **438 929 вершин**, **871 306 треугольников**.
- Happy Buddha - модель смеющегося Будды из Stanford Scan: **549 333 вершины**, **1 087 474 треугольника**.
- Hairball - модель массы тонких волосков от NVIDIA Research: **1 441 098 вершин**, **2 880 000 треугольников**.
- Rungholt - средневековая деревня, экспортированная из Minecraft: **12 308 528 вершин**, **6 704 264 треугольника**.
- San Miguel - детальная модель местности в Сан-Мигель-де-Альенде, Мексика: **5 933 233 вершин**, **9 980 699 треугольников**.
- Powerplant - полная модель угольной электростанции: **10 614 919 вершин**, **12 759 246 треугольников**.

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
  --experiments <список_экспериментов> \
  --scenes <список_сцен> \
  [--disable_warmup] \
  [--device <id>]
```

`run_experiments` использует adaptive warmup на CUDA events: перед замером каждый этап прогревается минимум 5 секунд, после чего раннер проверяет стабилизацию медианы. Если за 60 секунд стабилизации нет, печатается предупреждение. Для отключения прогрева есть флаг `--disable_warmup`.

Поддерживаемые эксперименты:

- `lbvh`
- `hploc`
- `hploc_bvh4`
- `hploc_bvh8`

Пример запуска:

```bash
./build/run_experiments \
  --bench_iters 10 \
  --experiments hploc,hploc_bvh4,lbvh \
  --scenes data/gnome/gnome.ply,data/powerplant/powerplant.obj
```

Пример запуска одного эксперимента на одной сцене:

```bash
./build/run_experiments \
  --bench_iters 5 \
  --disable_warmup \
  --experiments hploc_bvh4 \
  --scenes data/hairball/hairball.obj \
  --device 0
```
