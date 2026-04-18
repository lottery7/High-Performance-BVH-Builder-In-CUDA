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
- [ ] **Реализация Fused Collapsing для H-PLOC**  
  Статья: [Fused Collapsing for Wide BVH Construction, 2025](https://www.researchgate.net/publication/395662267_Fused_Collapsing_for_Wide_BVH_Construction)
- [ ] **Сравнение производительности**

## Датасеты

- [data/gnome](data/gnome) — простая модель гнома: **1 297 вершин**, **764 треугольника**. Уже добавлена в репозиторий.
- [data/powerplant](data/powerplant) — детальная модель угольной электростанции: **5 984 083 вершин**, **12 759 246 треугольников**.  
  Скачать можно [с Яндекс.Диска](https://disk.yandex.ru/d/u4ORSCvWdITAkw) или [отсюда](https://casual-effects.com/g3d/data10/research/model/powerplant/powerplant.zip).
- [data/san-miguel](data/san-miguel) — детальная модель местности в Сан-Мигель-де-Альенде, Мексика: **5 933 233 вершин**, **9 980 699 треугольников**.  
  Скачать можно [с Яндекс.Диска](https://disk.yandex.ru/d/mIP8q6V9nJiBLw) или [отсюда](https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip).
