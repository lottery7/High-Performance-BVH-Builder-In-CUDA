# High Performance BVH Builder In CUDA

Репозиторий дипломной работы по теме "Реализация и оптимизация алгоритма параллельного построения BVH на GPU для рендеринга в реальном времени".

Тут будет:
1) Тестирующая система
2) Реализация LBVH: [Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees, Tero Karras, 2012](https://devblogs.nvidia.com/wp-content/uploads/2012/11/karras2012hpg_paper.pdf)
3) Реализация H-PLOC: TODO
4) Реализация Fused Collapsing для H-PLOC: TODO

## Датасеты

 - [data/gnome](data/gnome) - простая модель гнома - 1.297 вершин, 764 треугольника, уже скачано в репозиторий
 - [data/powerplant](data/powerplant) - детальная модель угольной электростанции - 5.984.083 вершин, 12.759.246 треугольников, нужно скачать [с яндекс.диска](https://disk.yandex.ru/d/u4ORSCvWdITAkw) или [отсюда](https://casual-effects.com/g3d/data10/research/model/powerplant/powerplant.zip)
 - [data/san-miguel](data/san-miguel) - детальная модель местечка расположенного в поместье в Сан-Мигель-де-Альенде, Мексика - 5.933.233 вершин, 9.980.699 треугольников, нужно скачать [с яндекс.диска](https://disk.yandex.ru/d/mIP8q6V9nJiBLw) или [отсюда](https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip)

TODO

Смотрите какая красота! Это отрисованный frame buffer с ambient occlusion для powerplant:

<img width="1614" height="951" alt="image" src="https://github.com/user-attachments/assets/404c9b6d-9141-4b27-9e80-0347fe481004" />

## Структура исходников проекта в папке 

 - ```cpu_helpers/build_bvh_cpu.h``` - эталонная однопоточная CPU-реализация построения LBVH
 - ```cpu_helpers/morton_code_cpu.h``` - вспомогательная функция для построения 30-битного 3D кода Мортона для точки из единичного куба
 - ```debug/debug_bvh.h``` - не используется, использовалось для визуализации построенного BVH для отладки
 - ```debug/debug_geometry.h``` - не используется, использовалось для отладки корректности считанной геометрии (через сохранение вершин и треугольников в .ply файл)
 - ```io/camera_reader.h``` - позволяет считать ```CameraViewGPU``` из файла camera.txt - это ракурс который мы хотим отрисовать для каждой сцены, для эксперимента можете выбрать другой ракурс - для этого откройте модель в [MeshLab](https://www.meshlab.net/), расположите камеру как вам нравится, нажмите ```Ctrl+C```, затем откройте ```camera.txt``` файл и нажмите ```Ctrl+V```
 - ```io/scene_reader.h``` - позволяет считать ```SceneGeometry``` - геометрию датасета (множество 3D точек и треугольников на них покоящихся) из ```.ply```/```.obj```-файла
 - ```kernels/shared_structs/..._gpu_shared.h``` - ```CameraViewGPU```, ```BVHNodeGPU```, ```AABBGPU```, ```MortonCode``` - это аккуратно сделанные структуры в которых нет сложных типов вроде ```point3f```/```bool```/```unsigned short``` которые на разных API могут привести к разной выкладке (layout) объектов этой структуры (разный padding, разный alignment, разный fabric), благодаря этому есть шанс что объект на CPU выглядит так же как выглядит на GPU в OpenCL/CUDA/Vulkan (GLSL) кернелах

## Что уже реализовано

1) Уже сделано построение LBVH на CPU
2) Уже сделана простая трассировка лучей на GPU

