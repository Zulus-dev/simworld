SYSTEM PROMPT: ADCE ENGINE v3.3 DEVELOPMENT ARCHITECT
ОБЗОР ПРОЕКТА
ADCE Engine v3.3 — это Task-Agnostic, детерминированная вычислительная песочница для симуляции агентов на ГА и Матки с роевым интеллектом.

Цель: Генерация датасетов во VRAM/RAM для обучения внешних нейросетей.

Стек: Python, PyTorch, CUDA, ModernGL (Local), torchvision/imageio (Cloud).

1. СТРАТЕГИЧЕСКИЕ ИНВАРИАНТЫ И ПАМЯТЬ
1.1 Битовый детерминизм гетерогенных вычислений
Множитель Fixed-Point: M=2 
16
  (65536). Координаты, энергия, фитнес — строго в int.



структура файлов и назначение 


Стохастика: Только изолированный torch.Generator(device=COMPUTE_DEVICE).

ЧТО ДЕЛАЕМ (DO)	ЧТО ЗАПРЕЩЕНО (DON'T)
Масштабирование величин: M≤2 
16
  (макс. 2 
20
 ).	Использование float32/float16/bfloat16 для глобальных состояний, наград и ГА.
Перемножение шкал: в torch.int64 со сдвигом вправо на 16 бит.	Допущение переполнения знакового torch.int64 (расчеты >2 
64
  промежуточно).
Инициализация фиксированным сидом из config.json.	Использование глобальных torch.rand(), numpy.random.
Блокировка окружения при старте: CUBLAS_WORKSPACE_CONFIG=":4096:8", deterministic = True.	torch.backends.cudnn.benchmark = True.

1.2 Принцип Absolute Zero-Allocation
Memory Tank: Вся память выделяется одним монолитным блоком T_global (torch.uint8) на этапе инициализации.

Workspace Alignment: 15% физической памяти GPU резервируются (не размечаются) под внутренние маски PyTorch (index_add_).

ЧТО ДЕЛАЕМ (DO)	ЧТО ЗАПРЕЩЕНО (DON'T)
Изменение геометрии только через представления: .view(), .expand(), .transpose(), .narrow().	Вызовы cudaMalloc / malloc, неявные аллокации промежуточных масок внутри step() и render().
Предаллокация фиксированных статических скретч-буферов (Scratch Buffers) в конструкторах модулей.	Срезы по динамическим маскам: T_dna_pool[T_entities > 0] (вызывает скрытый аллокатор CUDA).
Выравнивание памяти по 128-бит (16 байт) по формуле: (offset + 15) & ~15.	Прямой слайсинг байтового тензора с изменением типа без контроля кратности (assert).

1.3 Независимость от задач (Task Agnosticism)
Адаптация: ДНК кодирует параметры обучения Хебба (η,γ). Изменение синапсов: ΔW 
ij
​
 =clip(η⋅(O 
i
​
 ⋅I 
j
​
 −γ⋅O 
j
​
 ⋅W 
ij
​
 ),−V 
max
​
 ,V 
max
​
 ).

Сенсоры: Пространство приводится модулем среды к плоскому вектору абстрактных токенов через Perceiver (total_spatial_tokens).

Действия (Affordance Interface): Агент выдает абстрактный тензор A 
map
​
  (K токенов). Среда сама дешифрует их.

2. АРХИТЕКТУРА ДАННЫХ (STRUCTURE OF ARRAYS - SoA)
2.1 Структура монолита T_global
Разметка памяти строго по SoA с валидацией смещений:

[ T_global (torch.uint8) ]
├── T_environment [uint8] -> Битовая упаковка слоев ячейки
├── T_entities    [int16] -> Индексы сущностей (0: пусто, >0: ГА-агенты, <0: когорты Матки)
└── T_dna_pool    [int32] -> Матрица геномов всей популяции [Max_Agents, Genome_Length]
2.2 Битовая упаковка T_environment (1 байт на ячейку)
Бит 0 (0x01): Проходимость (0 — стена, 1 — пустота).

Бит 1 (0x02): Спавнер ресурсов (0 — пассивный, 1 — активный).

Биты 2–4 (0x1C): Феромон А Роя (8 уровней интенсивности, извлечение: (cell & 0x1C) >> 2).

Биты 5–7 (0xE0): Феромон Б Матки (8 уровней интенсивности, извлечение: (cell & 0xE0) >> 5).

2.3 Двухуровневая сетка (Two-Level Block-Sparse Grid)
Верхний уровень: Индексационный тензор макро-блоков (8×8×8), доступ за O(1).

Нижний уровень: Линейный Flat-индекс вектора памяти: Linear_Index=∑X 
c
​
 ⋅strides[c]. Для 2D высшие оси = 1, strides = 0.

3. ВЫЧИСЛИТЕЛЬНЫЕ МОДУЛЕЙ И ИНТЕРФЕЙСЫ
3.1 Модуль 0: ADCE-Core API (Двухконтурный обмен)
Контур А (In-Process): Прямые тензорные ссылки (Python reference) на Пинг-Понг буферы VRAM/RAM. Без IPC.

Контур Б (Inter-Process CUDA IPC): Передача дескрипторов изолированных буферов обмена через .untyped_storage()._share_cuda_().

ЧТО ДЕЛАЕМ (DO)	ЧТО ЗАПРЕЩЕНО (DON'T)
Защита от гонок: Фиксация окончания шага через torch.cuda.Event и вызов .wait_stream() принимающим процессом.	Чтение разделяемой памяти внешней нейросетью без явной синхронизации потоков CUDA.
Пинг-Понг двойная буферизация: Сеть читает буфер А (шаг t), физический движок пишет в буфер Б (шаг t+1).	Блокировка конвейера (Pipeline Stall) из-за синхронного ожидания завершения инференса сети.
Передача только изолированных буферов обмена.	Передача хэндла закрытого общего монолита T_global через IPC.

3.2 Модуль 1 & 3: Tensor-Engine и Эволюция (Branchless & No-Sort)
Перемещения: Вычисление позиций без if/else через битовые маски.

Пример: X_next = pass_mask * X_proposed + (1 - pass_mask) * X_current.

Коллизии: Разрешение конфликтов шага нескольких агентов через атомарный torch.index_select и index_add_ во вспомогательный T_collision_scratch.

Селекция (No-Sort GA): Кроссинговер и мутации прямо в T_dna_pool. Выбор родительских пар — векторизованный турнир через генерацию случайных индексов и логическое маскирование. Сортировки (torch.sort, torch.topk) запрещены.

3.3 Модуль 2 & 4: Суррогат-WM и Ассоциативная память (Гиппокамп)
Surrogate-WM: Vision Transformer (ViT) + RSSM. Фрактальный Гиперкубический Патч-Проектор сохраняет объем патча в токенах константным для любой мерности пространства (D-мерного). Пространственное смещение сохраняется через Relative Continuous Spatial Attention.

HDC-Граф памяти: Неиерархический граф на гипервекторах (лимит N 
max
​
 =50000 узлов).

Очистка памяти: Безветвящийся масочный механизм стохастического когнитивного дропаута: ΔW 
node
​
 =−λ⋅(1− 
Max_Centrality
Centrality 
node
​
 
​
 ) 
p
 ⋅Δt. Освобожденные индексы переиспользуются как циклический буфер (LRU).

4. МОДУЛЬ ВИЗУАЛИЗАЦИИ (RENDER PLANE)
Работает по паттерну Blit-View как независимый наблюдатель. Запрашивает ссылки через _create_soa_views(). Не имеет права менять логику симуляции.

4.1 Алгоритм бескопийной сборки кадра (True Zero-Allocation)
Сборка кадра векторизованно, строго in-place в предаллоцированный статический буфер T_framebuffer ([H, W, 3], uint8) с использованием аргумента out=.

ЧТО ДЕЛАЕМ (DO)	ЧТО ЗАПРЕЩЕНО (DON'T)
Стены: Выделение бита 0 через torch.bitwise_and(..., out=env_scratch), инверсия через torch.eq, наложение цвета через torch.mul(..., out=color_broadcast_scratch).	Использование torch.where() или бинарных перемножений со скалярами вида wall_mask * COLOR.
Феромоны: Извлечение сдвигом torch.bitwise_right_shift, масштабирование in-place .mul_(32), сложение .add_() в срез канала T_framebuffer[..., 1].	Прямое присвоение или создание новых промежуточных тензоров цветовых каналов во VRAM.
Сущности: Логическое маскирование (entities > 0 / < 0). Зануление пикселей под объектами умножением на инвертированную маску, затем add_ константного цвета.	Использование циклов for и условий if/else попиксельно при сборке кадра.

4.2 Режимы рендеринга
Локальный (Бэкенд 1): PyTorch + ModernGL через CUDA-OpenGL Interoperability. Графический движок OpenGL забирает прямой указатель на T_framebuffer. Копирование через CPU запрещено.

Облачный (Бэкенд 2): Headless-режим. Запись кадра по триггеру (раз в N тысяч шагов). Упаковка напрямую в памяти в .mp4 через аппаратный NVENC GPU с сохранением на диск.

5. ЗАЩИТА ЯДРА И ОПТИМИЗАЦИЯ
5.1 Математическая защита
Награды: SymLog-трансформация: f(x)=sign(x)⋅ln(∣x∣+1). Строго заменяет torch.clamp, сохраняя градиенты непрерывными.

Регуляризация Матки: Энтропийный штраф Кульбака-Лейблера с динамическим коэффициентом β 
t
​
  для подавления ВЧ-колебаний среды:

R 
m
​
 =α⋅Goal_Metric−β 
t
​
 ⋅D 
KL
​
 (P(S 
stigmergy
​
 )∥U(S))
β 
t
​
 =β 
max
​
 ⋅exp(−κ⋅ 

​
  
dt
dFitness
​
  

​
 )
5.2 Критические правила кодинга (Framework Pitfalls Control)
Побитовые операции: Выполнять только через функциональные аналоги PyTorch с pre-allocated приемником: torch.bitwise_and(a, b, out=self.scratch).

Тотальный In-place: Запрещено X = X + Y. Разрешено только X.add_(Y) или X += Y.

Индексация: Вместо динамических масок использовать только torch.index_select(src, dim, pre_allocated_indices).

ИНВАРИАНТНЫЕ КОНТРАКТЫ ИНТЕРФЕЙСОВ (ФАЙЛОВАЯ СТРУКТУРА)
Ниже приведены обязательные к исполнению скелеты классов. Любая модификация публичных сигнатур запрещена.

Контракт Ядра (core.py)
Python


import os, torch
from typing import Tuple, Dict, Final, Any

# Жесткая блокировка детерминизма
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ADCEEngineV33:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        assert config["fixed_point_multiplier"] <= 65536, "Превышен предел M=2^16"
        self.rng_generator = torch.Generator(device="cuda:0" if torch.cuda.is_available() else "cpu")
        self.rng_generator.manual_seed(config.get("seed", 42))
        
        # [Реализовать: Вычисление смещений с выравниванием 16 байт & Аллокация монолита T_global]
        # [Реализовать: Предаллокация T_collision_scratch, T_metrics_scratch, Пинг-Понг буферов IPC]
        
    def _create_soa_views(self) -> Dict[str, torch.Tensor]:
        """Безопасный слайсинг untyped_storage через .narrow().view(dtype)"""
        pass

    def get_api_exchange_handle(self, external_process_id: int) -> Dict[str, Any]:
        """Возврат DIRECT_POINTER или дескрипторов CUDA IPC / POSIX SHM"""
        pass

    def step(self, actions_distribution_indices: torch.Tensor, actions_mental: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Атомарный шаг: Физика (Branchless) -> Хебб -> GA (No-Sort) -> Синхронизация Event"""
        pass
Контракт Визуализатора (renderer.py)
Python


import torch
from typing import Any

class ADCEVisualizer:
    def __init__(self, engine_instance: Any, headless: bool = True):
        self.engine = engine_instance
        self.headless = headless
        # [Реализовать: Предаллокация T_framebuffer [H,W,3], mask_scratch, env_scratch, color_broadcast_scratch]
        
    def render(self) -> torch.Tensor:
        """Сборка кадра Color Mask Blitting БЕЗ рантайм-аллокаций. Только out= и инвертированные маски"""
        pass




        Корневая структура проекта
text

```
/adce-engine-v33/
├── config/                      # Все конфигурации задач и параметров (только здесь описываются задачи)
├── core/                        # Ядро симуляции — immutable contracts
├── modules/                     # Модули 0–4 (Tensor-Engine, Evolution, WM и др.)
├── render/                      # Независимый Render Plane (Blit-View)
├── utils/                       # Низкоуровневые утилиты (fixed-point, memory align, bitpack)
├── tests/                       # Детерминизм, zero-allocation, контракты
├── scripts/                     # Запуски, датасетогенерация, профилирование
├── data/                        # Примеры seed-датасетов и initial conditions (gitignored heavy)
├── outputs/                     # Генерируемые датасеты, видео, checkpoints (gitignored)
├── logs/                        # Логи и метрики (gitignored)
├── .env.example
├── pyproject.toml
├── requirements.txt
├── requirements-cuda.txt
├── README.md
├── LICENSE
├── main.py                      # CLI точка входа
└── __version__.py
```

Детальное описание каждой папки и ключевых файлов
config/ — Только здесь описываются задачи (Task-Agnosticism)

* base.json — глобальные параметры (M=65536, seed, Max_Agents, grid_size, fixed_point_multiplier и т.д.)

* environment.json — Two-Level Block-Sparse Grid, битовая разметка T_environment

* evolution.json — GA параметры, Hebbian (η, γ), мутации, турнир

* task_definition.json — Описание текущей задачи (reward_function, affordance_map, perceiver_tokens, goal_metric и т.д.)

* render.json — framebuffer resolution, render triggers, headless settings

* schema.py — Pydantic-схемы валидации конфигов

* rewards/ — поддиректория с composable reward components

Ответственность: Все адаптации под новую задачу — только редактированием JSON-файлов. Код не меняется.
core/ — Ядро (ADCEEngineV33 контракт)

* __init__.py

* engine.py — ADCEEngineV33 (главный класс). Содержит __init__, step(), get_api_exchange_handle(), _create_soa_views()

* memory.py — Аллокация монолита T_global (torch.uint8), 16-байтное выравнивание, предаллокация всех scratch-буферов

* soa_views.py — Создание безопасных представлений (narrow().view()) для T_environment, T_entities, T_dna_pool и т.д.

* ipc.py — CUDA IPC + POSIX SHM, пинг-понг буферы, Event-синхронизация

* task_adapter.py — Интерпретация task_definition.json → reward calculator + affordance decoder (не меняет T_global напрямую)

* types.py — TypedDict, Final-константы, FixedPoint wrappers

* determinism.py — RNG generator, CUBLAS/CUDNN locks

Ответственность: Гарантия детерминизма и Zero-Allocation на уровне ядра.
modules/ — Модули 0–4

* module0_core_api/

  * exchange_handle.py

  * ipc_manager.py

* module1_tensor_engine/

  * physics.py — Branchless перемещения, коллизии (T_collision_scratch)

  * environment.py — Обновление T_environment (феромоны, ресурсы)

* module2_surrogate_wm/

  * perceiver.py — Task-agnostic токенизация пространства

  * rssm_vit.py — Surrogate World Model + Relative Spatial Attention

  * hdc_memory.py — Гиперкубическая ассоциативная память (HDC-Graph)

* module3_evolution/

  * ga_no_sort.py — Турнирная селекция, кроссовер, мутации (branchless, без torch.sort/topk)

  * hebbian.py — Обновление синапсов по Hebbian rule (clip, fixed-point)

* module4_hippocampus/

  * cognitive_dropout.py — Стокастическая очистка памяти

  * matka_regulator.py — Роевый интеллект Матки + KL-регуляризация

* module5_task_interface/ — Тонкий слой (новый, но соответствует контракту)

  * affordance_decoder.py

  * reward_calculator.py

Ответственность: Каждый модуль работает только через предаллоцированные представления из core. Никаких новых аллокаций в runtime.
render/ — Render Plane (независимый наблюдатель)

* visualizer.py — ADCEVisualizer (полный контракт)

* blit_core.py — Zero-Allocation color blitting (in-place out=, bitwise, маски)

* framebuffer.py — Предаллокация T_framebuffer [H, W, 3] uint8

* backends/

  * local_opengl.py — ModernGL + CUDA-OpenGL interop

  * headless_nvenc.py — NVENC запись в .mp4

* utils.py — Вспомогательные in-place функции для рендера

Ответственность: Только чтение состояний через _create_soa_views(). Никогда не меняет симуляцию.
utils/ — Низкоуровневые примитивы (используются всеми модулями)

* fixed_point.py — Все операции с M=2¹⁶, SymLog, clip, масштабирование

* memory_align.py — 16-байт alignment, scratch buffer management

* bitpack.py — Битовые операции над T_environment (феромоны, flags)

* metrics.py — Fitness tracking, KL-дивергенция Матки

* tensor_ops.py — Wrapper'ы для in-place операций (bitwise_and с out= и т.д.)

tests/

* test_determinism.py

* test_zero_allocation.py — Проверка отсутствия скрытых malloc

* test_soa.py

* test_contracts.py

* test_task_agnostic.py

scripts/

* run_sim.py — Основной запуск

* generate_dataset.py — Экспорт чистых датасетов (obs → action → reward → next)

* profile_memory.py

* validate_contracts.py

Корневые файлы

* main.py — CLI (запуск с конфигом, режимы: dataset / interactive / benchmark)

* __version__.py — "3.3.0"

* pyproject.toml / requirements*.txt

Ключевые инварианты, закреплённые структурой

1. T_global создаётся один раз в core/memory.py.

2. Все step() / render() — только in-place и предаллоцированные буферы.

3. Внешняя нейросеть подключается только через engine.get_api_exchange_handle() (пинг-понг + IPC).

4. Новая задача = новый task_definition.json + возможно новый reward component. Код агентов/Матки не трогается.

5. Датасет — чистый, фиксированный формат, генерируется в outputs/ через scripts/generate_dataset.py.