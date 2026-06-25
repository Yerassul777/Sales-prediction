# Walmart Store Sales Prediction — Linear Regression

> **Portfolio Project** | Machine Learning | Linear Regression  
> Курс: Data Science | Преподаватель: Jamil Zhumabek (mental bee)

---

## Problem Statement

Walmart — крупнейший ритейлер мира. Каждую неделю тысячи магазинов по всей Америке
фиксируют выручку, которая зависит от огромного числа факторов.

**Задача:** предсказать нормированные недельные продажи магазина (`Sales_z`)
на основе макроэкономических и операционных факторов:
температуры в регионе, цены на топливо, индекса потребительских цен,
уровня безработицы, наличия праздников и сезонных паттернов.

**Почему это интересная задача?**
Линейная регрессия здесь — не только предсказательная модель,
но и инструмент объяснения: коэффициент при `Holiday_Flag` буквально говорит
«насколько больше продаёт магазин в праздничную неделю». Это прямой ответ
на бизнес-вопрос.

---

## Dataset

| Параметр | Значение |
|---|---|
| **Источник** | [Kaggle — Walmart Store Sales](https://www.kaggle.com/datasets/yasserh/walmart-dataset) |
| **Строк** | 6,435 |
| **Признаков** | 8 |
| **Период** | февраль 2010 — октябрь 2012 |
| **Магазинов** | 45 |

### Описание признаков

| Признак | Тип | Описание |
|---|---|---|
| `Store` | int | Номер магазина (1–45) |
| `Date` | date | Дата начала недели |
| `Weekly_Sales` | float | ⭐ Недельная выручка магазина ($) |
| `Holiday_Flag` | bool | 1 = праздничная неделя (Super Bowl, Thanksgiving, Christmas, Labor Day) |
| `Temperature` | float | Средняя температура в регионе (°F) |
| `Fuel_Price` | float | Цена топлива ($/gallon) |
| `CPI` | float | Consumer Price Index — индекс потребительских цен |
| `Unemployment` | float | Уровень безработицы в регионе (%) |

**Целевая переменная:** `Sales_z` — z-score нормализация `Weekly_Sales`
внутри каждого магазина. Это позволяет модели объяснять *отклонения от нормы*
магазина, а не абсолютные значения (которые сильно отличаются между магазинами типа A и C).

---

## EDA: Ключевые находки

### Распределения признаков
![Distributions](images/1_distributions.png)
- `Weekly_Sales` имеет правостороннее распределение — большинство магазинов продают $1–2M/неделю
- `Temperature` равномерно распределена по сезонам (данные покрывают 2.5 года)
- `Unemployment` сосредоточен в диапазоне 7–9% (пост-кризисный период 2008–2012)
- `Holiday_Flag`: только ~5% строк — праздничные недели. Это реалистично.

### Корреляционная матрица
![Heatmap](images/2_correlation_heatmap.png)
- **Temperature** (r=+0.59) — сильнейшая линейная связь с продажами
- **Holiday_Flag** (r=+0.21) — праздники значимо поднимают продажи
- **Month/Week** (r≈-0.33) — сезонность: начало года = спад

### Scatter: признак vs продажи
![Scatter](images/3_scatter_features_vs_sales.png)

### Сезонный анализ
![Seasonal](images/4_seasonal_analysis.png)
- Праздничные недели дают **+11.7%** к средним продажам
- Декабрь — пиковый месяц (Christmas season)
- Январь — минимум (послепраздничный спад)

---

## Модель

### Архитектура

```
Модель:     Multiple Linear Regression (sklearn.linear_model.LinearRegression)
Признаки:   Temperature, Fuel_Price, CPI, Unemployment, Holiday_Flag, Month, Week
Цель:       Sales_z (z-score нормализованные продажи внутри магазина)
Split:      80% train / 20% test | random_state=42
```

### Уравнение модели

```
Sales_z = -2.11
        + 0.045 × Temperature
        - 0.082 × Fuel_Price
        - 0.001 × CPI
        - 0.007 × Unemployment
        + 0.207 × Holiday_Flag
        + 0.004 × Month
        + 0.006 × Week
```

---

## Результаты

### Метрики

| Метрика | Train | Test | CV-10 mean |
|---|---|---|---|
| **MAE** | 0.560 σ | 0.575 σ | — |
| **RMSE** | 0.721 σ | 0.724 σ | — |
| **R²** | 0.471 | **0.495** | 0.475 ± 0.021 |

> **Выбранная главная метрика: R²**  
> MAE и RMSE нельзя сравнивать напрямую (разные единицы интерпретации).
> R² — сопоставимая метрика: показывает долю объяснённой дисперсии.

### Actual vs Predicted
![Actual vs Predicted](images/5_actual_vs_predicted.png)
- Точки группируются вдоль диагонали: модель работает корректно
- Residuals нормально распределены с центром в нуле: нет систематической ошибки
- Разброс остаётся: часть дисперсии объясняется признаками вне датасета

### Метрики — графический вид
![Metrics](images/7_metrics_summary.png)

---

## Интерпретация коэффициентов

![Coefficients](images/6_coefficients.png)

| Признак | Станд. коэф. | Интерпретация |
|---|---|---|
| `Temperature` | **+0.856** | Главный предиктор. +1σ температуры → продажи на 0.86σ выше нормы. Летний сезон: BBQ, садовые товары, back-to-school |
| `Holiday_Flag` | **+0.302** | Праздник → +0.30σ к продажам. Super Bowl, Thanksgiving, Christmas — ключевые моменты |
| `Week` | +0.206 | Поздние недели года (предрождественский сезон) → рост продаж |
| `Month` | +0.087 | Аналогично Week — сезонный паттерн |
| `Fuel_Price` | -0.027 | Дорогое топливо → люди реже едут в магазин |
| `CPI` | -0.003 | Инфляция незначительно давит на реальные продажи |
| `Unemployment` | -0.002 | Рост безработицы → меньше располагаемых доходов |

---

## Ограничения модели

1. **Линейность**: модель предполагает линейные зависимости. В ритейле много нелинейных взаимодействий (`Holiday × Temperature`, `Month²`)
2. **Нет store-level fixed effects**: z-score нормализация частично решает проблему, но не полностью — тип магазина (A/B/C) не в модели
3. **Нет данных по отделам**: Groceries и Electronics ведут себя совершенно по-разному
4. **Временной период**: данные 2010–2012, паттерны потребления с тех пор изменились (e-commerce, COVID и т.д.)
5. **Нет данных о конкурентах**: наличие Target или Costco рядом напрямую влияет на продажи

---

## Стек

```
Python 3.12
pandas       — загрузка и обработка данных
numpy        — математические вычисления
matplotlib   — визуализации
seaborn      — statistical plots
scikit-learn — LinearRegression, train_test_split, cross_val_score, метрики
```

---

## Структура проекта

```
walmart-sales-prediction/
├── data/
│   └── Walmart_Store_Sales.csv       # Датасет (6,435 строк)
├── notebooks/
│   └── walmart_sales_linear_regression.ipynb  # Основной ноутбук
├── images/
│   ├── 1_distributions.png           # Распределения признаков
│   ├── 2_correlation_heatmap.png     # Корреляционная матрица
│   ├── 3_scatter_features_vs_sales.png  # Scatter plots
│   ├── 4_seasonal_analysis.png       # Праздники и сезонность
│   ├── 5_actual_vs_predicted.png     # Actual vs Predicted + Residuals
│   ├── 6_coefficients.png            # Интерпретация коэффициентов
│   └── 7_metrics_summary.png         # Сводные метрики
└── README.md
```

---

## Как запустить

```bash
git clone https://github.com/<your-username>/walmart-sales-prediction.git
cd walmart-sales-prediction
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook notebooks/walmart_sales_linear_regression.ipynb
```

---

## Источники

- Dataset: [Kaggle — Walmart Store Sales](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
- Академическое обоснование linear regression на Walmart: [ResearchGate (2024)](https://www.researchgate.net/publication/383210355_Walmart_Sales_Prediction_Using_Multiple_Linear_Regression)
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (2009)

---

*Проект выполнен в рамках курса Data Science. Цель — не скопировать код, а понять весь путь модели: от данных до интерпретируемых выводов.*
