# Cryptocurrency-Price-Prediction-with-XGBoost

Этот репозиторий содержит pipeline машинного обучения для прогнозирования изменения цен на криптовалюту с использованием различных технических индикаторов и модели XGBoost.

Обзор
Программа выполняет следующие шаги:

Загрузка данных:

Загружает данные о криптовалюте из Excel файла.
Инженерия признаков:

Рассчитывает технические индикаторы:
Экспоненциальную скользящую среднюю (EMA) для периодов 20 и 50
Индекс относительной силы (RSI) для 14-дневного периода
Стохастический осциллятор для 14-дневного периода
Добавляет дополнительные признаки:
5-дневные скользящие средние по цене закрытия и объему торгов
Лагированные признаки (цена и объем за предыдущий день)
Целевая переменная:

Целевая переменная определяется как изменение цены между двумя последовательными днями. Если изменение цены больше или равно 0.04%, то это считается положительным изменением (целевое значение = 1). Если изменение меньше или равно -0.04%, это считается отрицательным изменением (целевое значение = 0). Промежуточные значения игнорируются.
Обучение модели и прогнозирование:

Данные делятся на обучающую и тестовую выборки.
Используется классификатор XGBoost для обучения модели, с применением GridSearchCV для оптимизации гиперпараметров.
Модель предсказывает изменения цен на основе созданных признаков.
Оценка модели:

Оценивается точность модели на тестовой выборке, и точности записываются для каждой итерации.
Выводятся статистики по точностям: минимальная, максимальная, средняя точность, а также процент точностей ниже 60%.
Критерии успеха:

Программа проверяет, выполнены ли определенные пороговые значения точности:
Минимальная точность должна быть не ниже 0.57
Процент итераций с точностью ниже 60% должен составлять не более 20%
Средняя точность должна быть выше 0.64
Количество итераций с точностью равной 0 должно быть не более 2
Требования
pandas
numpy
sklearn
xgboost
