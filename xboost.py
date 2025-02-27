import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Технические индикаторы (без изменений)
def calculate_ema(df, window):
    return df['close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(df, window):
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(df, window):
    low_min = df['low'].rolling(window=window, min_periods=1).min()
    high_max = df['high'].rolling(window=window, min_periods=1).max()
    K = 100 * ((df['close'] - low_min) / (high_max - low_min))
    return K

# Улучшение признаков
def create_features(df):
    df['ema_20'] = calculate_ema(df, 20)
    df['ema_50'] = calculate_ema(df, 50)
    df['rsi_14'] = calculate_rsi(df, 14)
    df['stochastic_14'] = calculate_stochastic(df, 14)
    
    # Скользящие окна и лаги
    df['close_mean_5'] = df['close'].rolling(window=5).mean()
    df['volume_mean_5'] = df['volume'].rolling(window=5).mean()
    
    # Lagged Features
    df['close_lag_1'] = df['close'].shift(1)
    df['volume_lag_1'] = df['volume'].shift(1)

    # Заполнение nan значений нулями
    df.fillna(0, inplace=True)
    return df

# Целевая переменная
def create_target(df):
    df['open'] = df['close'].shift(1)
    df.dropna(inplace=True)
    df['price_change'] = (df['close'] - df['open']) / df['open']
    
    # Отладочная информация
    print("Первые 5 значений price_change:", df['price_change'].head())
    
    # Изменение порогов и значений целевой переменной
    df['Target'] = df['price_change'].apply(lambda x: 1 if x >= 0.0004 else (0 if x <= -0.0004 else None))
    
    # Отладочная информация
    print("Первые 5 значений Target:", df['Target'].head())
    
    df.dropna(subset=['Target'], inplace=True)
    return df

# Загрузка данных
try:
    df = pd.read_excel('f_06267a02d9b066aa.xlsx')
    print("Датасет загружен")
except FileNotFoundError:
    print("Ошибка: Файл f_06267a02d9b066aa не найден.")
    exit()

print(f"Размер данных: {df.shape}")

df = create_target(df)

# Проверка наличия колонки 'Magic'
if 'Magic' in df.columns:
    print("Колонка 'Magic' присутствует в данных")
else:
    print("Колонка 'Magic' отсутствует в данных")

# Параметры
train_size = 150000
test_size = 10000
min_gap = 3000

# Основной цикл обучения и предсказания
all_accuracy_scores = []

for i in range(0, len(df) - train_size - test_size + 1, test_size):
    start_train = i
    end_train = i + train_size
    start_test = end_train + min_gap
    end_test = start_test + test_size

    if end_test > len(df):
        break
    
    train_data = df.iloc[start_train:end_train].copy()
    test_data = df.iloc[start_test:end_test].copy()

    # Проверяем размеры train и test данных
    print(f"Количество строк в train: {train_data.shape[0]}")
    print(f"Количество строк в test: {test_data.shape[0]}")

    # Проверка наличия целевой переменной
    if 'Target' not in train_data.columns or 'Target' not in test_data.columns:
        print("Ошибка: Целевая переменная 'Target' отсутствует в данных.")
        continue

    # Проверка на наличие значений в целевой переменной
    print(f"Целевая переменная в train: {train_data['Target'].value_counts()}")
    print(f"Целевая переменная в test: {test_data['Target'].value_counts()}")

    train_data = create_features(train_data)
    test_data = create_features(test_data)

    # Проверка на наличие признаков
    if train_data.empty or test_data.empty:
        print("Ошибка: train_data или test_data пустые после создания признаков.")
        continue

    # Исключаем колонку Target и 'Magic' (если она присутствует)
    features = [col for col in train_data.columns if col not in ['Target', 'open', 'Magic']]  # Исключаем Magic
    print(f"Признаки для обучения: {features}")

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[features])
    X_test = scaler.transform(test_data[features])
    y_train = train_data['Target']
    y_test = test_data['Target']

    # Проверка на наличие данных для обучения
    if len(y_train) == 0 or len(y_test) == 0:
        print("Ошибка: Нет данных для обучения или тестирования.")
        continue

    # Вывод размеров данных для отладки
    print(f"Размеры данных для обучения: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Размеры данных для тестирования: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # XGBoost Model
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    
    # Оптимизация гиперпараметров с GridSearchCV
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    print("Обучение модели...")
    grid_search.fit(X_train, y_train)
    print(f"Лучшие параметры модели: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Predictions made, accuracy_score: {accuracy}")
    
    # Добавляем accuracy_score в список
    all_accuracy_scores.append(accuracy)

    print(f"Iteration {i // test_size + 1}, accuracy_score: {accuracy}")

# Анализ результатов
min_accuracy = np.min(all_accuracy_scores) if all_accuracy_scores else None
max_accuracy = np.max(all_accuracy_scores) if all_accuracy_scores else None
average_accuracy = np.mean(all_accuracy_scores) if all_accuracy_scores else None
below_60_percent = (sum(a < 0.6 for a in all_accuracy_scores) / len(all_accuracy_scores)) * 100 if all_accuracy_scores else None

print("-" * 40)
print(f"Минимальный accuracy_score: {min_accuracy}")
print(f"Максимальный accuracy_score: {max_accuracy}")
print(f"Средний accuracy_score: {average_accuracy}")
print(f"Процент accuracy_score ниже 0.6: {below_60_percent}")
print("-" * 40)

# Проверка критериев
if (min_accuracy is not None and 
    below_60_percent is not None and 
    average_accuracy is not None and
    min_accuracy >= 0.57 and 
    below_60_percent <= 20 and 
    average_accuracy >= 0.64 and
    all_accuracy_scores.count(0) <= 2):
    print("Критерии пройдены.")
else:
    print("Критерии не пройдены.")
