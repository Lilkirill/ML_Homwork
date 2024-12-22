import optuna
import json
from sklearn.datasets import load_wine, fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

# Настройка подключения к PostgreSQL
storage = "postgresql+psycopg2://postgres:1707@192.168.1.6/optuna_db"

# Данные для классификации (Wine dataset)
data = load_wine()
X, y = data.data, data.target

# Данные для регрессии (California Housing dataset через OpenML)
boston = fetch_openml(name="house_prices", version=1, as_frame=False)
X_reg = boston.data[:, :-1]  # Исключаем целевую переменную
y_reg = boston.data[:, -1]  # Целевая переменная

# Оптимизация для классификации
def objective_classification(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 50)  # Уменьшен диапазон
    max_depth = trial.suggest_int("max_depth", 2, 16, log=True)  # Уменьшен диапазон
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    return cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()

# Оптимизация для регрессии
def objective_regression(trial):
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)  # Уменьшен диапазон
    model = SVR(C=C)

    return cross_val_score(model, X_reg, y_reg, cv=3, scoring="neg_mean_squared_error").mean()

# Оптимизация классификации
study_classification = optuna.create_study(
    study_name="classification_optimization",
    storage=storage,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(),
    load_if_exists=True,
)
study_classification.optimize(objective_classification, n_trials=30)  # Уменьшено число итераций

# Сохранение лучших параметров классификации
best_params_classification = study_classification.best_params
with open("best_params_classification.json", "w") as f:
    json.dump(best_params_classification, f)

# Оптимизация регрессии
study_regression = optuna.create_study(
    study_name="regression_optimization",
    storage=storage,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(),
    load_if_exists=True,
)
study_regression.optimize(objective_regression, n_trials=30)  # Уменьшено число итераций

# Сохранение лучших параметров регрессии
best_params_regression = study_regression.best_params
with open("best_params_regression.json", "w") as f:
    json.dump(best_params_regression, f)

# Графики
optuna.visualization.plot_optimization_history(study_classification).show()
optuna.visualization.plot_optimization_history(study_regression).show()
optuna.visualization.plot_param_importances(study_classification).show()
optuna.visualization.plot_param_importances(study_regression).show()
