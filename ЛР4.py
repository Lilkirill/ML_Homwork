import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from scipy.stats import entropy

# Генерация случайных данных
np.random.seed(42)  # Фиксируем случайность для воспроизводимости
# Объединяем два нормальных распределения с разными центрами и масштабами
data = np.concatenate([
    np.random.normal(loc=-3, scale=1, size=300),  # Первое распределение
    np.random.normal(loc=3, scale=1, size=300)   # Второе распределение
])[:, None]  # Добавляем размерность для работы с sklearn

# Метод KDE (Kernel Density Estimation)
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)  # Оценка плотности ядром Гаусса
x_grid = np.linspace(-7, 7, 1000)[:, None]  # Создаем сетку для оценки плотности
kde_density = np.exp(kde.score_samples(x_grid))  # Вычисляем плотность на сетке

# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, random_state=42).fit(data)  # Подгоняем GMM с двумя компонентами
gmm_density = np.exp(gmm.score_samples(x_grid))  # Вычисляем плотность на сетке

# Визуализация данных и оценок плотности
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Гистограмма данных')  # Гистограмма данных
plt.plot(x_grid, kde_density, label='KDE', color='blue')  # Плотность KDE
plt.plot(x_grid, gmm_density, label='GMM (EM)', color='red')  # Плотность GMM
plt.legend()
plt.title("Восстановление плотности")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.show()

# Реализация алгоритма Метрополиса-Гастингса
def metropolis_hastings(p, n_samples, x0):

  #  Алгоритм семплирования методом Метрополиса-Гастингса
    #p: целевая плотность
   # n_samples: количество семплов
  #  x0: начальная точка

    samples = [x0]  # Список семплов, начинаем с x0
    for _ in range(n_samples - 1):
        x_current = samples[-1]  # Текущая точка
        x_proposal = x_current + np.random.normal(0, 1)  # Генерируем предложение
        # Вычисляем отношение вероятностей и принимаем/отклоняем предложение
        acceptance_ratio = p(x_proposal) / p(x_current)
        if np.random.rand() < min(1, acceptance_ratio):
            samples.append(x_proposal)  # Принимаем предложение
        else:
            samples.append(x_current)  # Оставляем текущую точку
    return np.array(samples)

# Реализация алгоритма Гиббса
def gibbs_sampling(kde, n_samples, x0, y0):

   # Алгоритм семплирования методом Гиббса
   # kde: объект KernelDensity для оценки плотности
   # n_samples: количество семплов
   # x0, y0: начальные значения x и y

    samples = [(x0, y0)]  # Список семплов, начинаем с (x0, y0)
    for _ in range(n_samples - 1):
        x_current, y_current = samples[-1]  # Текущая пара точек
        x_next = np.random.normal(loc=y_current, scale=1)  # Генерируем новое x
        y_next = np.random.normal(loc=x_next, scale=1)  # Генерируем новое y
        samples.append((x_next, y_next))  # Добавляем пару в семплы
    return np.array(samples)

# Использование KDE-плотности для Метрополиса-Гастингса и Гиббса
def p_density(x):
    return np.exp(kde.score_samples(np.array([[x]])))[0]

# Семплирование методом Метрополиса-Гастингса
mh_samples = metropolis_hastings(p_density, n_samples=1000, x0=0)

# Семплирование методом Гиббса
gibbs_samples = gibbs_sampling(kde, n_samples=1000, x0=0, y0=0)

# Визуализация исходных данных и семплов Метрополиса-Гастингса
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Исходные данные')  # Гистограмма данных
plt.hist(mh_samples, bins=30, density=True, alpha=0.5, color='blue', label='MH Семплы')  # Гистограмма MH семплов
plt.legend()
plt.title("Исходные данные vs. Семплы Метрополиса-Гастингса")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Исходные данные')
plt.hist(mh_samples, bins=30, density=True, alpha=0.5, color='blue', label='MH Семплы')
plt.hist(gibbs_samples[:, 0], bins=30, density=True, alpha=0.5, color='green', label='Gibbs Семплы')
plt.legend()
plt.title("Сравнение данных: исходные vs. MH vs. Gibbs")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.show()



