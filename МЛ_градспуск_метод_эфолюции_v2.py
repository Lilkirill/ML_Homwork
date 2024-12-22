import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sympy as sp


x_sym, y_sym = sp.symbols('x y')
func_sym = 2 * x_sym**2 - 1.05 * x_sym**4 + (x_sym**6) / 6 + x_sym * y_sym + y_sym**2
grad_func_sym = [sp.diff(func_sym, var) for var in (x_sym, y_sym)]


grad_func = sp.lambdify((x_sym, y_sym), grad_func_sym, 'numpy')

func = sp.lambdify((x_sym, y_sym), func_sym, 'numpy')

# Параметры алгоритма
eta = 0.01  
beta_1 = 0.9  
beta_2 = 0.999  
llambda = 0.001  
epsilon = 1e-8 
iterations = 1000 


x, y = 2.5, 4.5  # Начальная точка
m = np.zeros(2) 
v = np.zeros(2)  
trajectory = [(x, y, func(x, y))]  

for t in range(1, iterations + 1):
    grad = np.array(grad_func(x, y))  # Вычисление градиента

   
    m = beta_1 * m + (1 - beta_1) * grad
    v = beta_2 * v + (1 - beta_2) * (grad**2)
    m_hat = m / (1 - beta_1**t)
    v_hat = v / (1 - beta_2**t)
    r = m_hat / (np.sqrt(v_hat) + epsilon)
    lr = eta * np.linalg.norm([x, y]) / (np.linalg.norm(r + llambda * np.array([x, y])) + epsilon)
    x -= lr * (r[0] + llambda * x)
    y -= lr * (r[1] + llambda * y)
    trajectory.append((x, y, func(x, y)))
error = np.linalg.norm([x, y])

# Вывод результатов
print(f"Оптимальные значения: x = {x:.4f}, y = {y:.4f}")
print(f"Значение функции в точке минимума: f(x, y) = {func(x, y):.4f}")
print(f"Погрешность: {error:.4e}")

x_vals = [point[0] for point in trajectory]
y_vals = [point[1] for point in trajectory]
z_vals = [point[2] for point in trajectory]

# Построение графика траектории
fig, ax = plt.subplots()
ax.plot(range(len(z_vals)), z_vals, label='Значение f(x, y)') 
ax.set_xlabel('Итерация')  
ax.set_ylabel('Значение f(x, y)')  
ax.set_title('Сходимость градиентного спуска')  
plt.legend()  
plt.grid(True)  
plt.show()

# 3D-график функции и траектории
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

x_grid = np.linspace(-10, 10, 100)
y_grid = np.linspace(-10, 10, 100)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
z_mesh = func(x_mesh, y_mesh)
ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.7, cmap='viridis', edgecolor='none')
trajectory = np.array(trajectory)
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', label='Траектория')
ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='red')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('3D график функции и траектория оптимизации')
ax.legend()
plt.show()

from matplotlib.animation import FuncAnimation

def update(num):
    ax.clear()
    ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.7, cmap='viridis', edgecolor='none')
    ax.plot(trajectory[:num, 0], trajectory[:num, 1], trajectory[:num, 2], 'r-')
    ax.scatter(trajectory[num, 0], trajectory[num, 1], trajectory[num, 2], c='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(f'Итерация {num}')

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ani = FuncAnimation(fig, update, frames=len(trajectory), interval=50)
plt.show()