import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sympy import symbols, diff, lambdify

# Определим функцию
x_sym, y_sym = symbols('x y')
f_sym = 2 * x_sym**2 - 1.05 * x_sym**4 + (x_sym**6) / 6 + x_sym * y_sym + y_sym**2

# Вычисление градиента с помощью sympy
grad_x_sym = diff(f_sym, x_sym) 
grad_y_sym = diff(f_sym, y_sym) 
grad_x_func = lambdify((x_sym, y_sym), grad_x_sym)
grad_y_func = lambdify((x_sym, y_sym), grad_y_sym)
f = lambdify((x_sym, y_sym), f_sym)

# Реализация адаптивного градиентного спуска
def gradient_descent(start_x, start_y, learning_rate=0.01, epsilon=1e-8, iterations=1000):
   
    x, y = start_x, start_y
    G_x, G_y = 0, 0 
    history = [(x, y, f(x, y))]

    for i in range(iterations):
        # Вычисляем градиент
        grad_x = grad_x_func(x, y)
        grad_y = grad_y_func(x, y)

      
        G_x += grad_x**2
        G_y += grad_y**2

        # Обновляем x и y с учетом адаптивного шага обучения
        x -= (learning_rate / (np.sqrt(G_x) + epsilon)) * grad_x
        y -= (learning_rate / (np.sqrt(G_y) + epsilon)) * grad_y
        history.append((x, y, f(x, y)))

    return x, y, history

# Параметры начальной точки и алгоритма
start_x, start_y = 2.5, 4.5  
learning_rate = 0.2  
iterations = 500  

# Запуск градиентного спуска
x_opt, y_opt, history = gradient_descent(start_x, start_y, learning_rate, iterations=iterations)

# Вычисление погрешности
error = np.sqrt((x_opt - 0)**2 + (y_opt - 0)**2)

# Вывод результатов
print(f"Оптимальные значения: x = {x_opt:.4f}, y = {y_opt:.4f}") 
print(f"Значение функции в точке минимума: f(x, y) = {f(x_opt, y_opt):.4f}")  
print(f"Погрешность относительно точки (0, 0): {error:.4f}")  

x_vals = [point[0] for point in history]  
y_vals = [point[1] for point in history]  
z_vals = [point[2] for point in history]  


fig, ax = plt.subplots()
ax.plot(range(len(z_vals)), z_vals, label='Значение f(x, y)')  
ax.set_xlabel('Итерация')  
ax.set_ylabel('Значение f(x, y)') 
ax.set_title('Сходимость градиентного спуска')  
plt.legend()  
plt.grid(True)
plt.show()

# Визуализация траектории 
plt.figure(figsize=(8, 6))  
plt.plot(x_vals, y_vals, 'o-', label='Траектория', markersize=4) 
plt.xlabel('x')  
plt.ylabel('y')  
plt.title('Траектория градиентного спуска') 
plt.grid(True)
plt.legend()  
plt.show()

# Отображение 3D-графика 
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
z_mesh = f(x_mesh, y_mesh)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis', alpha=0.8)

trajectory, = ax.plot([], [], [], 'r-', label='Траектория') 
point, = ax.plot([], [], [], 'bo', markersize=8, label='Текущая точка') 

ax.set_xlabel('x')  
ax.set_ylabel('y') 
ax.set_zlabel('f(x, y)')
ax.set_title('3D график f(x, y) и траектория градиентного спуска') 
plt.legend()

def update(num):
    trajectory.set_data(x_vals[:num], y_vals[:num])
    trajectory.set_3d_properties(z_vals[:num])
    point.set_data([x_vals[num - 1]], [y_vals[num - 1]])
    point.set_3d_properties([z_vals[num - 1]])

ani = FuncAnimation(fig, update, frames=len(x_vals), interval=100, repeat=False)
plt.show()
