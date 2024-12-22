import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def f(x, y):
    return (x + 2 * y - 7)**2 + (2 * x + y - 5)**2

# Ввод частных производных 
def input_gradient():
    print("Введите частные производные функции по переменным x и y.")
    df_dx_input = input("Частная производная по x (например, '10 * x + 8 * y - 34'): ")
    df_dy_input = input("Частная производная по y (например, '8 * x + 10 * y - 38'): ")

    def df_dx(x, y):
        return eval(df_dx_input)

    def df_dy(x, y):
        return eval(df_dy_input)

    return df_dx, df_dy

df_dx, df_dy = input_gradient()

# Реализация градиентного спуска
def gradient_descent(start_x, start_y, learning_rate=0.01, iterations=1000):
 
    x, y = start_x, start_y
    history = [(x, y, f(x, y))]

    for i in range(iterations):
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)

        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

        history.append((x, y, f(x, y)))

    return x, y, history

# Параметры начальной точки и алгоритма
start_x, start_y = 2.5, 4.5  
learning_rate = 0.01  # Шаг обучения
iterations = 500  # Количество итераций

# Запуск градиентного спуска
x_opt, y_opt, history = gradient_descent(start_x, start_y, learning_rate, iterations)

# Вычисление погрешности
error = np.sqrt((x_opt - 0)**2 + (y_opt - 0)**2)

# Вывод результатов
print(f"Оптимальные значения: x = {x_opt:.4f}, y = {y_opt:.4f}")  
print(f"Значение функции в точке минимума: f(x, y) = {f(x_opt, y_opt):.4f}")  
print(f"Погрешность относительно точки (0, 0): {error:.4f}")  

x_vals = [point[0] for point in history]  
y_vals = [point[1] for point in history]  
z_vals = [point[2] for point in history] 

# Построение графика траектории
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

# Добавление анимации
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
