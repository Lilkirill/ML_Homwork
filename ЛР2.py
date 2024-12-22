import pygmo as pg
import numpy as np


def func_1(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

def func_2(x):
    return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6) / 6 + x[0] * x[1] + x[1]**2


class CustomProblem:
    def __init__(self, func, dim):
        self.func = func
        self.dim = dim

    def fitness(self, x):
        return [self.func(x)]

    def get_bounds(self):
        return ([-5] * self.dim, [5] * self.dim)


func_1_problem = pg.problem(CustomProblem(func_1, 2))
func_2_problem = pg.problem(CustomProblem(func_2, 2))


algorithms = {
    "Differential Evolution": pg.algorithm(pg.de(gen=100)),
    "Simple Genetic Algorithm": pg.algorithm(pg.sga(gen=100)),
    "Artificial Bee Colony": pg.algorithm(pg.bee_colony(gen=100))
}


def optimize(problem, algorithms):
    results = {}
    for name, algo in algorithms.items():

        pop = pg.population(problem, size=20)  # Создание начальной популяции
        pop = algo.evolve(pop)  # Выполнение оптимизации
        results[name] = (pop.champion_x, pop.champion_f[0])  # Лучшее решение и значение функции
    return results


func_1_results = optimize(func_1_problem, algorithms)

func_2_results = optimize(func_2_problem, algorithms)



def print_results(problem_name, results):
    print(f"\nРезультаты для тестовой функции {problem_name}:")
    print(f"{'Алгоритм':<30} | {'Точка оптимума':<30} | {'Минимум функции':<15}")
    print("-" * 80)
    for algo, (x, f) in results.items():
        print(f"{algo:<30} | {np.array_str(np.array(x), precision=5):<30} | {f:<15.5f}")

print_results("1", func_1_results)
print_results("2", func_2_results)