import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x = np.asarray(x)
    sin3x = np.sin(3 * x)
    tan2x = np.tan(2 * x)
    under_sqrt = sin3x * (tan2x + 2)
    under_sqrt = np.maximum(under_sqrt, 0)
    return -1.5 + np.sqrt(under_sqrt)

def f_prime_analytic(x):
    x = np.asarray(x)
    sin3x = np.sin(3 * x)
    cos3x = np.cos(3 * x)
    tan2x = np.tan(2 * x)
    sec2_2x = 1 / (np.cos(2 * x)**2)
    numerator = 3 * cos3x * (tan2x + 2) + sin3x * (2 * sec2_2x)
    under_sqrt = sin3x * (tan2x + 2)
    under_sqrt = np.maximum(under_sqrt, 0)
    denominator = 2 * np.sqrt(under_sqrt)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        if np.isscalar(x):
            if np.isclose(x, 0):
                return 0.0
        else:
            result = np.where(np.isclose(x, 0), 0.0, result)
    return result

def finite_diff_derivative(x_values, y_values, x_star):
    h = x_values[1] - x_values[0]
    if not np.allclose(np.diff(x_values), h):
        raise ValueError("Точки должны быть равномерными!")
    y = y_values
    dy = y[1] - y[0]
    d2y = y[2] - 2*y[1] + y[0]
    d3y = y[3] - 3*y[2] + 3*y[1] - y[0]
    d4y = y[4] - 4*y[3] + 6*y[2] - 4*y[1] + y[0]
    q = (x_star - x_values[0]) / h
    derivative = (dy + (2*q - 1)/2 * d2y + (3*q**2 - 6*q + 2)/6 * d3y + (4*q**3 - 18*q**2 + 22*q - 6)/24 * d4y) / h
    return derivative

a, b = 0.0, 0.5
n_nodes = 5
h = (b - a) / (n_nodes - 1)
x_nodes = np.linspace(a, b, n_nodes)
y_nodes = f(x_nodes)
x_fine = np.linspace(a, b, 500)
y_fine = f(x_fine)
y_prime_true = f_prime_analytic(x_fine)
y_prime_fd = np.zeros_like(x_fine)

for i, x_star in enumerate(x_fine):
    try:
        y_prime_fd[i] = finite_diff_derivative(x_nodes, y_nodes, x_star)
    except ValueError:
        y_prime_fd[i] = np.nan

y_prime_at_nodes = np.zeros(n_nodes)
for i in range(n_nodes):
    y_prime_at_nodes[i] = finite_diff_derivative(x_nodes, y_nodes, x_nodes[i])

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x_fine, y_fine, 'b-', label='$f(x)$', linewidth=2)
plt.plot(x_nodes, y_nodes, 'ro', label='Узлы (5 точек)', markersize=8, markeredgecolor='darkred')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Исходная функция и узлы интерполяции')
plt.legend()
plt.xlim([a, b])

plt.subplot(2, 1, 2)
plt.plot(x_fine, y_prime_true, 'g-', label='Аналитическая производная', linewidth=2)
plt.plot(x_fine, y_prime_fd, 'm--', label='Конечно-разностная формула (до Δ⁴y)', linewidth=2)
plt.plot(x_nodes, y_prime_at_nodes, 'ko', label='Значения в узлах', markersize=6, markerfacecolor='none')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Сравнение производных')
plt.legend()
plt.xlim([a, b])

plt.tight_layout()
plt.show()

print("=" * 60)
print("Значения в узлах и производные")
print("=" * 60)
print(f"{'i':<3} {'x_i':<10} {'f(x_i)':<15} {'f\'(x_i) FD':<15} {'f\'(x_i) analytic':<15}")
print("-" * 60)
for i in range(n_nodes):
    print(f"{i:<3} {x_nodes[i]:<10.6f} {y_nodes[i]:<15.6f} {y_prime_at_nodes[i]:<15.6f} {f_prime_analytic(x_nodes[i]):<15.6f}")
print("=" * 60)

error = np.abs(y_prime_true - y_prime_fd)
print(f"\nМаксимальная абсолютная погрешность на всем отрезке: {np.max(error):.6f}")
print(f"Средняя абсолютная погрешность: {np.mean(error):.6f}")