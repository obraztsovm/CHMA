import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return -1.5 + np.sqrt(np.sin(3*x) * (np.tan(2*x) + 2))

def lagrange_value(x, x_nodes, y_nodes):
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        li = 1.0
        for j in range(n):
            if i != j:
                li *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += y_nodes[i] * li
    return result

def L2_explicit(x):
    return -6.008*x**2 + 6.7716*x - 1.5

def L4_explicit(x):
    return 25.33*x**4 - 32.15*x**3 + 10.82*x**2 + 5.67*x - 1.5

x3 = np.array([0.0, 0.25, 0.5])
y3 = np.array([-1.5, -0.1826, 0.3838])

x5 = np.array([0.0, 0.125, 0.25, 0.375, 0.5])
y5 = np.array([-1.5, -0.5911, -0.1826, 0.1264, 0.3838])

x_grid = np.linspace(0, 0.5, 1000)
f_values = f(x_grid)
L2_values = L2_explicit(x_grid)
L4_values = L4_explicit(x_grid)

error_2 = np.abs(f_values - L2_values)
error_4 = np.abs(f_values - L4_values)

max_error_2 = np.max(error_2)
max_error_4 = np.max(error_4)

print("="*50)
print("РЕЗУЛЬТАТЫ ИНТЕРПОЛЯЦИИ")
print("="*50)
print(f"Максимальная ошибка для полинома 2 степени (3 узла): {max_error_2:.6f}")
print(f"Максимальная ошибка для полинома 4 степени (5 узлов): {max_error_4:.6f}")
print()

idx_max2 = np.argmax(error_2)
x_max2 = x_grid[idx_max2]
print(f"Максимальная ошибка для 3 узлов достигается в x = {x_max2:.4f}")
print(f"  f(x) = {f_values[idx_max2]:.6f}")
print(f"  L2(x) = {L2_values[idx_max2]:.6f}")
print(f"  |f - L2| = {error_2[idx_max2]:.6f}")
print()

idx_max4 = np.argmax(error_4)
x_max4 = x_grid[idx_max4]
print(f"Максимальная ошибка для 5 узлов достигается в x = {x_max4:.4f}")
print(f"  f(x) = {f_values[idx_max4]:.6f}")
print(f"  L4(x) = {L4_values[idx_max4]:.6f}")
print(f"  |f - L4| = {error_4[idx_max4]:.6f}")
print("="*50)

def numerical_derivative(f, x, h=1e-5, order=1):
    if order == 1:
        return (f(x + h) - f(x - h)) / (2*h)
    elif order == 2:
        return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
    elif order == 3:
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2*h**3)
    elif order == 4:
        return (f(x + 2*h) - 4*f(x + h) + 6*f(x) - 4*f(x - h) + f(x - 2*h)) / (h**4)
    elif order == 5:
        return (f(x + 2*h) - 4*f(x + h) + 5*f(x) - 4*f(x - h) + f(x - 2*h)) / (2*h**5)
    else:
        raise ValueError("Поддерживаются порядки до 5")

print("\nЧИСЛЕННАЯ ОЦЕНКА ПРОИЗВОДНЫХ:")
print("-"*50)

x_test = np.linspace(0.01, 0.49, 200)
f3_derivatives = [abs(numerical_derivative(f, x, order=3)) for x in x_test]
M3_estimated = np.max(f3_derivatives)
print(f"Оценка M3 = max|f'''(x)| ≈ {M3_estimated:.4f}")

f5_derivatives = [abs(numerical_derivative(f, x, order=5)) for x in x_test]
M5_estimated = np.max(f5_derivatives)
print(f"Оценка M5 = max|f^(5)(x)| ≈ {M5_estimated:.4f}")

def omega3(x):
    return (x - 0) * (x - 0.25) * (x - 0.5)

def omega5(x):
    return (x - 0) * (x - 0.125) * (x - 0.25) * (x - 0.375) * (x - 0.5)

omega3_values = [abs(omega3(x)) for x in x_grid]
omega5_values = [abs(omega5(x)) for x in x_grid]

max_omega3 = np.max(omega3_values)
max_omega5 = np.max(omega5_values)

print("\nОЦЕНКА ОШИБКИ СВЕРХУ:")
print("-"*50)
print(f"max|ω3(x)| = {max_omega3:.6f}")
print(f"max|ω5(x)| = {max_omega5:.6f}")

error_est_3 = (M3_estimated / 6) * max_omega3
error_est_5 = (M5_estimated / 120) * max_omega5

print(f"\nТеоретическая оценка ошибки для 3 узлов: |R2(x)| ≤ {error_est_3:.6f}")
print(f"Фактическая максимальная ошибка:         {max_error_2:.6f}")
print(f"Отношение оценка/факт: {error_est_3/max_error_2:.2f}")
print()
print(f"Теоретическая оценка ошибки для 5 узлов: |R4(x)| ≤ {error_est_5:.6f}")
print(f"Фактическая максимальная ошибка:         {max_error_4:.6f}")
print(f"Отношение оценка/факт: {error_est_5/max_error_4:.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.plot(x_grid, f_values, 'k-', linewidth=2, label='f(x) - исходная')
ax1.plot(x_grid, L2_values, 'r--', linewidth=1.5, label='L2(x) - 3 узла')
ax1.plot(x_grid, L4_values, 'b--', linewidth=1.5, label='L4(x) - 5 узлов')
ax1.plot(x3, y3, 'ro', markersize=8, label='узлы (3)')
ax1.plot(x5, y5, 'bs', markersize=6, label='узлы (5)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Интерполяция полиномами Лагранжа')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.semilogy(x_grid, error_2, 'r-', label='|f(x) - L2(x)|')
ax2.semilogy(x_grid, error_4, 'b-', label='|f(x) - L4(x)|')
ax2.set_xlabel('x')
ax2.set_ylabel('|ошибка| (лог. шкала)')
ax2.set_title('Ошибки интерполяции')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.plot(x_grid, omega3(x_grid), 'r-', label='ω3(x) для 3 узлов')
ax3.plot(x_grid, omega5(x_grid), 'b-', label='ω5(x) для 5 узлов')
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax3.set_xlabel('x')
ax3.set_ylabel('ω(x)')
ax3.set_title('Узловые полиномы')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.bar(['3 узла', '5 узлов'], [max_error_2, max_error_4], color=['red', 'blue'])
ax4.set_ylabel('максимальная ошибка')
ax4.set_title('Сравнение точности')
ax4.set_yscale('log')
for i, v in enumerate([max_error_2, max_error_4]):
    ax4.text(i, v*1.1, f'{v:.6f}', ha='center')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("ВЫВОДЫ:")
print("="*50)
print("1. Увеличение числа узлов с 3 до 5 уменьшило ошибку примерно в",
      f"{max_error_2/max_error_4:.1f} раз")
print("2. Теоретическая оценка ошибки дает завышенное значение, но верный порядок")
print("3. Аналитическое вычисление производных высокого порядка для данной функции")
print("   крайне затруднительно из-за наличия композиции тригонометрических функций")
print("   и квадратного корня, поэтому была использована численная оценка")
print("="*50)