import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

data = pd.read_csv('../dataset/cars93.csv')
data = data[['Price', 'MPG.city', 'MPG.highway', 'Horsepower']]
data = data.dropna()

# 1 – ПОСТРОЕНИЕ ЛИНЕЙНОЙ МОДЕЛИ
# создаем матрицу признаков
X = data[['MPG.city', 'MPG.highway', 'Horsepower']].values
X = np.hstack((np.ones((X.shape[0], 1)), X))
y = data['Price'].values
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
y_pred = X @ beta_hat
residuals = y - y_pred
n, p = X.shape
sigma_squared = (residuals.T @ residuals) / (n - p)
cov_beta = sigma_squared * np.linalg.inv(X.T @ X)
std_err_beta = np.sqrt(np.diag(cov_beta))
alpha = 0.05
t_value = t.ppf(1 - alpha / 2, df=n - p)
conf_intervals = np.array([beta_hat - t_value * std_err_beta, beta_hat + t_value * std_err_beta]).T
SS_total = np.sum((y - np.mean(y)) ** 2)
SS_residual = np.sum(residuals ** 2)
R_squared = 1 - (SS_residual / SS_total)

plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
plt.scatter(y, y_pred, edgecolor='k', alpha=0.7)
plt.plot([min(y), max(y)], [min(y), max(y)], '--r')
plt.xlabel('Фактические значения цены')
plt.ylabel('Предсказанные значения цены')
plt.title('Фактические vs Предсказанные значения')

plt.subplot(2, 2, 2)
plt.scatter(y_pred, residuals, edgecolor='k', alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения цены')
plt.ylabel('Остатки')
plt.title('График остатков')

plt.subplot(2, 2, 3)
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Остатки')
plt.ylabel('Частота')
plt.title('Гистограмма остатков')

plt.subplot(2, 2, 4)
plt.scatter(data['Horsepower'], data['Price'], edgecolor='k', alpha=0.7)
line_x = np.linspace(min(data['Horsepower']), max(data['Horsepower']), 100)
line_y = beta_hat[0] + beta_hat[3] * line_x
plt.plot(line_x, line_y, '--r')
plt.xlabel('Мощность')
plt.ylabel('Цена')
plt.title('Зависимость цены от мощности')

plt.tight_layout()
plt.show()

print(f'Оценки коэффициентов: {beta_hat}')
print('Доверительные интервалы для коэффициентов:')
for i, (lower, upper) in enumerate(conf_intervals):
    print(f'beta_{i}: ({lower}, {upper})')
print(f'Коэффициент детерминации (R^2): {R_squared}')

# 2 ПРОВЕРКА ГИПОТЕЗ
t_stat = beta_hat[3] / std_err_beta[3]
p_value = 1 - t.cdf(t_stat, df=n - p)
print(f'Проверка гипотезы "Чем больше мощность, тем больше цена": t = {t_stat}, p = {p_value}')

t_stat = beta_hat[1] / std_err_beta[1]
p_value = 2 * (1 - t.cdf(abs(t_stat), df=n - p))
print(f'Проверка гипотезы "Цена изменяется в зависимости от расхода в городе": t = {t_stat}, p = {p_value}')

X_reduced = X[:, [0, 3]]
beta_hat_reduced = np.linalg.inv(X_reduced.T @ X_reduced) @ X_reduced.T @ y
y_pred_reduced = X_reduced @ beta_hat_reduced
SS_residual_reduced = np.sum((y - y_pred_reduced) ** 2)
F_stat = ((SS_residual_reduced - SS_residual) / 2) / (SS_residual / (n - p))
p_value = 1 - t.cdf(F_stat, df=n - p)
print(f'Проверка гипотезы "H0: beta_1 = beta_2 = 0": F = {F_stat}, p = {p_value}')
