import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f

data = pd.read_csv('../dataset/iris.csv')

# находим суммарную площадь чашелистика и лепестка
data['Total.Area'] = data['Sepal.Length'] * data['Sepal.Width'] + data['Petal.Length'] * data['Petal.Width']
groups = data.groupby('Species')['Total.Area']
grand_mean = data['Total.Area'].mean()

# межгрупповая сумма квадратов (SSB)
ssb = sum(group.size * (group.mean() - grand_mean) ** 2 for name, group in groups)
# внутригрупповая сумма квадратов (SSW)
ssw = sum(((group - group.mean()) ** 2).sum() for name, group in groups)
# число групп и число наблюдений
k = len(groups)
n = len(data)
# степени свободы
dfb = k - 1
dfw = n - k
# средние квадраты
msb = ssb / dfb
msw = ssw / dfw
# F-статистика
F = msb / msw
# Критическое значение F
alpha = 0.05
F_critical = f.ppf(1 - alpha, dfb, dfw)
# P-значение
p_value = 1 - f.cdf(F, dfb, dfw)

print(f'F-статистика: {F}')
print(f'Критическое значение F: {F_critical}')
print(f'P-значение: {p_value}')

if F > F_critical:
    print("Отвергаем нулевую гипотезу о равенстве средних на каждом уровне фактора")
else:
    print("Не отвергаем нулевую гипотезу о равенстве средних на каждом уровне фактора")

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Species', y='Total.Area', data=data)
plt.title('Boxplot площади чашелистика и лепестка по видам ирисов')
plt.xlabel('Вид ириса')
plt.ylabel('Суммарная площадь (чашелистик + лепесток)')

plt.subplot(1, 2, 2)
sns.barplot(x='Species', y='Total.Area', data=data, errorbar='sd')
plt.title('Средняя площадь чашелистика и лепестка по видам ирисов')
plt.xlabel('Вид ириса')
plt.ylabel('Суммарная площадь (чашелистик + лепесток)')

plt.tight_layout()
plt.show()
