import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()  # устанавливаем seaborn по умолчанию для отрисовки графиков

#2. Загрузить данные в датафрейм
NHANES = pd.read_csv('D:\\Учеба\\Нейронки\\ЛАБЫ\\Андрей\\ЛБ 1\\NHANES_age_prediction.csv',
                     delimiter=';',

                     )
#3. Вывести статистическую информацию о наборе данных
print(NHANES.info())

#4. Вывести названия столбцов и строк
print(NHANES.head())

#5. Заменить категориальные данные количественными
NHANES['age_group'] = NHANES['age_group'].replace({'Adult': 1, 'Senior': 0})
print(NHANES.head())
#6. Визуализации
# SEQN - Порядковый номер респондента
# age_group - Возрастная группа респондента (старший/нестарший)
# RIDAGEYR - Возраст респондента
# RIAGENDR - Пол респондента
# PAQ605 - Если респондент занимается спортом средней или высокой интенсивности, фитнесом или развлекательными мероприятиями в течение типичной недели
# BMXBMI - Индекс массы тела респондента
# LBXGLU - Уровень глюкозы в крови респондента после голодания
# DIQ010 - Если респондент болен диабетом
# LBXGLT - Устные показания ответчика
# LBXIN - Уровень инсулина в крови респондента
#

#age_group

sns.countplot(x=NHANES["age_group"],);
plt.suptitle("Adult = 1; Senior = 0")
plt.show()

#age_group + RIAGENDR
sns.countplot(data=NHANES, x="RIAGENDR", hue='age_group');
plt.suptitle("Зависимость пола и возрастной группы")
plt.show()
#plt.hist(NHANES[['PAQ605']])


#RIDAGEYR
plt.hist(x=NHANES["RIDAGEYR"],);
plt.suptitle("Возраст участников")
plt.show()


#PAQ605
plt.hist(x=NHANES["PAQ605"],);
plt.suptitle("Занятия спортом респондентов")
plt.show()

#age_group + PAQ605
sns.countplot(data=NHANES, x="PAQ605", hue='age_group');
plt.suptitle("Зависимость спортивной активности и возрастной группы")
plt.show()

#BMXBMI + LBXGLU
sns.scatterplot(data=NHANES, x="BMXBMI", y="LBXGLU");
plt.suptitle("Зависимость индекса массы тела от уровня глюкозы")
plt.show()

#"DIQ010
plt.hist(x=NHANES["DIQ010"],);
plt.suptitle("Если респондент болен диабетом")
plt.show()

#RIDAGEYR + DIQ010
sns.scatterplot(data=NHANES, x="RIDAGEYR", y="DIQ010");
plt.suptitle("Зависимость индекса массы тела от уровня глюкозы")
plt.show()

#BMXBMI + LBXIN
sns.scatterplot(data=NHANES, x="BMXBMI", y="LBXIN");
plt.suptitle("Зависимость индекса массы тела от уровня инсулина")
plt.show()

#RIAGENDR + LBXGLT
sns.scatterplot(data=NHANES, x="RIAGENDR", y="LBXGLT");
plt.suptitle("Зависимость пола от устных показаний")
plt.show()

# LBXIN + RIDAGEYR
sns.lineplot(data=NHANES, x='RIDAGEYR', y='LBXIN');
plt.suptitle("Зависимость возраста и уровня инсулина")
plt.show()