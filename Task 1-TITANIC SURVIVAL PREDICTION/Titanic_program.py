import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set()


titanic_train = pd.read_csv('./DATASET/train.csv')
titanic_test = pd.read_csv('./DATASET/test.csv')

#print(titanic_train.shape)

sns.catplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=titanic_train)
#plt.show()


