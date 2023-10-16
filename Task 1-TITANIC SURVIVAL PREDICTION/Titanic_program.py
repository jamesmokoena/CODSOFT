import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set()


titanic_train = pd.read_csv('Task 1-TITANIC SURVIVAL PREDICTION/train.csv')
titanic_test = pd.read_csv('Task 1-TITANIC SURVIVAL PREDICTION/test.csv')

def Age_vs_Survival():
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    sns.violinplot(x="Embarked", y="Age", hue="Survived", data=titanic_train, split=True, ax=ax1)
    sns.violinplot(x="Pclass", y="Age", hue="Survived", data=titanic_train, split=True, ax=ax2)
    sns.violinplot(x="Sex", y="Age", hue="Survived", data=titanic_train, split=True, ax=ax3)
    plt.show()

if __name__ == "__main__":
    Age_vs_Survival()