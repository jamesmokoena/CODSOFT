from Titanic_program import titanic_train
import seaborn as sns
import matplotlib.pyplot as plt




def PclasS_VS_Survival():
    titanic_train.Pclass.value_counts()
    titanic_train.groupby('Pclass').Survived.value_counts()
    sns.barplot(x='Pclass', y='Survived', data=titanic_train)
    plt.show()

def Sex_vs_Survival(): 
    titanic_train.groupby('Sex').Survived.value_counts()
    titanic_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
    sns.barplot(x='Sex', y='Survived', data=titanic_train)
    



if __name__ == '__main__':
    print("  Which Relationship between Features and Survival do you want look at: \n 1. PclasS_VS_Survival\n 2. Sex_vs_Survival\n")
    answer = int(input ("SELECTION: (e.g 2):  "))

    if answer == 1:
        PclasS_VS_Survival()
    elif answer ==2 :
        Sex_vs_Survival()