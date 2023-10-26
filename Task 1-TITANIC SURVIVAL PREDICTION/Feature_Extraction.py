from Titanic_program import titanic_train, titanic_test
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

titanic_combined_data = [titanic_train, titanic_test] # combining train and test dataset

def combine_datasets():
    for dataset in titanic_combined_data:
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    print(titanic_train['Title'].value_counts())
    print(titanic_test['Title'].value_counts())

def title_map():
    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
    "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
    "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

    for dataset in titanic_combined_data:
        dataset['Title'] = dataset['Title'].map(title_mapping)   
        print(dataset.head())

def dropping_unnecessary_features():

    titanic_test.drop('Name', axis=1, inplace=True)
    titanic_train.drop('Name', axis=1, inplace=True)
    print(titanic_train.head())


def sex_mapping():
    sex_mapping = {"male": 0, "female": 1}
    for dataset in titanic_combined_data:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    print(dataset)


    
     


