from Titanic_program import titanic_train, titanic_test
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



titanic_combined_data = [titanic_train, titanic_test] # combining train and test dataset

for dataset in titanic_combined_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')