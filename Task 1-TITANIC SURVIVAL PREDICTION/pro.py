# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

titanic_train = pd.read_csv('Task 1-TITANIC SURVIVAL PREDICTION/DATASET/train.csv')


# Step 1: Load the titanic_trainset
# titanic_train = pd.read_csv("Task 1-TITANIC SURVIVAL PREDICTION/titanic_trainSET/train.csv")

def load_titanic_train_Data(titanic_train):
    # Step 2: titanic_train Exploration and Visualization
    # Display basic information about the titanic_trainset
    print(titanic_train.info())

    # Display the first few rows of the titanic_trainset
    print(titanic_train.head())

    # Visualize some basic statistics
    print(titanic_train.describe())


    # Visualize missing titanic_train
    plt.figure(figsize=(10, 6))
    sns.heatmap(titanic_train.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing titanic_train")
    plt.show()

load_titanic_train_Data(titanic_train)

def handling_missing_values(titanic_train):
    # Step 3: Titanic Train Preprocessing
    # Handling missing values
    titanic_train.drop("Name", axis=1, inplace=True)
    titanic_train.drop(["Cabin", "Ticket"], axis=1, inplace=True)  # Drop the "Ticket" column
    titanic_train["Age"].fillna(titanic_train["Age"].median(), inplace=True)
    titanic_train["Embarked"].fillna(titanic_train["Embarked"].mode()[0], inplace=True)
    titanic_train = pd.get_dummies(titanic_train, columns=["Sex", "Embarked"], drop_first=True)

    # Create the target variable (Survived)
    X = titanic_train.drop("Survived", axis=1)
    Y = titanic_train["Survived"]

    return X, Y

    


def split_dataSETS( X,Y):
    # Step 4: Split the titanic_train into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training titanic_train
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

X,Y = handling_missing_values(titanic_train)
split_dataSETS(X,Y)