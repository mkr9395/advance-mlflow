# Logging dataset in MlFlow
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# connect to dagshub
# import dagshub
# dagshub.init(repo_owner='mkr9395', repo_name='mlflow-dagshub-demo', mlflow=True)


# mlflow.set_tracking_uri('https://dagshub.com/mkr9395/mlflow-dagshub-demo.mlflow')

# Load the iris dataset
iris = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
X = iris.iloc[:,:-1]
y = iris.iloc[:,-1]

le = LabelEncoder()
y = le.fit_transform(y)


# Get the target names
target_names = le.classes_

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = X_train
train_df['variety'] = y_train


test_df = X_test
test_df['variety'] = y_test


# Define the parameters for the Random Forest model
max_depth = 1
n_estimators = 100

# apply mlflow

mlflow.set_experiment('iris-rf')

with mlflow.start_run():
    
    rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)
    
    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # save the plot as an artifact
    plt.savefig("confusion_matrix.png")
    
    # mlflow code:
    
    # log the plots
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    
    # log the model
    mlflow.sklearn.log_model(rf,"random_forest")
    
    # mlflow tags
    mlflow.set_tag('author','Mohit')
    mlflow.set_tag('model','decision_tree')
    
    print('accuracy',accuracy)
    
    
    # converting data into mlflow format
    
    train_df = mlflow.data.from_pandas(train_df)
    test_df = mlflow.data.from_pandas(test_df)
    
    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"validation")
    
    
    