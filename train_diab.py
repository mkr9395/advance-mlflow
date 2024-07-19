import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1] 


rf = RandomForestClassifier(random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    
    'n_estimators' : [10, 20, 30],
    'max_depth' : [None, 10 , 20, 30]
}
    
    
    

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv = 5, n_jobs=-1, verbose=2)

grid_search.fit(X_train,y_train)


# Displaying best params and best score

best_params = grid_search.best_params_

best_score = grid_search.best_score_  

print('best_params',best_params)

print('best_score',best_score)  
    
    



