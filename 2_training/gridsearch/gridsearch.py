#!/usr/bin/env python3
# Preload packages
import pandas as pd
import numpy as np
import sklearn
import joblib

#load data
bc = pd.read_csv("../Mix_BC_srbal.csv.gz")
bc_input = bc.iloc[0:466, 0:300]
bc_output = bc['Class']
# Data partition (mathematical notation)

from sklearn.model_selection import train_test_split as tts
X, Xt, y, yt = tts(bc_input,bc_output,random_state=74)

#GridSearching fot tuning hyperparameters
# RBF
# 1) gamma y C de 0.0001 a 1000 becomes 0.91
# 2) gamma de 0.0001 a 1 y C de 1 a 1000 becomes 0.9484
# 3) gamma de 0.3 a 0.33 y C de 20 a 40 becomes 
gamma = [i/1000 for i in range(300,330,1)]
C = [i for i in range(10,40,1)]
# Linear: 
# gamma = [j/100000 for j in range(200,600,20)]
# C = [i/2 for i in range(10,50,2)] 
kernel = ['rbf']
param_grid = {'C': C,'gamma': gamma, 'kernel':kernel}
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(SVC(), param_grid, cv=5)
gs.fit(X,y)
#gs.best_estimator_
#gs.best_score_
#gs.best_accuracy_
results = pd.DataFrame(gs.cv_results_)
Features = list(results.columns)
params = pd.DataFrame(results,columns=Features)
params.to_csv('params2.csv', index=False)
'''
Problema dual: maximizas el valor de la ganancia
Los coefcientes dan los  hiperplanos que claifican entre cancer y no cancer. 
Se puede graficar el CV, como heatmap
Hacer un scatter plot con los mejores accuracy (como superficie)
Se buscaron los parámetros con la funcion gridsearchcv en sklearn para svm: discretiza el espacio de parametros y prueba (C: de 1 a 20) y (gamma: de 0.001 hasta 1 con salto de 0.1)
'''
# Grid search en busca de los parametros del SVM lineal
# the process goes like this:
# best_score of 0.811 with C=100, gamma=0.001
# best_score of 0.796 with C=5, gamma=0.01
# best score of 0.811 with C=45, gmma = 0.001
# best_score of 0.814 with C=7, gamma=0.0044 (0.829 con Xt y yt)
# best score of 0.8137 with C=21, gamma = 0.002
# best score of 0.817 with C = 9, gamma = 0.002
# best score of 0.817 with C = 9, gamma = 0.00315 (0.8205 con Xt y yt)
# best score of 0.819 with C = 11, gamma = 0.003  (0.8205 con Xt y yt)
# best score of 0.8193374 with C = 10, gamma = 0.0028  (0.82051 con Xt y yt)
# Parece que existen varios maximos locales
