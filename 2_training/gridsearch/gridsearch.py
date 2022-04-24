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
# 3) gamma de 0.3 a 0.33 y C de 20 a 40 becomes 0.9484
# 4) Se estabiliza entre C: 25, gamma = 0.303 
gamma = [i/100 for i in range(1,101)]
C = [i for i in range(1,101)]
kernel = ['linear']
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
params.to_csv('lin.csv', index=False)
'''
Problema dual: maximizas el valor de la ganancia
Los coefcientes dan los  hiperplanos que claifican entre cancer y no cancer. 
Se puede graficar el CV, como heatmap
Hacer un scatter plot con los mejores accuracy (como superficie)
Se buscaron los parámetros con la funcion gridsearchcv en sklearn para svm: discretiza el espacio de parametros y prueba (C: de 1 a 20) y (gamma: de 0.001 hasta 1 con salto de 0.1)
'''
# Grid search en busca de los parametros del SVM lineal
# 100*100 mesh from gamma: 0.01 to 1, and C from 1 to 100
# generates an accuracy of 0.91 with C: 3, gamma= 0.8
