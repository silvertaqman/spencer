#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sklearn
import joblib

#Load data
cancer = pd.read_csv("./Mix_BC.csv.gz")
#Data Warehousing and Feature selection
## Change name of ID and class columns
cancer.columns = list(cancer.columns[:-2])+["ProtID","Class"]
## Reordering IDs and Class columns
cancer.insert(0, 'ProtID', cancer.pop('ProtID'))

## MinMax Scale to numerical data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(cancer.iloc[0:377, 3:8744])
scaled = scaler.fit_transform(cancer.iloc[0:377, 3:8744])
cancer_s = pd.DataFrame(scaled, columns = cancer.iloc[0:377, 3:8744].columns)
### exporting scaler: loadable with joblib.load("minmax.pkl"")
joblib.dump(scaler, "minmax.pkl")

## Remove duplicated values (No duplicates)
cancer_s.drop_duplicates(keep=False, inplace=True)

## Mathematical notation
y = cancer_s['Class']
X = cancer_s.drop('Class', axis = 1)
F = list(X.columns)

<<<<<<< HEAD
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, SelectPercentile, chi2, f_classif, mutual_info_classif
=======
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, chi2, f_classif, mutual_info_classif
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
## Remove invariant columns 
selector= VarianceThreshold()
Xs = selector.fit_transform(X.values)

### export invariant features
FS = []
[FS.append(F[i]) for i in selector.get_support(indices=True)]
with open("invariant_features.txt","w") as f:
	[f.write("%s\n" % i) for i in list(set(F) - set(FS))]

F = FS

## Feature Subset Selection 
<<<<<<< HEAD

## Principal Component Analysis 
Xs = pd.DataFrame(Xs, columns = FS)
from sklearn.decomposition import PCA
pca = PCA(0.99).fit(Xs.T)
CP = pd.DataFrame(pca.components_.T)
CP.to_csv("PCAComponents.csv")
EV = pd.DataFrame(pca.explained_variance_ratio_.T)
EV.to_csv("PCAVarianceRatios.csv")

## We can explain 99 percent of variance with nearly 355 feats
## Now we use FSS with chi2, Anova-F and mutual-info 

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV

# Compare methods: k-top ordered, p-top ordered and regularization

from sklearn.ensemble import ExtraTreesClassifier
selector = {
	'kbest_chi2': SelectKBest(chi2, k=350),
	'kbest_f':SelectKBest(f_classif, k=350), 
	'kbest_mutual':SelectKBest(mutual_info_classif, k=350),
	'perc_chi2':SelectPercentile(chi2, percentile=3),
	'perc_f':SelectPercentile(percentile=3.5),
	'perc_mutual':SelectPercentile(mutual_info_classif, percentile=3.5)
}

# ElasticNetCV
clf = ElasticNetCV(l1_ratio = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], alphas=np.logspace(-5,5,200), cv=10).fit(Xs,y)
importance = np.abs(clf.coef_)
threshold = np.sort(importance)[-350] + 0.01
elasticnet = SelectFromModel(clf, threshold=threshold).fit(Xs, y)

# Final comparison

sel=[p.fit_transform(Xs,y) for p in selector.values()]
features = [p.get_support(indices=True) for p in selector.values()]
=======
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

# Compare methods: kbest, percentile, False positive rate, false discovery rate, family wise, ridgeCV and Tree Model based
from sklearn.ensemble import ExtraTreesClassifier
selector = {
	'kbest_chi2': SelectKBest(chi2, k=356),
	'kbest_f':SelectKBest(f_classif), 
	'kbest_mutual':SelectKBest(mutual_info_classif),
	'perc_chi2':SelectPercentile(chi2, percentile=3),
	'perc_f':SelectPercentile(percentile=3),
	'perc_mutual':SelectPercentile(mutual_info_classif, percentile=3),
	'fpr_chi2':SelectFpr(chi2, alpha=1e-1),
	'fpr_f':SelectFpr(f_classif, alpha=0.015),
#	'fpr_mutual':SelectFpr(mutual_info_classif(X,y), alpha=1e-1),
#	'fdr_chi2':SelectFdr(chi2, alpha=1e-1),
	'fdr_f':SelectFdr(f_classif, alpha=1e-1),
#	'fdr_mutual':SelectFdr(mutual_info_classif, alpha=1e-1),
#	'fwe_chi2':SelectFwe(chi2, alpha=1e-1),
	'fwe_f':SelectFwe(f_classif, alpha=1e-1)
#	'fwe_mutual':SelectFwe(mutual_info_classif, alpha=1e-1)
}

# RidgeCV
clf = RidgeCV(alphas=np.logspace(-5,5,10)).fit(X,y)
importance = np.abs(clf.coef_)
threshold = np.sort(importance)[-356] + 0.01
ridge = SelectFromModel(clf, threshold=threshold).fit(X, y)

# LassoCV
clf = LassoCV(alphas=np.logspace(-5,5,100)).fit(X,y)
importance = np.abs(clf.coef_)
threshold = np.sort(importance)[-356] + 0.01
lasso = SelectFromModel(clf, threshold=threshold).fit(X, y)

# ElasticNetCV
clf = ElasticNetCV(l1_ratio = [0.05, 0.1, 0.5, 0.9, 0.95],alphas=np.logspace(-5,5,200), cv=10).fit(X,y)
importance = np.abs(clf.coef_)
threshold = np.sort(importance)[-356] + 0.01
elasticnet = SelectFromModel(clf, threshold=threshold).fit(X, y)

# Final comparison

sel=[p.fit_transform(X,y) for p in selector.values()]
features = [p.get_support(indices=True) for p in selector.values()]
features.append(ridge.get_support(indices=True))
features.append(lasso.get_support(indices=True))
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
features.append(elasticnet.get_support(indices=True))
total_feat=[]
for i in features:
	total_feat.append(np.array([F[k] for k in i]))


<<<<<<< HEAD
=======
selector['ridgeCV']=[]
selector['lassoCV']=[]
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
selector['ElasticNetCV']=[]
metrics = {k:f for k,f in zip(selector.keys(),total_feat)}
metrics = pd.DataFrame().from_dict(metrics, orient="index").T
metrics.to_csv("selection.csv")

<<<<<<< HEAD
# Compare FSS trough dendrograms/heatmap (graphic)
# Selected features shared between methods are used to better explain variability

# Rearranged and selected as 275-top. 
feat = pd.read_csv("topfeatures.csv")

## Create and export the working dataframe: scaled and reduced
cancer_sr = Xs[feat["Coincidence"]]
cancer_sr['Class'] = y
cancer_sr.to_csv("./Mix_BC_sr.csv")
=======
# Compare sel features trough dendrograms/heatmap (graphic)

# Using kbest Chi2
selector = SelectKBest(chi2, k=356)
Xs = selector.fit_transform(Xs, y)
feat = selector.get_support(indices=True)

## Create and export the working dataframe: scaled and reduced
cancer_sr = X.iloc[:,feat]
cancer_sr['Class'] = y
cancer_sr.to_csv("./Mix_BC_sr.csv", index=False)
'''
## Principal Component Analysis (from 8708 to 356 to explain 0.99 of variance)
from sklearn.decomposition import PCA
pca = PCA(0.99).fit(X)
'''
>>>>>>> 04b0460809710406095b4e1a427dc0fcdc9198cd
