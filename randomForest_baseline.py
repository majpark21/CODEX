########################################################################################################################
# Automatically extract explicit features and train a random forest to establish baseline of acceptable classification #
########################################################################################################################

# TODO: clean and rewrite all the code

import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
import treeinterpreter.treeinterpreter as ti
import time
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


t0 = time.time()
# Read and format data
timeseries = pd.read_csv("/home/marc/Desktop/KTR_7repl_long.csv")
y = timeseries[['ID', 'Class']].copy()
timeseries = timeseries.drop('Class', axis=1)
y = y.drop_duplicates()
y.index = y['ID']
y = y.drop('ID', axis=1)
y = y.iloc[:,0]

t1 = time.time()
# Extract and select features
extracted_features = extract_features(timeseries, column_id="ID", column_sort="Time", n_jobs=4)
impute(extracted_features)
features_filtered = select_features(extracted_features, y)

t2 = time.time()
# Fit Random Forest
X_train, X_test, y_train, y_test = train_test_split(features_filtered, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

t3 = time.time()
y_predict = model.predict(X_test)
# Truth on rows; Predicted on columns
metrics.confusion_matrix(y_test.values, y_predict, labels=[0,1,2])
metrics.accuracy_score(y_test.values, y_predict)

# Feature importance
features_dict = {}
for i in range(len(model.feature_importances_)):
    features_dict[list(features_filtered.columns.values)[i]] = model.feature_importances_[i]
features_dict=sorted(features_dict.items(), key=lambda x:x[1], reverse=True)


# Interprete the tree with treeinterpreter package
prediction, bias, contributions = ti.predict(model, X_test)
for c, feature in zip(contributions[0], X_test.columns.values):
    print(feature, c)

print('Reading time: {0:.2f}s; \nFeature time: {1:.2f}s; \nFitting time: {2:.2f}s'.format(t1-t0, t2-t1, t3-t2))

# grid search random forest
features_filtered = pd.read_csv('/home/marc/Desktop/features.csv')
X_train, X_test, y_train, y_test = train_test_split(features_filtered, y, test_size=0.3, random_state=42)
gridsearch_forest = RandomForestClassifier()
params = {
    "n_estimators": [5*i for i in range(1,42,2)],
    "max_depth": [i for i in range(1, 20)],
    "min_samples_leaf" : [i for i in range(1, 10, 2)],
    "n_jobs": [4]
}
clf = GridSearchCV(gridsearch_forest, param_grid=params, cv=5 )
clf.fit(features_filtered, y)
# clf.best_params_
# Out[14]: {'max_depth': 19, 'min_samples_leaf': 9, 'n_estimators': 95, 'n_jobs': 4}
# clf.best_score_
# Out[15]: 0.7353333333333333
model = RandomForestClassifier(n_estimators=95, max_depth=19, min_samples_leaf=9)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
metrics.confusion_matrix(y_test.values, y_predict, labels=[0,1,2])
metrics.accuracy_score(y_test.values, y_predict)

# tSNE embedding
scaler = StandardScaler()
features_filtered_scaled = features_filtered.copy()
features_filtered_scaled = scaler.fit_transform(features_filtered_scaled[features_filtered_scaled.columns])
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(features_filtered_scaled)
feature_embedded = TSNE(n_components=2, init='pca', n_iter=500, learning_rate=200).fit_transform(pca_result_50)
plt.scatter(feature_embedded[:, 0], feature_embedded[:, 1], c=y)
plt.colorbar()

# With a simple decision tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
metrics.confusion_matrix(y_test.values, y_predict, labels=[0,1,2])
metrics.accuracy_score(y_test.values, y_predict)


# Compare with cnn
from load_data import DataProcesser
myProc = DataProcesser('/home/marc/Dropbox/Work/TSclass/data/paolo/KTR_DMSO_noEGF_len120_7repl.zip')
id_train_valid = myProc.id_set
train_ids = id_train_valid[id_train_valid['set'] == 'train']['ID']
validation_ids = id_train_valid[id_train_valid['set'] == 'validation']['ID']

features_filtered = pd.read_csv('/home/marc/Desktop/features.csv')
features_filtered = features_filtered.set_index('Id')

timeseries = pd.read_csv("/home/marc/Desktop/KTR_7repl_long.csv")
y = timeseries[['ID', 'Class']].copy()
y = y.drop_duplicates()
y = y.set_index('ID')

X_train = features_filtered[features_filtered.index.isin(train_ids)]
X_test = features_filtered[features_filtered.index.isin(validation_ids)]
y_train = y[y.index.isin(train_ids)]
y_train = y_train.reindex(index=X_train.index)
y_test = y[y.index.isin(validation_ids)]
y_test = y_test.reindex(index=X_test.index)

model = RandomForestClassifier(n_estimators=95, max_depth=19, min_samples_leaf=9)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_proba = model.predict_proba(X_test)
metrics.confusion_matrix(y_test.values, y_predict, labels=[0,1,2])
metrics.accuracy_score(y_test.values, y_predict)

y_proba = pd.DataFrame(y_proba, index = X_test.index)
y_proba.to_csv('/home/marc/Dropbox/Work/TSclass/comp_cnn_rf/rf_probs.csv')