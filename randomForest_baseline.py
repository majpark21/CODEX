########################################################################################################################
# Automatically extract explicit features and train a random forest to establish baseline of acceptable classification #
########################################################################################################################

# TODO: clean and rewrite all the code with CNN part
#%%
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
import treeinterpreter.treeinterpreter as ti
from sklearn.model_selection import GridSearchCV
from load_data import DataProcesser
import zipfile

#%%
# Read from zip archive and format data into long
data_file = 'data/ErkAkt_6GF_len240.zip'
meas_var = 'AKT'
data = DataProcesser(data_file)
data.subset(sel_groups=meas_var, start_time=0, end_time=600)
data = data.dataset
data = pd.melt(data, id_vars=['ID', 'class'], var_name='Time', value_name='Ratio_' + meas_var)
data['Time'] = data['Time'].str.replace('^{}_'.format(meas_var), '').astype('int')
data = data.sort_values(['ID', 'Time'])

#%%
# Setup for feature extraction
dt_timeseries = data.loc[:, ['ID', 'Time', 'Ratio_' + meas_var]].copy()
dt_class = data.loc[:, ['ID', 'class']].copy()
del data  # Free memory

dt_class = dt_class.drop_duplicates()
dt_class.index = dt_class['ID']
dt_class = dt_class.drop('ID', axis=1)
y_target =  dt_class.iloc[:, 0]
# Tsfresh doesn't allow for NaN
dt_timeseries = dt_timeseries.dropna()

#%%
# Extract and select features
extracted_features = extract_features(dt_timeseries, column_id='ID', column_sort='Time',
                                      column_value='Ratio_' + meas_var, n_jobs=4)
impute(extracted_features)
features_filtered = select_features(extracted_features, dt_class['class'])
# extracted_features.to_csv('data/randForest_rawFeatures.csv')
# features_filtered.to_csv('data/randForest_fltrFeatures.csv')

#%%
# Read saved features
features_archive = zipfile.ZipFile('data/randForest_Features/randForest_Features_{}.zip'.format(meas_var), 'r')
features_filtered = pd.read_csv(features_archive.open('randForest_fltrFeatures.csv'),
                                index_col='id')
X_train, X_test, y_train, y_test = train_test_split(features_filtered, y_target, test_size=0.3, random_state=42)

#%%
# Fit a simple decision tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
# Truth on rows; Predicted on columns
metrics.confusion_matrix(y_test.values, y_predict, labels=[0,1,2,3,4,5,6])
metrics.accuracy_score(y_test.values, y_predict)

#%%
# Fit Random Forest
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
# Truth on rows; Predicted on columns
metrics.confusion_matrix(y_test.values, y_predict, labels=[0,1,2,3,4,5,6])
metrics.accuracy_score(y_test.values, y_predict)

# %%
# grid search random forest
gridsearch_forest = RandomForestClassifier()
params = {
    'n_estimators': [5*i for i in range(1, 42, 2)],
    'max_depth': [i for i in range(1, 20)],
    'min_samples_leaf' : [i for i in range(1, 10, 2)],
    'n_jobs': [-1]
}
clf = GridSearchCV(gridsearch_forest, param_grid=params, cv=5, verbose=1)
clf.fit(features_filtered, y_target)
print(clf.best_params_)
print(clf.best_score_)
# {'max_depth': 18, 'min_samples_leaf': 1, 'n_estimators': 175, 'n_jobs': -1}
# 0.5212863355458915

#%%
# Best random forest after grid search
model = RandomForestClassifier(n_estimators=175, max_depth=18, min_samples_leaf=1)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
metrics.confusion_matrix(y_test.values, y_predict, labels=[0,1,2,3,4,5,6])
metrics.accuracy_score(y_test.values, y_predict)

#%%
# Feature importance
features_dict = {}
for i in range(len(model.feature_importances_)):
    features_dict[list(features_filtered.columns.values)[i]] = model.feature_importances_[i]
features_dict=sorted(features_dict.items(), key=lambda x:x[1], reverse=True)


# Interprete the tree with treeinterpreter package
prediction, bias, contributions = ti.predict(model, X_test)
for c, feature in zip(contributions[0], X_test.columns.values):
    print(feature, c)


#%%
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