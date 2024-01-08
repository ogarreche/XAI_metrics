import pandas as pd


req_cols = ['FLOW_DURATION_MILLISECONDS','FIRST_SWITCHED',
            'TOTAL_FLOWS_EXP','TCP_WIN_MSS_IN','LAST_SWITCHED',
            'TCP_WIN_MAX_IN','TCP_WIN_MIN_IN','TCP_WIN_MIN_OUT',
           'PROTOCOL','TCP_WIN_MAX_OUT','TCP_FLAGS',
            'TCP_WIN_SCALE_OUT','TCP_WIN_SCALE_IN','SRC_TOS',
            'DST_TOS','FLOW_ID','L4_SRC_PORT','L4_DST_PORT',
           'MIN_IP_PKT_LEN','MAX_IP_PKT_LEN','TOTAL_PKTS_EXP',
           'TOTAL_BYTES_EXP','IN_BYTES','IN_PKTS','OUT_BYTES','OUT_PKTS',
            'ALERT']




 
print('Loading Database')
print('--------------------------------------------------')
fraction = 0.3
#Denial of Service
df0 = pd.read_csv ('sensor_db/dos-03-15-2022-15-44-32.csv', usecols=req_cols).sample(frac = fraction)
df1 = pd.read_csv ('sensor_db/dos-03-16-2022-13-45-18.csv', usecols=req_cols).sample(frac = fraction)
df2 = pd.read_csv ('sensor_db/dos-03-17-2022-16-22-53.csv', usecols=req_cols).sample(frac = fraction)
df3 = pd.read_csv ('sensor_db/dos-03-18-2022-19-27-05.csv', usecols=req_cols).sample(frac = fraction)
df4 = pd.read_csv ('sensor_db/dos-03-19-2022-20-01-53.csv', usecols=req_cols).sample(frac = fraction)
df5 = pd.read_csv ('sensor_db/dos-03-20-2022-14-27-54.csv', usecols=req_cols).sample(frac = fraction)
#Malware
#df6 = pd.read_csv ('sensor_db/malware-03-25-2022-17-57-07.csv', usecols=req_cols)
#Normal
df7 = pd.read_csv  ('sensor_db/normal-03-15-2022-15-43-44.csv', usecols=req_cols).sample(frac = fraction)
df8 = pd.read_csv  ('sensor_db/normal-03-16-2022-13-44-27.csv', usecols=req_cols).sample(frac = fraction)
df9 = pd.read_csv  ('sensor_db/normal-03-17-2022-16-21-30.csv', usecols=req_cols).sample(frac = fraction)
df10 = pd.read_csv ('sensor_db/normal-03-18-2022-19-17-31.csv', usecols=req_cols).sample(frac = fraction)
df11 = pd.read_csv ('sensor_db/normal-03-18-2022-19-25-48.csv', usecols=req_cols).sample(frac = fraction)
df12 = pd.read_csv ('sensor_db/normal-03-19-2022-20-01-16.csv', usecols=req_cols).sample(frac = fraction)
df13 = pd.read_csv ('sensor_db/normal-03-20-2022-14-27-30.csv', usecols=req_cols).sample(frac = fraction)

#PortScanning
df14 = pd.read_csv  ('sensor_db/portscanning-03-15-2022-15-44-06.csv', usecols=req_cols).sample(frac = fraction)
df15 = pd.read_csv  ('sensor_db/portscanning-03-16-2022-13-44-50.csv', usecols=req_cols).sample(frac = fraction)
df16 = pd.read_csv  ('sensor_db/portscanning-03-17-2022-16-22-53.csv', usecols=req_cols).sample(frac = fraction)
df17 = pd.read_csv  ('sensor_db/portscanning-03-18-2022-19-27-05.csv', usecols=req_cols).sample(frac = fraction)
df18 = pd.read_csv  ('sensor_db/portscanning-03-19-2022-20-01-45.csv', usecols=req_cols).sample(frac = fraction)
df19 = pd.read_csv  ('sensor_db/portscanning-03-20-2022-14-27-49.csv', usecols=req_cols).sample(frac = fraction)

frames = [df0, df1, df2, df3, df4, df5, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]
df = pd.concat(frames,ignore_index=True)

# shuffle the DataFrame rows
df = df.sample(frac =1)

# assign alert column to y
y = df.pop('ALERT')

# join alert back to df
df = df.assign( ALERT = y) 

#Fill NaN with 0s
df = df.fillna(0)

# Specify the name of the output text file
output_file_name = "Wilcoxon_SML.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
###################################################
###################################################
###################################################
###################################################

from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn
split = 0.8
filtered_normal = df[df['ALERT'] == 'None']

#reduce

reduced_normal = filtered_normal.sample(frac=0.2)

#join

df = pd.concat([df[df['ALERT'] != 'None'], reduced_normal])


# Normalize database
print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')

# make a df copy
df_max_scaled = df.copy()

# assign alert column to y
y = df_max_scaled.pop('ALERT')

#Normalize operation
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
#assign df copy to df
df = df_max_scaled.assign( ALERT = y)
#Fill NaN with 0s
df = df.fillna(0)

y = df.pop('ALERT')
X = df

y, y_Label = pd.factorize(y)

X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split,random_state=42)
df = X.assign( ALERT = y)


# %%
#Model Parameters RF
from sklearn.ensemble import RandomForestClassifier
max_depth = 5
n_estimators = 5
min_samples_split = 2

print('---------------------------------------------------------------------------------')
print('Defining the RF model')
print('---------------------------------------------------------------------------------')
print('')

rf = RandomForestClassifier(max_depth = max_depth,  n_estimators = n_estimators, min_samples_split = min_samples_split, n_jobs = -1)

# model = clf.fit(X, y)

# %%
#Model Parameters SVM
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

max_iter=10
loss='hinge'
gamma=0.1


rbf_feature = RBFSampler(gamma=gamma, random_state=1)
X_features = rbf_feature.fit_transform(X_train)
svm = SGDClassifier(max_iter=max_iter,loss=loss)
# clf.fit(X_features, y_train)
# clf.score(X_features, y_train)


# %%
#Model Parameters ADA
from sklearn.ensemble import AdaBoostClassifier
n_estimators=100
learning_rate=0.5

ada = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)


# %%

# #Model Parameters DNN
# import tensorflow as tf
# dropout_rate = 0.01
# nodes = 70
# out_layer = 7
# optimizer='adam'
# loss='sparse_categorical_crossentropy'
# epochs=5
# batch_size=2*256


# print('---------------------------------------------------------------------------------')
# print('Defining the DNN model')
# print('---------------------------------------------------------------------------------')
# print('')


# num_columns = X_train.shape[1]

# dnn = tf.keras.Sequential()

# # Input layer
# dnn.add(tf.keras.Input(shape=(num_columns,)))

# # Dense layers with dropout
# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# dnn.add(tf.keras.layers.Dense(nodes))
# dnn.add(tf.keras.layers.Dropout(dropout_rate))

# # Output layer
# dnn.add(tf.keras.layers.Dense(out_layer))



# dnn.compile(optimizer=optimizer, loss=loss)

# dnn.summary()

import tensorflow as tf
from scikeras.wrappers import KerasClassifier, KerasRegressor

def getModel(optimizer):


    num_columns = X_train.shape[1]

    dnn = tf.keras.Sequential()

    # Input layer
    dnn.add(tf.keras.Input(shape=(num_columns,)))

    # Dense layers with dropout
    dnn.add(tf.keras.layers.Dense(nodes))
    dnn.add(tf.keras.layers.Dropout(dropout_rate))

    dnn.add(tf.keras.layers.Dense(nodes))
    dnn.add(tf.keras.layers.Dropout(dropout_rate))

    dnn.add(tf.keras.layers.Dense(nodes))
    dnn.add(tf.keras.layers.Dropout(dropout_rate))

    dnn.add(tf.keras.layers.Dense(nodes))
    dnn.add(tf.keras.layers.Dropout(dropout_rate))

    dnn.add(tf.keras.layers.Dense(nodes))
    dnn.add(tf.keras.layers.Dropout(dropout_rate))

    # Output layer
    dnn.add(tf.keras.layers.Dense(out_layer))



    dnn.compile(optimizer=optimizer, loss=loss)

    return dnn


#Model Parameters DNN
dropout_rate = 0.01
nodes = 70
out_layer = 7
optimizer='adam'
loss='sparse_categorical_crossentropy'
epochs=5
batch_size=2*256
# optimizer = ['Adam']
epochs = [5]


param_grid = dict(epochs=epochs, optimizer=optimizer)
Kmodel = KerasClassifier(build_fn=getModel, verbose=1)

# %%
#Model Parameters KNN
from sklearn.neighbors import KNeighborsClassifier
n_neighbors=5

knn_clf=KNeighborsClassifier(n_neighbors=n_neighbors)

# %%
#Model Parameters LGBM
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
n_split= 3
n_repeat= 15

lgbm = LGBMClassifier()
# cv = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat)
# n_scores = cross_val_score(lgbm, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# fit the model on the whole dataset
# lgbm = LGBMClassifier()



# %%
#Model Parameters MLP
from sklearn.neural_network import MLPClassifier
max_iter=70
# MLP = MLPClassifier(random_state=1, max_iter=max_iter).fit(X_train, y_train)
MLP = MLPClassifier(random_state=42, max_iter=max_iter)

# %%
from scipy.stats import wilcoxon
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Prepare models and select your CV method
# model1 = ExtraTreesClassifier()
# model2 = RandomForestClassifier()
model1 = svm
model2 = rf
model3 = ada
model4 = Kmodel
model5 = lgbm
model6 = MLP
model7 = knn_clf
kf = KFold(n_splits=20, random_state=42)

# Extract results for each model on the same folds
# results_model1 = cross_val_score(model1, X, y, cv=kf)
# results_model2 = cross_val_score(model2, X, y, cv=kf)
# results_model3 = cross_val_score(model3, X, y, cv=kf)
# results_model4 = cross_val_score(model4, X, y, cv=kf)
# results_model5 = cross_val_score(model5, X, y, cv=kf)
# results_model6 = cross_val_score(model6, X, y, cv=kf)
# results_model7 = cross_val_score(model7, X, y, cv=kf)


# %%

# Extract results for each model on the same folds
results_model1 = cross_val_score(model1, X, y, cv=kf) #SVM
results_model2 = cross_val_score(model2, X, y, cv=kf) # RF

# %%
# Calculate p value
#SVM and RF
stat, p = wilcoxon(results_model1, results_model2, zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model1)
median_model2 = np.median(results_model2)
print("SVM and RF")
with open(output_file_name, "a") as f:print("SVM and RF", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("SVM is better.")
        with open(output_file_name, "a") as f:print("SVM is better.", file = f)
    elif median_model1 < median_model2:
        print("RF is better.")
        with open(output_file_name, "a") as f:print("RF is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#ADA
results_model3 = cross_val_score(model3, X, y, cv=kf)

# %%
#SVM and ADA
stat, p = wilcoxon(results_model1, results_model3,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model1)
median_model2 = np.median(results_model3)


print("SVM and ADA")
with open(output_file_name, "a") as f:print("SVM and ADA", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("SVM is better.")
        with open(output_file_name, "a") as f:print("SVM is better.", file = f)
    elif median_model1 < median_model2:
        print("ADA is better.")
        with open(output_file_name, "a") as f:print("ADA is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#RF and ADA
stat, p = wilcoxon(results_model2, results_model3,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model2)
median_model2 = np.median(results_model3)


print("RF and ADA")
with open(output_file_name, "a") as f:print("RF and ADA", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("RF is better.")
        with open(output_file_name, "a") as f:print("RF is better.", file = f)
    elif median_model1 < median_model2:
        print("ADA is better.")
        with open(output_file_name, "a") as f:print("ADA is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#DNN
results_model4 = cross_val_score(model4, X, y, cv=kf)

# %%


# %%
#SVM and DNN
stat, p = wilcoxon(results_model1, results_model4,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model1)
median_model2 = np.median(results_model4)



print("SVM and DNN")
with open(output_file_name, "a") as f:print("SVM and DNN", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("SVM is better.")
        with open(output_file_name, "a") as f:print("SVM is better.", file = f)
    elif median_model1 < median_model2:
        print("DNN is better.")
        with open(output_file_name, "a") as f:print("DNN is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#RF and DNN
stat, p = wilcoxon(results_model2, results_model4,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model2)
median_model2 = np.median(results_model4)

print("RF and DNN")
with open(output_file_name, "a") as f:print("RF and DNN", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("RF is better.")
        with open(output_file_name, "a") as f:print("RF is better.", file = f)
    elif median_model1 < median_model2:
        print("DNN is better.")
        with open(output_file_name, "a") as f:print("DNN is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#ADA and DNN
stat, p = wilcoxon(results_model3, results_model4,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model3)
median_model2 = np.median(results_model4)



print("ADA and DNN")
with open(output_file_name, "a") as f:print("ADA and DNN", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("ADA is better.")
        with open(output_file_name, "a") as f:print("ADA is better.", file = f)
    elif median_model1 < median_model2:
        print("DNN is better.")
        with open(output_file_name, "a") as f:print("DNN is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#LGBM
results_model5 = cross_val_score(model5, X, y, cv=kf)

# %%
#SVM and LGBM
stat, p = wilcoxon(results_model1, results_model5,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model1)
median_model2 = np.median(results_model5)



print("SVM and LGBM")
with open(output_file_name, "a") as f:print("SVM and LGBM", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("SVM is better.")
        with open(output_file_name, "a") as f:print("SVM is better.", file = f)
    elif median_model1 < median_model2:
        print("LGBM is better.")
        with open(output_file_name, "a") as f:print("LGBM is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#RF and LGBM
stat, p = wilcoxon(results_model2, results_model5,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model2)
median_model2 = np.median(results_model5)



print("RF and LGBM")
with open(output_file_name, "a") as f:print("RF and LGBM", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("RF is better.")
        with open(output_file_name, "a") as f:print("RF is better.", file = f)
    elif median_model1 < median_model2:
        print("LGBM is better.")
        with open(output_file_name, "a") as f:print("LGBM is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#ADA and LGBM
stat, p = wilcoxon(results_model3, results_model5,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model3)
median_model2 = np.median(results_model5)



print("ADA and LGBM")
with open(output_file_name, "a") as f:print("ADA and LGBM", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("ADA is better.")
        with open(output_file_name, "a") as f:print("ADA is better.", file = f)
    elif median_model1 < median_model2:
        print("LGBM is better.")
        with open(output_file_name, "a") as f:print("LGBM is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#DNN and LGBM
stat, p = wilcoxon(results_model4, results_model5,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model4)
median_model2 = np.median(results_model5)



print("DNN and LGBM")
with open(output_file_name, "a") as f:print("DNN and LGBM", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("DNN is better.")
        with open(output_file_name, "a") as f:print("DNN is better.", file = f)
    elif median_model1 < median_model2:
        print("LGBM is better.")
        with open(output_file_name, "a") as f:print("LGBM is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
# MLP
results_model6 = cross_val_score(model6, X, y, cv=kf)


# %%
#SVM and MLP
stat, p = wilcoxon(results_model1, results_model6,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model1)
median_model2 = np.median(results_model6)



print("SVM and MLP")
with open(output_file_name, "a") as f:print("SVM and MLP", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("SVM is better.")
        with open(output_file_name, "a") as f:print("SVM is better.", file = f)
    elif median_model1 < median_model2:
        print("MLP is better.")
        with open(output_file_name, "a") as f:print("MLP is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#RF and MLP
stat, p = wilcoxon(results_model2, results_model6,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model2)
median_model2 = np.median(results_model6)



print("RF and MLP")
with open(output_file_name, "a") as f:print("RF and MLP", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("RF is better.")
        with open(output_file_name, "a") as f:print("RF is better.", file = f)
    elif median_model1 < median_model2:
        print("MLP is better.")
        with open(output_file_name, "a") as f:print("MLP is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#ADA and MLP
stat, p = wilcoxon(results_model3, results_model6,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model3)
median_model2 = np.median(results_model6)



print("ADA and MLP")
with open(output_file_name, "a") as f:print("ADA and MLP", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("ADA is better.")
        with open(output_file_name, "a") as f:print("ADA is better.", file = f)
    elif median_model1 < median_model2:
        print("MLP is better.")
        with open(output_file_name, "a") as f:print("MLP is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#DNN and MLP
stat, p = wilcoxon(results_model4, results_model6,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model4)
median_model2 = np.median(results_model6)



print("DNN and MLP")
with open(output_file_name, "a") as f:print("DNN and MLP", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("DNN is better.")
        with open(output_file_name, "a") as f:print("DNN is better.", file = f)
    elif median_model1 < median_model2:
        print("MLP is better.")
        with open(output_file_name, "a") as f:print("MLP is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#LGBM and MLP
stat, p = wilcoxon(results_model5, results_model6,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model5)
median_model2 = np.median(results_model6)



print("LGBM and MLP")
with open(output_file_name, "a") as f:print("LGBM and MLP", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("LGBM is better.")
        with open(output_file_name, "a") as f:print("LGBM is better.", file = f)
    elif median_model1 < median_model2:
        print("MLP is better.")
        with open(output_file_name, "a") as f:print("MLP is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
# KNN
results_model7 = cross_val_score(model7, X, y, cv=kf)

# %%
#SVM and KNN
stat, p = wilcoxon(results_model1, results_model7,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model1)
median_model2 = np.median(results_model7)



print("SVM and KNN")
with open(output_file_name, "a") as f:print("SVM and KNN", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("SVM is better.")
        with open(output_file_name, "a") as f:print("SVM is better.", file = f)
    elif median_model1 < median_model2:
        print("KNN is better.")
        with open(output_file_name, "a") as f:print("KNN is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#RF and KNN
stat, p = wilcoxon(results_model2, results_model7,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model2)
median_model2 = np.median(results_model7)



print("RF and KNN")
with open(output_file_name, "a") as f:print("RF and KNN", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("RF is better.")
        with open(output_file_name, "a") as f:print("RF is better.", file = f)
    elif median_model1 < median_model2:
        print("KNN is better.")
        with open(output_file_name, "a") as f:print("KNN is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#ADA and KNN
stat, p = wilcoxon(results_model3, results_model7,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model3)
median_model2 = np.median(results_model7)



print("ADA and KNN")
with open(output_file_name, "a") as f:print("ADA and KNN", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("ADA is better.")
        with open(output_file_name, "a") as f:print("ADA is better.", file = f)
    elif median_model1 < median_model2:
        print("KNN is better.")
        with open(output_file_name, "a") as f:print("KNN is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#DNN and KNN
stat, p = wilcoxon(results_model4, results_model7,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model4)
median_model2 = np.median(results_model7)



print("DNN and KNN")
with open(output_file_name, "a") as f:print("DNN and KNN", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("DNN is better.")
        with open(output_file_name, "a") as f:print("DNN is better.", file = f)
    elif median_model1 < median_model2:
        print("KNN is better.")
        with open(output_file_name, "a") as f:print("KNN is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#LGBM and KNN
stat, p = wilcoxon(results_model5, results_model7,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model5)
median_model2 = np.median(results_model7)



print("LGBM and KNN")
with open(output_file_name, "a") as f:print("LGBM and KNN", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("LGBM is better.")
        with open(output_file_name, "a") as f:print("LGBM is better.", file = f)
    elif median_model1 < median_model2:
        print("KNN is better.")
        with open(output_file_name, "a") as f:print("KNN is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)

# %%
#MLP and KNN
stat, p = wilcoxon(results_model6, results_model7,  zero_method='zsplit'); p

# %%
import numpy as np

# Calculate medians
median_model1 = np.median(results_model6)
median_model2 = np.median(results_model7)



print("MLP and KNN")
with open(output_file_name, "a") as f:print("MLP and KNN", file = f)
# Compare medians
if p < 0.05:
    if median_model1 > median_model2:
        print("MLP is better.")
        with open(output_file_name, "a") as f:print("MLP is better.", file = f)
    elif median_model1 < median_model2:
        print("KNN is better.")
        with open(output_file_name, "a") as f:print("KNN is better.", file = f)
    else:
        print("Models are statistically different but have the same median.")
        with open(output_file_name, "a") as f:print("Models are statistically different but have the same median.", file = f)
else:
    print("No statistically significant difference.")
    with open(output_file_name, "a") as f:print("No statistically significant difference.", file = f)


