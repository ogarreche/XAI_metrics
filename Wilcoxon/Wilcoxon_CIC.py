# %%
import pandas as pd

req_cols = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',' Idle Std',' Idle Max',' Idle Min',' Label']

fraction = 1

print('---------------------------------------------------------------------------------')
print('Loading Databases')
print('---------------------------------------------------------------------------------')
print('')

df0 = pd.read_csv ('cicids_db/Wednesday-workingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)
df1 = pd.read_csv ('cicids_db/Tuesday-WorkingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)
df2 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)
df3 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)
df4 = pd.read_csv ('cicids_db/Monday-WorkingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)
df5 = pd.read_csv ('cicids_db/Friday-WorkingHours-Morning.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)
df6 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)
df7 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)

frames = [df0, df1, df2, df3, df4, df5,df6, df7]

df = pd.concat(frames,ignore_index=True)
# df = df.sample(frac = 1)
# df = pd.concat(frames,ignore_index=True)
# df = df.sample(frac = fraction )
y = df.pop(' Label')
df = df.assign(Label = y)


# Specify the name of the output text file
output_file_name = "Wilcoxon_CIC.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
###################################################
###################################################
###################################################
###################################################

# %%
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn
print('---------------------------------------------------------------------------------')
print('Reducing Normal rows')
print('---------------------------------------------------------------------------------')
print('')

frac_normal = 0.2
split = 0.8

#filters
filtered_normal = df[df['Label'] == 'BENIGN']

#reduce
reduced_normal = filtered_normal.sample(frac=frac_normal)

#join
df = pd.concat([df[df['Label'] != 'BENIGN'], reduced_normal])

''' ---------------------------------------------------------------'''
df_max_scaled = df.copy()


y = df_max_scaled['Label'].replace({'DoS GoldenEye': 'Dos/Ddos', 'DoS Hulk': 'Dos/Ddos', 'DoS Slowhttptest': 'Dos/Ddos', 'DoS slowloris': 'Dos/Ddos', 'Heartbleed': 'Dos/Ddos', 'DDoS': 'Dos/Ddos','FTP-Patator': 'Brute Force', 'SSH-Patator': 'Brute Force','Web Attack - Brute Force': 'Web Attack', 'Web Attack - Sql Injection': 'Web Attack', 'Web Attack - XSS': 'Web Attack'})

df_max_scaled.pop('Label')


print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')

df_max_scaled
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
df = df_max_scaled.assign( Label = y)
#df
df = df.fillna(0)

y = df.pop('Label')
X = df
df = df.assign(Label = y)

counter = Counter(y)
print(counter)


X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split,random_state=42)
df = X.assign( Label = y)
# df[['bp_before','bp_after']].describe()

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


