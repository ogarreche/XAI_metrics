
###################################################
#               Parameter Setting                #
###################################################

fraction= 0.5 # how much of that database you want to use
frac_normal = .2 #how much of the normal classification you want to reduce
split = 0.70 # how you want to split the train/test data (this is percentage fro train)

#Model Parameters
max_depth = 5
n_estimators = 5
min_samples_split = 2

# XAI Samples
samples = 5000

#Generate Explanations ?

explanator = False

# Specify the name of the output text file
output_file_name = "RF_SHAP_SML.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)

###################################################
###################################################
###################################################


print('--------------------------------------------------')
print('RF')
print('--------------------------------------------------')
print('Importing Libraries')
print('--------------------------------------------------')

# Makes sure we see all columns

from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.datasets import load_iris
# Loading Scikits random fo
#rest classifier
import sklearn
#import lime
from sklearn.preprocessing import OneHotEncoder

from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
#from sklearn.metrics import auc_score
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from sklearn.preprocessing import label_binarize
#loading pandas
import pandas as pd
#Loading numpy
import numpy as np
# Setting random seed
import time

import shap
from scipy.special import softmax
np.random.seed(0)
import matplotlib.pyplot as plt
import random

from sklearn.metrics import f1_score, accuracy_score
#from interpret.blackbox import LimeTabular
from interpret import show
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix

from lime_stability.stability import LimeTabularExplainerOvr
import lime

pd.set_option('display.max_columns', None)
shap.initjs()

print('Defining Function')
print('--------------------------------------------------')

def oversample(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy='minority')
    # Convert to numpy and oversample
    x_np = X_train.to_numpy()
    y_np = y_train.to_numpy()
    x_np, y_np = oversample.fit_resample(x_np, y_np)
    
    # Convert back to pandas
    x_over = pd.DataFrame(x_np, columns=X_train.columns)
    y_over = pd.Series(y_np)
    return x_over, y_over
def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the feature
s, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

def ACC(x,y,w,z):
    Acc = (x+y)/(x+w+z+y)
    return Acc

def PRECISION(x,w):
    Precision = x/(x+w)
    return Precision
def RECALL(x,z):
    Recall = x/(x+z)
    return Recall
def F1(Recall, Precision):
    F1 = 2 * Recall * Precision / (Recall + Precision)
    return F1
def BACC(x,y,w,z):
    BACC =(x/(x+z)+ y/(y+w))*0.5
    return BACC
def MCC(x,y,w,z):
    MCC = (y*x-z*w)/(((x+w)*(x+z)*(y+w)*(y+z))**.5)
    return MCC
def AUC_ROC(y_test_bin,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_avg = 0
    counting = 0
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
     # plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
      #print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
        auc_avg += auc(fpr[i], tpr[i])
        counting = i+1
    return auc_avg/counting
def oversample(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy='minority')
    # Convert to numpy and oversample
    x_np = X_train.to_numpy()
    y_np = y_train.to_numpy()
    x_np, y_np = oversample.fit_resample(x_np, y_np)

    # Convert back to pandas
    x_over = pd.DataFrame(x_np, columns=X_train.columns)
    y_over = pd.Series(y_np)
    return x_over, y_over




print('Selecting Column Features')
print('--------------------------------------------------')


'''
########################################### SIMARGL Features ########################################
'''

# Select which feature method you want to use by uncommenting it.

'''
all features
'''

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

#Denial of Service
df0 = pd.read_csv ('sensor_db/dos-03-15-2022-15-44-32.csv', usecols=req_cols)
df1 = pd.read_csv ('sensor_db/dos-03-16-2022-13-45-18.csv', usecols=req_cols)
df2 = pd.read_csv ('sensor_db/dos-03-17-2022-16-22-53.csv', usecols=req_cols)
df3 = pd.read_csv ('sensor_db/dos-03-18-2022-19-27-05.csv', usecols=req_cols)
df4 = pd.read_csv ('sensor_db/dos-03-19-2022-20-01-53.csv', usecols=req_cols)
df5 = pd.read_csv ('sensor_db/dos-03-20-2022-14-27-54.csv', usecols=req_cols)

#Normal
df7 = pd.read_csv  ('sensor_db/normal-03-15-2022-15-43-44.csv', usecols=req_cols)
df8 = pd.read_csv  ('sensor_db/normal-03-16-2022-13-44-27.csv', usecols=req_cols)
df9 = pd.read_csv  ('sensor_db/normal-03-17-2022-16-21-30.csv', usecols=req_cols)
df10 = pd.read_csv ('sensor_db/normal-03-18-2022-19-17-31.csv', usecols=req_cols)
df11 = pd.read_csv ('sensor_db/normal-03-18-2022-19-25-48.csv', usecols=req_cols)
df12 = pd.read_csv ('sensor_db/normal-03-19-2022-20-01-16.csv', usecols=req_cols)
df13 = pd.read_csv ('sensor_db/normal-03-20-2022-14-27-30.csv', usecols=req_cols)

#PortScanning
df14 = pd.read_csv  ('sensor_db/portscanning-03-15-2022-15-44-06.csv', usecols=req_cols)
df15 = pd.read_csv  ('sensor_db/portscanning-03-16-2022-13-44-50.csv', usecols=req_cols)
df16 = pd.read_csv  ('sensor_db/portscanning-03-17-2022-16-22-53.csv', usecols=req_cols)
df17 = pd.read_csv  ('sensor_db/portscanning-03-18-2022-19-27-05.csv', usecols=req_cols)
df18 = pd.read_csv  ('sensor_db/portscanning-03-19-2022-20-01-45.csv', usecols=req_cols)
df19 = pd.read_csv  ('sensor_db/portscanning-03-20-2022-14-27-49.csv', usecols=req_cols)

frames = [df0, df1, df2, df3, df4, df5, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]

#concat data frames
df = pd.concat(frames,ignore_index=True)

# shuffle the DataFrame rows
df = df.sample(frac = fraction)

# assign alert column to y
y = df.pop('ALERT')

# join alert back to df
df = df.assign( ALERT = y) 

#Fill NaN with 0s
df = df.fillna(0)

#df.pop('PROTOCOL_MAP')

print('---------------------------------------------------------------------------------')
print('Removing top features')
print('---------------------------------------------------------------------------------')
print('')






df.pop('TCP_WIN_MAX_IN')
df.pop('TCP_WIN_SCALE_IN')
df.pop('TCP_WIN_MSS_IN')
df.pop('TCP_WIN_MIN_IN')
df.pop('OUT_PKTS')
df.pop('FLOW_DURATION_MILLISECONDS')
df.pop('IN_PKTS')
df.pop('L4_DST_PORT')
df.pop('LAST_SWITCHED')
df.pop('TCP_WIN_MIN_OUT')
df.pop('TCP_FLAGS')
df.pop('TOTAL_FLOWS_EXP')
df.pop('FLOW_ID')
df.pop('TCP_WIN_MAX_OUT')
df.pop('SRC_TOS')
df.pop('TCP_WIN_SCALE_OUT')
df.pop('FIRST_SWITCHED')
df.pop('L4_SRC_PORT')
df.pop('PROTOCOL')
df.pop('DST_TOS')
# df.pop('MIN_IP_PKT_LEN')
# df.pop('MAX_IP_PKT_LEN')
# df.pop('TOTAL_PKTS_EXP')
# df.pop('TOTAL_BYTES_EXP')
# df.pop('IN_BYTES')
# df.pop('OUT_BYTES')


print('---------------------------------------------------------------------------------')
print('Reducing Normal rows')
print('---------------------------------------------------------------------------------')
print('')


#filters

filtered_normal = df[df['ALERT'] == 'None']

#reduce

reduced_normal = filtered_normal.sample(frac=frac_normal)

#join

df = pd.concat([df[df['ALERT'] != 'None'], reduced_normal])

''' ---------------------------------------------------------------'''

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

# Separate features and labels
y = df.pop('ALERT')
X = df

print('---------------------------------------------------------------------------------')
print('Balance Datasets')
print('---------------------------------------------------------------------------------')
print('')

counter = Counter(y)
print(counter)

# call balance operation until all labels have the same size
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        df, y = oversample(df, y)

counter = Counter(y)


df = df.assign(ALERT = y)
print('train len',counter)

y = df.pop('ALERT')
X = df

df = df.assign(ALERT = y)
# ## Train and Test split

print('---------------------------------------------------------------------------------')
print('Separating Training and Testing db')
print('---------------------------------------------------------------------------------')
print('')

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, y, train_size=split)
df = X.assign( ALERT = y)

# Balance Dataset

print('---------------------------------------------------------------------------------')
print('Balance Datasets')
print('---------------------------------------------------------------------------------')
print('')
counter = Counter(labels_train)
print(counter)
'''
# call balance operation until all labels have the same size
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        train, labels_train = oversample(train, labels_train)


counter = Counter(labels_train)
print(counter)

'''

# # After OverSampling training dataset

train = train.assign( ALERT = labels_train)

#Drop ALert column from train
train.pop('ALERT')
# ## Training the model
u, label = pd.factorize(labels_train)

print('Defining the model')
print('--------------------------------------------------')
rf = RandomForestClassifier(max_depth = max_depth,  n_estimators = n_estimators, min_samples_split = min_samples_split, n_jobs = -1)
#------------------------------------------------------------------------------

print('Training the model')
print('------------------------------------------------------------------------------')
#START TIMER MODEL
start = time.time()
model = rf.fit(train, labels_train)
#END TIMER MODEL
end = time.time()
print('ELAPSE TIME MODEL: ',(end - start)/60, 'min')

print('------------------------------------------------------------------------------')
#------------------------------------------------------------------------------
# # Oversampling and balancing test data
counter = Counter(labels_test)
print(counter)
counter_list = list(counter.values())
'''
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        test, labels_test = oversample(test, labels_test)

'''
counter = Counter(labels_test)

#joining features and label
test = test.assign(ALERT = labels_test)

#Randomize df order
test = test.sample(frac = 1)

#Drop label column
labels_test = test.pop('ALERT')

#------------------------------------------------------------------------------
#START TIMER PREDICTION
start = time.time()
preds = rf.predict(test)
#END TIMER PREDICTION
end = time.time()
print('ELAPSE TIME PREDICTION: ',(end - start)/60, 'min')

print('------------------------------------------------------------------------------')
#------------------------------------------------------------------------------
pred_label = preds
confusion_matrix = pd.crosstab(labels_test,  pred_label , rownames=['Actual ALERT'], colnames = ['Predicted ALERT'])
with open(output_file_name, "a") as f: print(confusion_matrix, file = f)

FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)

TP_total = sum(TP)
TN_total = sum(TN)
FP_total = sum(FP)
FN_total = sum(FN)

TP_total = np.array(TP_total,dtype=np.float64)
TN_total = np.array(TN_total,dtype=np.float64)
FP_total = np.array(FP_total,dtype=np.float64)
FN_total = np.array(FN_total,dtype=np.float64)

Acc = ACC(TP_total,TN_total, FP_total, FN_total)
Precision = PRECISION(TP_total, FP_total)
Recall = RECALL(TP_total, FN_total)
F1 = F1(Recall,Precision)
BACC = BACC(TP_total,TN_total, FP_total, FN_total)
MCC = MCC(TP_total,TN_total, FP_total, FN_total)

with open(output_file_name, "a") as f: print('Accuracy: ', Acc, file =f)
print('Precision: ', Precision )
print('Recall: ', Recall )
print('F1: ', F1 )
print('BACC: ', BACC)
print('MCC: ', MCC)
for i in range(0,len(TP)):
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    print('Accuracy: ', label[i] ,' - ' , Acc)
y_test = labels_test
y_score = rf.predict_proba(test)
#binarize the output
y_test_bin = label_binarize(y_test,classes = [0,1,2])
n_classes = y_test_bin.shape[1]
tmp, y_labels = pd.factorize(labels_test)
label = y_labels
print('AUC_ROC total: ', roc_auc_score(labels_test, rf.predict_proba(test), multi_class='ovr'))



###here
if explanator == False: None 
else:
    print('---------------------------------------------------------------------------------')
    print('Generating Explainer')
    print('---------------------------------------------------------------------------------')
    print('')
    plt.clf()
    # ## Summary Bar Plot Global
    explainer = shap.TreeExplainer(rf)
    start_index = 0
    end_index = samples
    shap_values = explainer.shap_values(test[start_index:end_index])
    shap_obj = explainer(test[start_index:end_index])
    # shap.summary_plot(shap_values = shap_values,
    #                   features = test[start_index:end_index],
    #                  class_names=[label[0],label[1],label[2],label[3],label[4],label[5],label[6]],show=False)
    # plt.savefig('RF_Shap_Summary_global_cicids.png')

    vals= np.abs(shap_values).mean(1)

    feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    feature_importance.head()
    print(feature_importance.to_string())

    print('---------------------------------------------------------------------------------')
    # feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
    feature_val = feature_importance['feature_importance_vals'].tolist()

    # col_name = 'col_name'  # Replace with the name of the column you want to extract
    feature_name = feature_importance['col_name'].tolist()


    # for item1, item2 in zip(feature_name, feature_val):
    #     print(item1, item2)


    # Use zip to combine the two lists, sort based on list1, and then unzip them
    zipped_lists = list(zip(feature_name, feature_val))
    zipped_lists.sort(key=lambda x: x[1],reverse=True)

    # Convert the sorted result back into separate lists
    sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

    for k in sorted_list1:
        with open(output_file_name, "a") as f:print("df.pop('",k,"')", sep='',file = f)

    print('---------------------------------------------------------------------------------')



    print('---------------------------------------------------------------------------------')
    print('Generating Sparsity Graph')
    print('---------------------------------------------------------------------------------')
    print('')


    # Find the minimum and maximum values in the list
    min_value = min(feature_val)
    max_value = max(feature_val)

    # Normalize the list to the range [0, 1]
    normalized_list = [(x - min_value) / (max_value - min_value) for x in feature_val]

    # print(feature_name,normalized_list,'\n')
    # for item1, item2 in zip(feature_name, normalized_list):
    #     print(item1, item2)

    #calculating Sparsity

    # Define the threshold
    threshold = 1e-10

    # Initialize a count variable to keep track of values below the threshold
    count_below_threshold = 0

    # Iterate through the list and count values below the threshold
    for value in normalized_list:
        if value < threshold:
            count_below_threshold += 1

    Sparsity = count_below_threshold/len(normalized_list)
    Spar = []
    print('Sparsity = ',Sparsity)
    X_axis = []
    #----------------------------------------------------------------------------
    for i in range(0, 11):
        i/10
        threshold = i/10
        for value in normalized_list:
            if value < threshold:
                count_below_threshold += 1

        Sparsity = count_below_threshold/len(normalized_list)
        Spar.append(Sparsity)
        X_axis.append(i/10)
        count_below_threshold = 0


    #---------------------------------------------------------------------------
    with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)

    with open(output_file_name, "a") as f:print('y_axis_RF = ', Spar ,'',file = f)
    with open(output_file_name, "a") as f:print('x_axis_RF = ', X_axis ,'',file = f)

    plt.clf()

    # Create a plot
    plt.plot(X_axis, Spar, marker='o', linestyle='-')

    # Set labels for the axes
    plt.xlabel('X-Axis')
    plt.ylabel('Y-Axis')

    # Set the title of the plot
    plt.title('Values vs. X-Axis')

    # Show the plot
    # plt.show()
    plt.savefig('sparsity_RF_LIME.png')
    plt.clf()