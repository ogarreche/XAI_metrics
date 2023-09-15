

###################################################
#               Parameter Setting                #
###################################################


fraction= 0.25 # how much of that database you want to use
frac_normal = .2 #how much of the normal classification you want to reduce
split = 0.70 # how you want to split the train/test data (this is percentage fro train)

#Model Parameters

n_estimators=50
learning_rate=0.2

# XAI Samples
samples = 1000


# Specify the name of the output text file
output_file_name = "ADA_SHAP_SML.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
###################################################
###################################################
###################################################

#Importing libraries
#----------------------------------------------------------------------------------------------------------


import builtins
import os
import sys
import logging
logging.basicConfig(filename="ADA_output.log", level=logging.INFO)
# Save the current stdout (usually the terminal)
# original_stdout = sys.stdout
# Open the output file in write mode

    # Redirect stdout to the output file
    # sys.stdout = f

import pandas as pd
#Loading numpy
import numpy as np
# Setting random seed
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
#from sklearn.metrics import auc_score
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from sklearn.preprocessing import label_binarize
import time
import shap
np.random.seed(0)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import sklearn
from imblearn.over_sampling import RandomOverSampler
import lime
#---------------------------------------------------------------------------------------------------------
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


     

#----------------------------------------------------------------------------------------------------------
#Defining metric functions
def ACC(TP,TN,FP,FN):
    Acc = (TP+TN)/(TP+FP+FN+TN)
    return Acc

def PRECISION(TP,FP):
    Precision = TP/(TP+FP)
    return Precision
def RECALL(TP,FN):
    Recall = TP/(TP+FN)
    return Recall
def F1(Recall, Precision):
    F1 = 2 * Recall * Precision / (Recall + Precision)
    return F1
def BACC(TP,TN,FP,FN):
    BACC =(TP/(TP+FN)+ TN/(TN+FP))*0.5
    return BACC
def MCC(TP,TN,FP,FN):
    MCC = (TN*TP-FN*FP)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**.5)
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





#----------------------------------------------------------------------------------------------------------
#Loading Database

#Denial of Service
df0 = pd.read_csv ('sensor_db/dos-03-15-2022-15-44-32.csv', usecols=req_cols )
df1 = pd.read_csv ('sensor_db/dos-03-16-2022-13-45-18.csv', usecols=req_cols)
df2 = pd.read_csv ('sensor_db/dos-03-17-2022-16-22-53.csv', usecols=req_cols)
df3 = pd.read_csv ('sensor_db/dos-03-18-2022-19-27-05.csv', usecols=req_cols)
df4 = pd.read_csv ('sensor_db/dos-03-19-2022-20-01-53.csv', usecols=req_cols)
df5 = pd.read_csv ('sensor_db/dos-03-20-2022-14-27-54.csv', usecols=req_cols) 

#Malware
df6 = pd.read_csv ('sensor_db/malware-03-25-2022-17-57-07.csv', usecols=req_cols)

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

#Merging Database in one pandas DF

frames = [df0, df1, df2, df3, df4, df5, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]
#frames = [df2, df8, df16]
  
df = pd.concat(frames,ignore_index=True)

df = df.sample(frac=fraction)

#----------------------------------------------------------------------------------------------------------
# df.pop('TCP_WIN_SCALE_IN')
# df.pop('IN_BYTES')
# df.pop('PROTOCOL')
# df.pop('L4_DST_PORT')
# df.pop('TCP_WIN_MAX_IN')
# df.pop('TCP_FLAGS')
# df.pop('TCP_WIN_MSS_IN')
# df.pop('OUT_BYTES')
# df.pop('TCP_WIN_MIN_OUT')
# df.pop('SRC_TOS')
# df.pop('OUT_PKTS')
# df.pop('TCP_WIN_MAX_OUT')
# df.pop('FLOW_DURATION_MILLISECONDS')
# df.pop('TOTAL_FLOWS_EXP')
# df.pop('FLOW_ID')
# df.pop('LAST_SWITCHED')
# df.pop('TCP_WIN_MIN_IN')
# df.pop('FIRST_SWITCHED')
# df.pop('DST_TOS')
# df.pop('IN_PKTS')
# df.pop('TCP_WIN_SCALE_OUT')
df.pop('L4_SRC_PORT')
df.pop('MIN_IP_PKT_LEN')
df.pop('MAX_IP_PKT_LEN')
df.pop('TOTAL_PKTS_EXP')
df.pop('TOTAL_BYTES_EXP')

y = df.pop('ALERT')
X=df


counter = Counter(y)
print(counter)
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        X, y = oversample(X,y)


X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split)
df = X.assign( ALERT = y)

y_train, label = pd.factorize(y_train)

X_train = np.array(X_train)

#----------------------------------------
with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
print('Balance Datasets')
with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
print('')

y_test, labels_test_label = pd.factorize(y_test)

y = y_test

#----------------------------------------------------------------------------------------------------------
#Model Construction

abc = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)


#----------------------------------------------------------------------------------------------------------
#Running the model

#START TIMER MODEL
start = time.time()
model = abc.fit(X_train, y_train)
#END TIMER MODEL
end = time.time()
print('ELAPSE TIME MODEL: ',(end - start)/60, 'min')

#----------------------------------------------------------------------------------------------------------
#Data preprocessing

X_test = np.array(X_test)
#----------------------------------------------------------------------------------------------------------
# Model predictions 

#START TIMER PREDICTION
start = time.time()

y_pred = model.predict(X_test)

#END TIMER PREDICTION
end = time.time()
print('ELAPSE TIME PREDICTION: ',(end - start)/60, 'min')

#----------------------------------------------------------------------------------------------------------

pred_label, label3 = pd.factorize(y_pred)

#----------------------------------------------------------------------------------------------------------
# Confusion Matrix
with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
print('Generating Confusion Matrix')
with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
print('')


confusion_matrix = pd.crosstab(y_test, pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
all_unique_values = sorted(set(pred_label) | set(y_test))
z = np.zeros((len(all_unique_values), len(all_unique_values)))
rows, cols = confusion_matrix.shape
z[:rows, :cols] = confusion_matrix
confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
with open(output_file_name, "a") as f: print(confusion_matrix, file = f)


#True positives and False positives and negatives
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)
#Sum each Labels TP,TN,FP,FN in one overall measure
TP_total = sum(TP)
TN_total = sum(TN)
FP_total = sum(FP)
FN_total = sum(FN)

#data preprocessin because numbers are getting big
TP_total = np.array(TP_total,dtype=np.float64)
TN_total = np.array(TN_total,dtype=np.float64)
FP_total = np.array(FP_total,dtype=np.float64)
FN_total = np.array(FN_total,dtype=np.float64)

#----------------------------------------------------------------------------------------------------------
#Metrics measure overall
Acc = ACC(TP_total,TN_total, FP_total, FN_total)
Precision = PRECISION(TP_total, FP_total)
Recall = RECALL(TP_total, FN_total)
F1 = F1(Recall,Precision)
BACC = BACC(TP_total,TN_total, FP_total, FN_total)
MCC = MCC(TP_total,TN_total, FP_total, FN_total)
with open(output_file_name, "a") as f: print('Accuracy: ', Acc,file = f)
with open(output_file_name, "a") as f: print('Precision: ', Precision, file = f )
with open(output_file_name, "a") as f:print('Recall: ', Recall,file = f )
with open(output_file_name, "a") as f:print('F1: ', F1 ,file = f)
with open(output_file_name, "a") as f:print('BACC: ', BACC,file = f)
with open(output_file_name, "a") as f:print('MCC: ', MCC,file = f)
#Metrics for each label
for i in range(0,len(TP)):
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    with open(output_file_name, "a") as f: print('Accuracy: ', label[i] ,' - ' , Acc, file = f)

#----------------------------------------
# y_score = abc.predict_proba(X_test)
# y_test_bin = label_binarize(y_test,classes = [0,1,2])
# n_classes = y_test_bin.shape[1]
# with open(output_file_name, "a") as f:print('AUC_ROC total: ',AUC_ROC(y_test_bin,y_score),file = f)
# #----------------------------------------------------------------------------------------------------------
    



# with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
# print('Generating explainer')
# with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
# print('')
# test = X_test
# train = X_train
# # ## Summary Bar Plot Global
# start_index = 0
# end_index = samples
# # test.pop('Label')
# # test.pop('is_train')
# # print(label2)
# explainer = shap.KernelExplainer(abc.predict_proba, test[start_index:end_index])

# shap_values = explainer.shap_values(test[start_index:end_index])

# # shap.summary_plot(shap_values = shap_values,
# #                   features = test[start_index:end_index],class_names=[label[0],label[1],label[2],label[3],label[4],label[5],label[6]],show=False)
# # plt.savefig('ADA_Shap_Summary_global.png')
# plt.clf()


# vals= np.abs(shap_values).mean(1)
# feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
# feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
# feature_importance.head()
# print(feature_importance.to_string())



# with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
# # feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
# feature_val = feature_importance['feature_importance_vals'].tolist()

# # col_name = 'col_name'  # Replace with the name of the column you want to extract
# feature_name = feature_importance['col_name'].tolist()


# # for item1, item2 in zip(feature_name, feature_val):
# #     print(item1, item2)


# # Use zip to combine the two lists, sort based on list1, and then unzip them
# zipped_lists = list(zip(feature_name, feature_val))
# zipped_lists.sort(key=lambda x: x[1],reverse=True)

# # Convert the sorted result back into separate lists
# sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

# for k in sorted_list1:
#   with open(output_file_name, "a") as f: print("df.pop('",k,"')", sep='',file = f)

# with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)



# with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
# print('Generating Sparsity Graph')
# with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)
# print('')

# # Find the minimum and maximum values in the list
# min_value = min(feature_val)
# max_value = max(feature_val)

# # Normalize the list to the range [0, 1]
# normalized_list = [(x - min_value) / (max_value - min_value) for x in feature_val]

# # print(feature_name,normalized_list,'\n')
# # for item1, item2 in zip(feature_name, normalized_list):
# #     print(item1, item2)

# #calculating Sparsity

# # Define the threshold
# threshold = 1e-10

# # Initialize a count variable to keep track of values below the threshold
# count_below_threshold = 0

# # Iterate through the list and count values below the threshold
# for value in normalized_list:
#     if value < threshold:
#         count_below_threshold += 1

# Sparsity = count_below_threshold/len(normalized_list)
# Spar = []
# print('Sparsity = ',Sparsity)
# X_axis = []
# #----------------------------------------------------------------------------
# for i in range(0, 11):
#     i/10
#     threshold = i/10
#     for value in normalized_list:
#         if value < threshold:
#             count_below_threshold += 1

#     Sparsity = count_below_threshold/len(normalized_list)
#     Spar.append(Sparsity)
#     X_axis.append(i/10)
#     count_below_threshold = 0


# #---------------------------------------------------------------------------

# with open(output_file_name, "a") as f: print('y_axis_RF = ', Spar ,'', file = f)
# with open(output_file_name, "a") as f: print('x_axis_RF = ', X_axis ,'',file = f)

# plt.clf()

# # Create a plot
# plt.plot(X_axis, Spar, marker='o', linestyle='-')

# # Set labels for the axes
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')

# # Set the title of the plot
# plt.title('Values vs. X-Axis')

# # Show the plot
# # plt.show()
# plt.savefig('sparsity.png')
# plt.clf()
# with open(output_file_name, "a") as f: print('---------------------------------------------------------------------------------', file = f)