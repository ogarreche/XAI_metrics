
###################################################
#               Parameter Setting                #
###################################################

fraction= 0.1 # how much of that database you want to use
frac_normal = .2 #how much of the normal classification you want to reduce
split = 0.70 # how you want to split the train/test data (this is percentage fro train)

#Model Parameters
n_neighbors=5

# XAI Samples
samples = 50

# Specify the name of the output text file
output_file_name = "KNN_SHAP_CIC.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)

###################################################
###################################################
###################################################
print('---------------------------------------------------------------------------------')
print('KNN')
print('---------------------------------------------------------------------------------')
print('Importing libraries ')
print('---------------------------------------------------------------------------------')

#----------------------------------------------------------------------
import tensorflow as tf
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from matplotlib import pyplot as plt
import numpy as np
# import pafy
import pandas as pd
#import csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
#from keras.utils import pad_sequences
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
np.random.seed(0)
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import lime

#----------------------------------------------------------------------
#---------------------------------------------------------------------
print('---------------------------------------------------------------------------------')
print('Defining Metrics')
print('---------------------------------------------------------------------------------')


def ACC(TP,TN,FP,FN):
    Acc = (TP+TN)/(TP+FP+FN+TN)
    return Acc
def ACC_2 (TP, FN):
    ac = (TP/(TP+FN))
    return ac
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




#-----------------------------------------------------------------------
#----------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('Defining features')
print('---------------------------------------------------------------------------------')
'''
########################################### CICIDS Features ########################################
'''

# Select which feature method you want to use by uncommenting it.

'''
all features
'''
req_cols = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',' Idle Std',' Idle Max',' Idle Min',' Label']

#Load Databases from csv file
print('---------------------------------------------------------------------------------')
print('loading databases')
print('---------------------------------------------------------------------------------')
#Denial of Service
fraction = fraction
df0 = pd.read_csv ('cicids_db/Wednesday-workingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)

df1 = pd.read_csv ('cicids_db/Tuesday-WorkingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df2 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df3 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df4 = pd.read_csv ('cicids_db/Monday-WorkingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df5 = pd.read_csv ('cicids_db/Friday-WorkingHours-Morning.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df6 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df7 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)



#frames = [df0, df1, df2, df3, df4, df5, df6, df7]
frames = [df1, df2, df3, df4,df5, df6, df7]

df = pd.concat(frames,ignore_index=True)

#---------------------------------------------------------------------------------------------------#----------------------------------------------------------------


print('---------------------------------------------------------------------------------')
print('Removing top features for acc')
print('---------------------------------------------------------------------------------')

# df.pop(' ACK Flag Count')
# df.pop('Init_Win_bytes_forward')
# df.pop(' PSH Flag Count')
# df.pop(' min_seg_size_forward')
# # df.pop('Bwd IAT Total')
# df.pop(' Average Packet Size')
# df.pop(' Destination Port')
# df.pop(' Bwd Packet Length Std')
# df.pop(' Flow Duration')
# df.pop(' Packet Length Std')
# df.pop('Fwd IAT Total')
# df.pop(' Packet Length Mean')
# df.pop(' Init_Win_bytes_backward')
# df.pop(' URG Flag Count')
# df.pop('Bwd Packet Length Max')
# df.pop(' Fwd IAT Max')
# df.pop(' Flow IAT Max')
# df.pop(' Avg Bwd Segment Size')
# df.pop(' SYN Flag Count')
# df.pop(' Bwd Packet Length Mean')
# df.pop(' Idle Max')
# df.pop(' Bwd IAT Max')
# df.pop('Fwd PSH Flags')
# df.pop(' Bwd Packets/s')
# df.pop(' Fwd IAT Std')
# df.pop(' Idle Std')
# df.pop(' Flow IAT Std')
# df.pop(' Packet Length Variance')
# df.pop(' Down/Up Ratio')
# df.pop('Idle Mean')
# df.pop(' Idle Min')
# df.pop('Fwd Packets/s')
# df.pop(' Max Packet Length')
# df.pop(' Fwd IAT Mean')
# df.pop(' Fwd Packet Length Min')
# df.pop(' Fwd Header Length')
# df.pop(' Fwd Packet Length Max')
# df.pop(' Bwd Packet Length Min')
# df.pop('Subflow Fwd Packets')
# df.pop(' Bwd IAT Std')
# df.pop(' Bwd Header Length')
# df.pop('Active Mean')
df.pop(' Flow IAT Mean')
df.pop(' Subflow Fwd Bytes')
df.pop(' Bwd IAT Min')
df.pop(' Fwd IAT Min')
df.pop(' Flow IAT Min')
df.pop(' Active Min')
df.pop('Total Length of Fwd Packets')
df.pop(' Subflow Bwd Bytes')
df.pop(' Avg Fwd Segment Size')
df.pop(' Active Std')
df.pop(' Fwd Packet Length Mean')
df.pop(' Active Max')
df.pop(' Subflow Bwd Packets')
df.pop(' Total Fwd Packets')
df.pop(' Total Length of Bwd Packets')
df.pop(' act_data_pkt_fwd')
df.pop(' Bwd IAT Mean')
df.pop(' Min Packet Length')
df.pop(' Total Backward Packets')
df.pop(' Fwd Packet Length Std')
df.pop('Bwd Avg Bulk Rate')
df.pop(' CWE Flag Count')
df.pop(' Bwd URG Flags')
df.pop('FIN Flag Count')
df.pop(' RST Flag Count')
df.pop(' Flow Packets/s')
df.pop('Flow Bytes/s')
df.pop(' ECE Flag Count')
df.pop(' Bwd Avg Packets/Bulk')
df.pop(' Fwd URG Flags')
df.pop('Fwd Avg Bytes/Bulk')
df.pop(' Fwd Avg Packets/Bulk')
df.pop(' Fwd Avg Bulk Rate')
df.pop(' Bwd Avg Bytes/Bulk')
df.pop(' Bwd PSH Flags')

#---------------------------------------------------------------------
y = df.pop(' Label')
df = df.assign(Label = y)


print('---------------------------------------------------------------------------------')
print('Reducing Normal rows')
print('---------------------------------------------------------------------------------')
print('')


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


df = df.assign(Label = y)
print('train len',counter)

y = df.pop('Label')
X = df

df = df.assign(Label = y)

print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Spliting the db in training and testing')
print('---------------------------------------------------------------------------------')

X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split)
df = X.assign( Label = y)

# Separate features and labels
u, label = pd.factorize(y_train)


#----------------------------------------------------------------#----------------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('Defining KNN Model')
print('---------------------------------------------------------------------------------')

knn_clf=KNeighborsClassifier(n_neighbors=n_neighbors)

print('---------------------------------------------------------------------------------')
print('Training Model')
print('---------------------------------------------------------------------------------')

start = time.time()
knn_clf.fit(X_train,y_train)
end = time.time()

print('---------------------------------------------------------------------------------')
print('ELAPSE TIME MODEL TRAINING: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')

start = time.time()
ypred=knn_clf.predict(X_test) #These are the predicted output value
end = time.time()

print('---------------------------------------------------------------------------------')
print('ELAPSE TIME MODEL PREDICTION: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')

#y_pred = knn_clf.predict_proba(X_test)
#----------------------------------------------------------------#----------------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('CONFUSION MATRIX')
print('---------------------------------------------------------------------------------')

pred_label = ypred
#pred_label = label[ypred]

confusion_matrix = pd.crosstab(y_test, pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
all_unique_values = sorted(set(pred_label) | set(y_test))
z = np.zeros((len(all_unique_values), len(all_unique_values)))
rows, cols = confusion_matrix.shape
z[:rows, :cols] = confusion_matrix
confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
confusion_matrix.to_csv('DNN_conf_matrix.csv')
with open(output_file_name, "a") as f:print(confusion_matrix,file = f)

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



#----------------------------------------------------------------#----------------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('METRICS')
print('---------------------------------------------------------------------------------')

Acc = ACC(TP_total,TN_total, FP_total, FN_total)
Precision = PRECISION(TP_total, FP_total)
Recall = RECALL(TP_total, FN_total)
F1 = F1(Recall,Precision)
BACC = BACC(TP_total,TN_total, FP_total, FN_total)
MCC = MCC(TP_total,TN_total, FP_total, FN_total)
for i in range(0,len(TP)):
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    print('Accuracy: ', label[i] ,' - ' , Acc)



with open(output_file_name, "a") as f:print('Accuracy total: ', Acc, file = f)
print('Precision total: ', Precision )
print('Recall total: ', Recall )
print('F1 total: ', F1 )
print('BACC total: ', BACC)
print('MCC total: ', MCC)
#----------------------------------------
y_test_bin = label_binarize(y_test,classes = [0,1,2,3,4,5,6])
n_classes = y_test_bin.shape[1]

y_pred,l = pd.factorize(ypred)
try:
    print('AUC ROC:  ', roc_auc_score(y_test_bin,y_pred.reshape(-1,1), multi_class='ovr'))
except: 
    print('rocauc is nan')


# print('---------------------------------------------------------------------------------')
# print('Generating Explainer')
# print('---------------------------------------------------------------------------------')
# print('')
# plt.clf()
# test = X_test
# train = X_train
# start_index = 0
# end_index = samples

# explainer = shap.KernelExplainer(knn_clf.predict_proba, test[start_index:end_index])
# shap_values = explainer.shap_values(test[start_index:end_index])

# # shap.summary_plot(shap_values = shap_values,features = test[start_index:end_index],
# # class_names=[label[0],label[1],label[2],label[3],label[4],label[5],label[6]],show=False)
# # plt.savefig('KNN_Shap_Summary_global_cicids.png')
# # plt.clf()



# vals= np.abs(shap_values).mean(1)
# feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
# feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
# feature_importance.head()
# print(feature_importance.to_string())



# print('---------------------------------------------------------------------------------')
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
#   with open(output_file_name, "a") as f:print("df.pop('",k,"')", sep='', file = f)

# print('---------------------------------------------------------------------------------')




# print('---------------------------------------------------------------------------------')
# print('Generating Sparsity Graph')
# print('---------------------------------------------------------------------------------')
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

# with open(output_file_name, "a") as f:print('y_axis_RF = ', Spar ,'', file = f)
# with open(output_file_name, "a") as f:print('x_axis_RF = ', X_axis ,'', file = f)

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
