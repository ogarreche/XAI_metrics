
###################################################
#               Parameter Setting                #
###################################################

fraction= 0.25 # how much of that database you want to use
frac_normal = .2 #how much of the normal classification you want to reduce
split = 0.70 # how you want to split the train/test data (this is percentage fro train)

#Model Parameters
n_split= 3
n_repeat= 15

# XAI Samples
samples = 300

# Specify the name of the output text file
output_file_name = "LGBM_LIME_CIC_output.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)

###################################################
###################################################
###################################################

print('-------------------------------------------------')
print('LightGBM CICIDS')
print('--------------------------------------------------')
print('Importing Libraries')
print('--------------------------------------------------')

# Makes sure we see all columns

from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
import time
import shap
from scipy.special import softmax
np.random.seed(0)
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score, accuracy_score
from interpret import show
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix


from sklearn.metrics import roc_auc_score
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
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
        auc_avg += auc(fpr[i], tpr[i])
        counting = i+1
    return auc_avg/counting

print('Selecting Column Features')
print('--------------------------------------------------')


########################################### CICIDS ###########################################################

# Select which feature method you want to use by uncommenting it.
'''
########################################### CICIDS Features ########################################
'''

'''
all features
'''

req_cols = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',' Idle Std',' Idle Max',' Idle Min',' Label']


# Choose features to import + label
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

#frames = [df0, df1, df2, df3, df4, df5, df6, df7]
frames = [df0,df1, df2, df3, df4, df5, df6, df7]
df = pd.concat(frames,ignore_index=True)
y = df.pop(' Label')
df = df.assign(Label = y)



print('---------------------------------------------------------------------------------')
print('Removing top features for acc')
print('---------------------------------------------------------------------------------')
# df.pop(' Bwd IAT Min')
# df.pop(' Bwd Packet Length Mean')
# df.pop(' Bwd Packet Length Min')
# df.pop(' Bwd Header Length')
# df.pop(' CWE Flag Count')
# df.pop(' Bwd PSH Flags')
# df.pop(' Bwd Packets/s')
# df.pop(' Packet Length Std')
# df.pop(' Bwd URG Flags')
# df.pop(' Packet Length Mean')
# df.pop(' Subflow Bwd Packets')
# df.pop(' Fwd IAT Max')
# df.pop(' RST Flag Count')
# df.pop(' Bwd IAT Max')
# df.pop(' Subflow Bwd Bytes')
# df.pop(' Min Packet Length')
# df.pop(' Subflow Fwd Bytes')
# df.pop(' Fwd Packet Length Min')
# df.pop(' Bwd Packet Length Std')
# df.pop('Active Mean')
# df.pop(' Fwd IAT Mean')
# df.pop('Init_Win_bytes_forward')
# df.pop(' Total Length of Bwd Packets')
# df.pop(' URG Flag Count')
# df.pop(' Idle Max')
# df.pop(' Packet Length Variance')
# df.pop(' Max Packet Length')
# df.pop(' Init_Win_bytes_backward')
# df.pop(' PSH Flag Count')
# df.pop('Idle Mean')
# df.pop(' Fwd URG Flags')
# df.pop(' min_seg_size_forward')
# df.pop(' SYN Flag Count')
# df.pop('Subflow Fwd Packets')
# df.pop(' Idle Std')
# df.pop('Fwd IAT Total')
# df.pop(' Total Backward Packets')
# df.pop(' Flow Packets/s')
# df.pop(' Bwd IAT Mean')
# df.pop(' Fwd IAT Std')
# df.pop(' Fwd Packet Length Max')
# df.pop(' Fwd Packet Length Std')
# df.pop('FIN Flag Count')
# df.pop(' act_data_pkt_fwd')
# df.pop(' ACK Flag Count')
# df.pop(' Idle Min')
# df.pop(' Active Min')
# df.pop(' Fwd Header Length')
# df.pop(' Fwd Packet Length Mean')
# df.pop(' Active Std')
# df.pop(' Total Fwd Packets')
# df.pop(' Down/Up Ratio')
# df.pop(' Fwd IAT Min')
# df.pop(' Flow IAT Std')
# df.pop('Bwd Packet Length Max')
# df.pop(' Fwd Avg Packets/Bulk')
# df.pop(' Flow IAT Max')
# df.pop(' Active Max')
# df.pop('Bwd Avg Bulk Rate')
# df.pop(' Flow IAT Min')
# df.pop(' Flow Duration')
# df.pop(' Flow IAT Mean')
# df.pop(' Fwd Avg Bulk Rate')
# df.pop(' Destination Port')
# df.pop('Bwd IAT Total')
# df.pop('Flow Bytes/s')
# df.pop('Fwd PSH Flags')
# df.pop(' ECE Flag Count')
# df.pop(' Bwd Avg Packets/Bulk')
# df.pop('Fwd Packets/s')
# df.pop(' Avg Fwd Segment Size')
# df.pop('Fwd Avg Bytes/Bulk')
# df.pop(' Average Packet Size')
# df.pop(' Bwd IAT Std')
# df.pop(' Bwd Avg Bytes/Bulk')
# df.pop(' Avg Bwd Segment Size')

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

print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')

''' 
Relabeling the attacks
'''

df_max_scaled = df.copy()
y = df_max_scaled['Label'].replace({'DDoS':'Dos/Ddos','DoS GoldenEye': 'Dos/Ddos', 'DoS Hulk': 'Dos/Ddos', 'DoS Slowhttptest': 'Dos/Ddos', 'DoS slowloris': 'Dos/Ddos', 'Heartbleed': 'Dos/Ddos','FTP-Patator': 'Brute Force', 'SSH-Patator': 'Brute Force','Web Attack - Brute Force': 'Web Attack', 'Web Attack - Sql Injection': 'Web Attack', 'Web Attack - XSS': 'Web Attack'})
df_max_scaled.pop('Label')
df_max_scaled

'''
Normalizing database
'''
print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')


for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    #print (t)
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
df = df_max_scaled.assign( Label = y)
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
"""
df = df.assign(Label = y)



print('---------------------------------------------------------------------------------')
print('Counting labels')
print('---------------------------------------------------------------------------------')
print('')

y = df.pop('Label')
X = df
"""
# # Oversampling and balancing test data
'''
counter = Counter(y)
print(counter)
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        X, y = oversample(X,y)

counter = Counter(y)
print('test len ', counter)

# count class distribution
counter = Counter(y)
print(counter)

df = X.assign( ALERT = y)

#Remove duplicates
df = df.drop_duplicates()

y = df.pop('ALERT')
X = df

df = df.assign( ALERT = y)
# summarize class distribution
counter = Counter(y)

print('after removing duplicates:',counter)
'''
# ## Train and Test split

print('---------------------------------------------------------------------------------')
print('Separating Training and Testing db')
print('---------------------------------------------------------------------------------')
print('')

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, y, train_size=split)
df = X.assign( ALERT = y)

# extracting label name
notused, y_labels = pd.factorize(y)
# Transforming numpy format list
y_labels = list(y_labels)

"""
print('---------------------------------------------------------------------------------')
print('Balance Datasets')
print('---------------------------------------------------------------------------------')
print('')

counter = Counter(labels_train)
print(counter)
"""
# call balance operation until all labels have the same size
'''
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        train, labels_train = oversample(train, labels_train)

counter = Counter(labels_train)
print('train len',counter)

# # After OverSampling training dataset

train = train.assign( ALERT = labels_train)
'''
#Drop ALert column from train
#train.pop('ALERT')


labels_train_number, labels_train_label = pd.factorize(labels_train)
labels_test_number, labels_test_label = pd.factorize(labels_test)
label = labels_train_label


# # Oversampling and balancing test data
'''
counter = Counter(labels_test)
print(counter)
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        test, labels_test = oversample(test, labels_test)

counter = Counter(labels_test)
print('test len ', counter)
'''
#joining features and label
test = test.assign(ALERT = labels_test)
'''
#Randomize df order
test = test.sample(frac = 1)
'''
#Drop label column
labels_test = test.pop('ALERT')

print('Defining the model')
print('--------------------------------------------------')
print('Training the model')
print('------------------------------------------------------------------------------')
#START TIMER MODEL
start = time.time()

# evaluate the model
model = LGBMClassifier()
cv = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=1)
n_scores = cross_val_score(model, train, labels_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# fit the model on the whole dataset
model = LGBMClassifier()
model.fit(train, labels_train)

#END TIMER MODEL
end = time.time()
print('ELAPSE TIME MODEL: ',(end - start)/60, 'min')
print('------------------------------------------------------------------------------')
#------------------------------------------------------------------------------
# Get the predicted class for each observation

start = time.time()

y_pred = model.predict(test)
print('prediction',y_pred)

p, label2 = pd.factorize(y_pred)
end = time.time()
print('ELAPSE TIME MODEL: ',(end - start)/60, 'min')

print('---------------------------------------------------------------------------------')
print('Generating Confusion Matrix')
print('---------------------------------------------------------------------------------')
print('')

print(y_pred)
print(labels_test)
pred_label = y_pred

confusion_matrix = pd.crosstab(labels_test, pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
all_unique_values = sorted(set(pred_label) | set(labels_test))
z = np.zeros((len(all_unique_values), len(all_unique_values)))
rows, cols = confusion_matrix.shape
z[:rows, :cols] = confusion_matrix
confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
with open(output_file_name, "a") as f:print(confusion_matrix, file = f)

#---------------------------------------------------------------------
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)
TP_total = sum(TP)
TN_total = sum(TN)
FP_total = sum(FP)
FN_total = sum(FN)

#---------------------------------------------------------------------
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
print('---------------------------------------------------------------------------------')

with open(output_file_name, "a") as f:print('Accuracy total: ', Acc, file = f)
print('Precision total: ', Precision )
print('Recall total: ', Recall )
print('F1 total: ', F1 )
print('BACC total: ', BACC)
print('MCC total: ', MCC)

for i in range(0,len(label)):
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    print('Accuracy: ', label[i] ,' - ' , Acc)
print('rocauc is ',roc_auc_score(labels_test, model.predict_proba(test), multi_class='ovr'))

'''
y_pred =model.predict_proba(test)
label = label2
#---------------------------------------------------------------------
classes_n = []
y_test = label
for i in label: classes_n.append(i)
y_test_bin = label_binarize(labels_test,classes = classes_n)
#-------------------------------------------------------------------
'''



print('---------------------------------------------------------------------------------')
print('Generating LIME explanation')
print('---------------------------------------------------------------------------------')
print('')



# test.pop ('Label')
print('------------------------------------------------------------------------------')

#START TIMER MODEL
start = time.time()
# train =  X_train
# test = X_test
test2 = test
test = test.to_numpy()

explainer = lime.lime_tabular.LimeTabularExplainer(train.to_numpy(), feature_names= list(train.columns.values) , class_names=label.values , discretize_continuous=True)


#creating dict 
feat_list = req_cols[:-1]
# print(feat_list)

feat_dict = dict.fromkeys(feat_list, 0)
# print(feat_dict)
c = 0

num_columns = df.shape[1] - 1
feature_name = req_cols[:-1]
feature_name.sort()
# print('lista',feature_name)
feature_val = []

for i in range(0,num_columns): feature_val.append(0)

for i in range(0,samples):

# i = sample
    # exp = explainer.explain_instance(test[i], rf.predict_proba)
    
    exp = explainer.explain_instance(test[i], model.predict_proba, num_features=num_columns, top_labels=num_columns)
    # exp.show_in_notebook(show_table=True, show_all=True)
    
    #lime list to string
    lime_list = exp.as_list()
    lime_list.sort()
    # print(lime_list)
    for j in range (0,num_columns): feature_val[j]+= abs(lime_list[j][1])
    # print ('debug here',lime_list[1][1])

    # lime_str = ' '.join(str(x) for x in lime_list)
    # print(lime_str)


    #lime counting features frequency 
    # for i in feat_list:
    #     if i in lime_str:
    #         #update dict
    #         feat_dict[i] = feat_dict[i] + 1
    
    c = c + 1 
    print ('progress',100*(c/samples),'%')

# Define the number you want to divide by
divider = samples

# Use a list comprehension to divide all elements by the same number
feature_val = [x / divider for x in feature_val]

# for item1, item2 in zip(feature_name, feature_val):
#     print(item1, item2)


# Use zip to combine the two lists, sort based on list1, and then unzip them
zipped_lists = list(zip(feature_name, feature_val))
zipped_lists.sort(key=lambda x: x[1],reverse=True)

# Convert the sorted result back into separate lists
sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

# print(sorted_list1)
# print(sorted_list2)
print('----------------------------------------------------------------------------------------------------------------')

for item1, item2 in zip(sorted_list1, sorted_list2):
    print(item1, item2)

for k in sorted_list1:
  with open(output_file_name, "a") as f:print("df.pop('",k,"')", sep='', file = f)
with open(output_file_name, "a") as f:print("Trial_ =[", file = f)
for k in sorted_list1:
  with open(output_file_name, "a") as f:print("'",k,"',", sep='', file = f)
with open(output_file_name, "a") as f:print("]", file = f)
print('---------------------------------------------------------------------------------')

# # print(feat_dict)
# # Sort values in descending order
# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print(k,v)

# for k,v in sorted(feat_dict.items(), key=lambda x: x[1], reverse=True):
#   print("df.pop('",k,"')", sep='')

print('---------------------------------------------------------------------------------')


end = time.time()
with open(output_file_name, "a") as f:print('ELAPSE TIME LIME GLOBAL: ',(end - start)/60, 'min', file = f)

print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Generating Sparsity Graph')
print('---------------------------------------------------------------------------------')
print('')
# print(feature_importance)

# feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
feature_val = sorted_list2

# col_name = 'col_name'  # Replace with the name of the column you want to extract
feature_name = sorted_list1

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

with open(output_file_name, "a") as f:print('y_axis_RF = ', Spar ,'', file = f)
with open(output_file_name, "a") as f:print('x_axis_RF = ', X_axis ,'', file = f)

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