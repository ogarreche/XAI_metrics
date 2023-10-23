import pandas as pd
import numpy as np
from utils import Params

def get_and_preprocess_compas_data(params):
    """Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis
    
    Parameters
    ----------
    params : Params

    Returns
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """
    PROTECTED_CLASS = params.protected_class
    UNPROTECTED_CLASS = params.unprotected_class
    POSITIVE_OUTCOME = params.positive_outcome
    NEGATIVE_OUTCOME = params.negative_outcome

    compas_df = pd.read_csv("data/compas-scores-two-years.csv", index_col=0)
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                              (compas_df['days_b_screening_arrest'] >= -30) &
                              (compas_df['is_recid'] != -1) &
                              (compas_df['c_charge_degree'] != "O") &
                              (compas_df['score_text'] != "NA")]

    compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
    X = compas_df[['age', 'two_year_recid','c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay']]

    # if person has high score give them the _negative_ model outcome
    y = np.array([NEGATIVE_OUTCOME if score == 'High' else POSITIVE_OUTCOME for score in compas_df['score_text']])
    sens = X.pop('race')

    # assign African-American as the protected class
    X = pd.get_dummies(X)
    sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
    X['race'] = sensitive_attr

    # make sure everything is lining up
    assert all((sens == 'African-American') == (X['race'] == PROTECTED_CLASS))
    cols = [col for col in X]
    
    return X, y, cols

def get_and_preprocess_cc(params):
    """"Handle processing of Communities and Crime.  We exclude rows with missing values and predict
    if the violent crime is in the 50th percentile.

    Parameters
    ----------
    params : Params

    Returns:
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """
    PROTECTED_CLASS = params.protected_class
    UNPROTECTED_CLASS = params.unprotected_class
    POSITIVE_OUTCOME = params.positive_outcome
    NEGATIVE_OUTCOME = params.negative_outcome

    X = pd.read_csv("data/communities_and_crime_new_version.csv", index_col=0)
    
    # everything over 50th percentil gets negative outcome (lots of crime is bad)
    high_violent_crimes_threshold = 50
    y_col = 'ViolentCrimesPerPop numeric'

    X = X[X[y_col] != "?"]
    X[y_col] = X[y_col].values.astype('float32')

    # just dump all x's that have missing values 
    cols_with_missing_values = []
    for col in X:
        if len(np.where(X[col].values == '?')[0]) >= 1:
            cols_with_missing_values.append(col)    

    y = X[y_col]
    y_cutoff = np.percentile(y, high_violent_crimes_threshold)
    X = X.drop(cols_with_missing_values + ['communityname string', 'fold numeric', 'county numeric', 'community numeric', 'state numeric'] + [y_col], axis=1)

    # setup ys
    cols = [c for c in X]
    y = np.array([NEGATIVE_OUTCOME if val > y_cutoff else POSITIVE_OUTCOME for val in y])

    return X ,y, cols


def get_and_preprocess_german(params):
    """"Handle processing of German.  We use a preprocessed version of German from Ustun et. al.
    https://arxiv.org/abs/1809.06514.  Thanks Berk!

    Parameters:
    ----------
    params : Params

    Returns:
    ----------
    Pandas data frame X of processed data, np.ndarray y, and list of column names
    """
    PROTECTED_CLASS = params.protected_class
    UNPROTECTED_CLASS = params.unprotected_class
    POSITIVE_OUTCOME = params.positive_outcome
    NEGATIVE_OUTCOME = params.negative_outcome	

    X = pd.read_csv("data/german_processed.csv")
    y = X["GoodCustomer"]

    X = X.drop(["GoodCustomer", "PurposeOfLoan"], axis=1)
    X['Gender'] = [1 if v == "Male" else 0 for v in X['Gender'].values]

    y = np.array([POSITIVE_OUTCOME if p == 1 else NEGATIVE_OUTCOME for p in y.values])

    return X, y, [c for c in X] 

def get_and_preprocess_nsl(params):
    PROTECTED_CLASS = params.protected_class
    UNPROTECTED_CLASS = params.unprotected_class
    POSITIVE_OUTCOME = params.positive_outcome
    NEGATIVE_OUTCOME = params.negative_outcome	
    fraction = 0.01
    frac_normal = 0.3
    # req_cols = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',' Idle Std',' Idle Max',' Idle Min',' Label']
    

    # attach the column names to the dataset
    feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
        "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
    # KDDTrain+_2.csv & KDDTest+_2.csv are the datafiles without the last column about the difficulty score
    # these have already been removed.
    train='KDDTrain+.txt'
    test='KDDTest+.txt'

    df=pd.read_csv(train,names=feature)

    # shape, this gives the dimensions of the dataset
    print('Dimensions of the Training set:',df.shape)
    from sklearn.preprocessing import LabelEncoder
    # Apply label encoding to the categorical columns
    le = LabelEncoder()
    df['protocol_type'] = le.fit_transform(df['protocol_type'])
    df['flag'] = le.fit_transform(df['flag'])
    df['service'] = le.fit_transform(df['service'])  # uncomment if you want to keep the 'service' column
    
    df.drop(['difficulty'], axis=1, inplace=True)  # dropping the 'service' column as per your original code
    cols = df.columns.tolist()
   
    

    X = df
    y = X['label']
    labeldf=X['label']
    # change the label column
    newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                            'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                            ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                            'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
    y=newlabeldf
    # put the new label column back

    X = X.drop(["label"], axis=1)
    # X[' Flow Duration'] = [label.index('BENIGN') if v > 10000 else label.index('Dos/Ddos')  for v in X[' Flow Duration'].values]
    X['dst_host_same_srv_rate'] = [0 if v > 0.5 else 1  for v in X['dst_host_same_srv_rate'].values]

    y = np.array([POSITIVE_OUTCOME if p == 0 else NEGATIVE_OUTCOME for p in y.values])
    # y = np.array([POSITIVE_OUTCOME if p == 1 else NEGATIVE_OUTCOME for p in y.values])
    print("Save")
    print(X)
    return X, y, cols