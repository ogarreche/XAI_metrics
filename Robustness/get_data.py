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

def get_and_preprocess_cicids(params):
    PROTECTED_CLASS = params.protected_class
    UNPROTECTED_CLASS = params.unprotected_class
    POSITIVE_OUTCOME = params.positive_outcome
    NEGATIVE_OUTCOME = params.negative_outcome	
    fraction = 0.5
    frac_normal = 0.3
    # req_cols = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',' Idle Std',' Idle Max',' Idle Min',' Label']
    

    req_cols = [' Init_Win_bytes_backward',
                            ' Destination Port',' Fwd Packet Length Std',
                            ' Flow IAT Max','Total Length of Fwd Packets',' Flow Duration', ' Label']

    cols = req_cols
    df0 = pd.read_csv ('cicids_db/Wednesday-workingHours.pcap_ISCX.csv', usecols=req_cols)
    df1 = pd.read_csv ('cicids_db/Tuesday-WorkingHours.pcap_ISCX.csv', usecols=req_cols)
    df2 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', usecols=req_cols)
    df3 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', usecols=req_cols)
    df4 = pd.read_csv ('cicids_db/Monday-WorkingHours.pcap_ISCX.csv', usecols=req_cols)
    df5 = pd.read_csv ('cicids_db/Friday-WorkingHours-Morning.pcap_ISCX.csv', usecols=req_cols)
    df6 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', usecols=req_cols)
    df7 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', usecols=req_cols)

    frames = [df0, df1, df2, df3, df4, df5,df6, df7]

    df = pd.concat(frames,ignore_index=True)
    df = df.sample(frac = fraction)

    y = df.pop(' Label')
    df = df.assign(Label = y)

    # print('reached')

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

    # print('reached2')

    df = df.assign(Label = y)


    X = df
    y = X['Label']

    y,label = pd.factorize(y)
    y = pd.DataFrame(y)
    label = list(label)
    print(label)
    df ['Label'] = y 
    X = X.drop(["Label"], axis=1)
    print(label.index('BENIGN'))
    # X[' Flow Duration'] = [label.index('BENIGN') if v > 10000 else label.index('Dos/Ddos')  for v in X[' Flow Duration'].values]
    X[' Flow Duration'] = [0 if v > 10000 else 1  for v in X[' Flow Duration'].values]



    y = np.array([POSITIVE_OUTCOME if p == 0 else NEGATIVE_OUTCOME for p in y.values])
    # y = np.array([POSITIVE_OUTCOME if p == 1 else NEGATIVE_OUTCOME for p in y.values])

    return X, y, cols

def get_and_preprocess_simargl(params):
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

    fraction = 0.1
    frac_normal = .2

    # req_cols = ['FLOW_DURATION_MILLISECONDS','FIRST_SWITCHED',
    #         'TOTAL_FLOWS_EXP','TCP_WIN_MSS_IN','LAST_SWITCHED',
    #         'TCP_WIN_MAX_IN','TCP_WIN_MIN_IN','TCP_WIN_MIN_OUT',
    #        'PROTOCOL','TCP_WIN_MAX_OUT','TCP_FLAGS',
    #         'TCP_WIN_SCALE_OUT','TCP_WIN_SCALE_IN','SRC_TOS',
    #         'DST_TOS','FLOW_ID','L4_SRC_PORT','L4_DST_PORT',
    #        'MIN_IP_PKT_LEN','MAX_IP_PKT_LEN','TOTAL_PKTS_EXP',
    #        'TOTAL_BYTES_EXP','IN_BYTES','IN_PKTS','OUT_BYTES','OUT_PKTS',
    #         'ALERT']


    req_cols = ['FLOW_DURATION_MILLISECONDS',
                            'L4_DST_PORT','TCP_WIN_MSS_IN',
                            'OUT_PKTS','IN_PKTS', 'ALERT']

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



    X = df
    y = X['ALERT']

    y,label = pd.factorize(y)
    y = pd.DataFrame(y)
    label = list(label)
    print(label)
    df ['ALERT'] = y 
    X = X.drop(["ALERT"], axis=1)
    print(label.index('None'))
    # X[' Flow Duration'] = [label.index('BENIGN') if v > 10000 else label.index('Dos/Ddos')  for v in X[' Flow Duration'].values]
    X['FLOW_DURATION_MILLISECONDS'] = [0 if v > 10000 else 1  for v in X['FLOW_DURATION_MILLISECONDS'].values]



    y = np.array([POSITIVE_OUTCOME if p == 0 else NEGATIVE_OUTCOME for p in y.values])
    # y = np.array([POSITIVE_OUTCOME if p == 1 else NEGATIVE_OUTCOME for p in y.values])

    cols = req_cols
    return X, y, cols

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
