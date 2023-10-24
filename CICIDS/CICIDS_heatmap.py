import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Features from SHAP for different models
shap_features = {
    'ADA_SHAP': ['destination_port', 'cwe_flag_count', 'fwd_avg_packets/bulk', 'fwd_avg_bytes/bulk', 'avg_bwd_segment_size', 'avg_fwd_segment_size', 'average_packet_size', 'down/up_ratio', 'ece_flag_count', 'urg_flag_count', 'max_packet_length', 'ack_flag_count', 'psh_flag_count', 'rst_flag_count', 'syn_flag_count', 'fin_flag_count', 'packet_length_variance', 'packet_length_std', 'fwd_avg_bulk_rate', 'bwd_avg_bytes/bulk'],
    'DNN_SHAP': ['psh_flag_count', 'ack_flag_count', 'flow_duration', 'bwd_iat_total', 'fwd_iat_total', 'destination_port', 'init_win_bytes_backward', 'idle_max', 'bwd_iat_max', 'fwd_psh_flags', 'idle_mean', 'flow_iat_max', 'init_win_bytes_forward', 'fwd_iat_std', 'urg_flag_count', 'average_packet_size', 'subflow_fwd_bytes', 'syn_flag_count', 'avg_bwd_segment_size', 'total_length_of_fwd_packets'],
    'LGBM_SHAP': ['destination_port', 'init_win_bytes_backward', 'bwd_iat_max', 'min_seg_size_forward', 'init_win_bytes_forward', 'bwd_packet/s', 'psh_flag_count', 'bwd_packet_length_std', 'fwd_iat_min', 'flow_iat_min', 'fwd_packet/s', 'flow_iat_mean', 'fwd_header_length', 'flow_iat_max', 'bwd_packet_length_mean', 'bwd_lat_std', 'total_fwd_packets', 'bwd_packet_length_min', 'flow_duration', 'packet_length_mean'],
    'SVM_SHAP': ['psh_flag_count', 'ack_flag_count', 'min_seg_size_forward', 'init_win_bytes_forward', 'init_win_bytes_backward', 'destination_port', 'urg_flag_count', 'bwd_iat_total', 'flow_duration', 'fwd_iat_max', 'flow_iat_max', 'bwd_iat_max', 'fwd_iat_total', 'idle_max', 'fin_flag_count', 'packet_length_std', 'packet_length_variance', 'fwd_psh_flags', 'syn_flag_count', 'bwd_packet_length_std'],
    'MLP_SHAP': ['bwd_packet/s', 'bwd_psh_flags', 'bwd_packet_length_mean', 'cwe_flag_count', 'bwd_urg_flags', 'bwd_iat_min', 'bwd_packet_length_min', 'bwd_packet_length_std', 'bwd_iat_mean', 'bwd_avg_bulk_rate', 'subflow_fwd_packets', 'bwd_iat_std', 'bwd_header_length', 'min_packet_length', 'fwd_iat_total', 'init_win_bytes_backward', 'idle_std', 'destination_port', 'idle_min', 'fwd_iat_mean'],
    'RF_SHAP': ['init_win_bytes_backward', 'destination_port', 'fwd_packet_length_std', 'flow_iat_max', 'total_length_of_fwd_packets', 'fwd_packet_length_max', 'flow_iat_mean', 'fwd_header_length', 'fwd_iat_std', 'total_fwd_packets', 'packet_length_mean', 'packet_length_variance', 'active_min', 'bwd_header_length', 'flow_duration', 'packet_length_std', 'bwd_packet_length_std', 'fwd_iat_total', 'bwd_packet_length_max', 'fwd_packet_length_mean'],
    'KNN_SHAP': ['init_win_bytes_backward', 'init_win_bytes_forward', 'psh_flag_count', 'down/up_ratio', 'min_seg_size_forward', 'ack_flag_count', 'fwd_iat_std', 'flow_duration', 'flow_iat_std', 'bwd_iat_total', 'fwd_iat_total', 'flow_iat_max', 'fwd_iat_max', 'destination_port', 'fwd_packet/s', 'flow_iat_mean', 'act_data_pkt_fwd', 'bwd_packet_length_max', 'bwd_header_length', 'bwd_packet/s'],
    
 
}

# Features from LIME for different models
lime_features = {
    'ADA_LIME': ['bwd_packet/s', 'bwd_psh_flags', 'bwd_packet_length_mean', 'cwe_flag_count', 'bwd_urg_flags', 'bwd_iat_min', 'bwd_packet_length_min', 'bwd_packet_length_std', 'bwd_iat_mean', 'bwd_avg_bulk_rate', 'subflow_fwd_packets', 'bwd_iat_std', 'bwd_header_length', 'min_packet_length', 'fwd_iat_total', 'init_win_bytes_backward', 'idle_std', 'destination_port', 'idle_min', 'fwd_iat_mean'],
    'DNN_LIME': ['bwd_avg_bulk_rate', 'psh_flag_count', 'active_mean', 'bwd_packet_length_max', 'idle_std', 'init_win_bytes_backward', 'subflow_bwd_packets', 'bwd_iat_total', 'min_packet_length', 'packet_length_variance', 'packet_length_std', 'fwd_iat_min', 'idle_max', 'min_seg_size_forward', 'fin_flag_count', 'fwd_iat_std', 'fwd_urg_flags', 'flow_iat_min', 'idle_min', 'fwd_iat_mean'],
    'SVM_LIME': ['ack_flag_count', 'bwd_iat_total', 'bwd_avg_bulk_rate', 'packet_length_variance', 'packet_length_std', 'active_mean', 'init_win_bytes_backward', 'bwd_packet_length_max', 'idle_std', 'min_seg_size_forward', 'fwd_iat_min', 'idle_mean', 'idle_max', 'packet_length_mean', 'psh_flag_count', 'max_packet_length', 'min_packet_length', 'subflow_bwd_bytes', 'fwd_psh_flags', 'fwd_iat_mean'],
    'MLP_LIME': ['psh_flag_count', 'destination_port', 'init_win_bytes_forward', 'ack_flag_count', 'bwd_iat_total', 'flow_iat_max', 'min_seg_size_forward', 'fwd_iat_total', 'urg_flag_count', 'flow_duration', 'init_win_bytes_backward', 'fwd_iat_max', 'bwd_iat_max', 'fwd_iat_std', 'syn_flag_count', 'bwd_packet/s', 'bwd_packet_length_std', 'packet_length_mean', 'packet_length_std', 'average_packet_size'],
    'RF_LIME': ['bwd_avg_bulk_rate', 'bwd_iat_total', 'active_mean', 'bwd_packet_length_mean', 'bwd_packet_length_max', 'min_packet_length', 'fin_flag_count', 'bwd_psh_flags', 'min_seg_size_forward', 'init_win_bytes_backward', 'packet_length_std', 'fwd_lat_max', 'idle_std', 'subflow_fwd_packets', 'packet_length_variance', 'init_win_bytes_forward', 'fwd_urg_flags', 'flow_byte/s', 'fwd_iat_min', 'syn_flag_count'],
    'KNN_LIME': ['idle_max', 'fwd_packet_length_max', 'bwd_iat_total', 'fwd_urg_flags', 'fwd_packet_length_min', 'bwd_psh_flags', 'ack_flag_count', 'max_packet_length', 'fwd_header_length', 'bwd_avg_bulk_rate', 'fwd_packet_length_std', 'bwd_packet_length_min', 'subflow_bwd_packets', 'fwd_iat_total', 'active_std', 'cwe_flag_count', 'total_backward_packets', 'packet_length_std', 'idle_std', 'psh_flag_count'],
}

# Combine all model names
all_models = list(shap_features.keys()) + list(lime_features.keys())

# Compute the number of common features between each pair of models
common_features_matrix = pd.DataFrame(
    [[len((set(shap_features.get(row, [])) & set(lime_features.get(col, []))) |
        (set(shap_features.get(col, [])) & set(lime_features.get(row, []))) |
        (set(shap_features.get(row, [])) & set(shap_features.get(col, []))) |
        (set(lime_features.get(row, [])) & set(lime_features.get(col, []))))
    for col in all_models] for row in all_models],
    index=all_models,
    columns=all_models
)

# Plot a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(common_features_matrix, annot=True, cmap="Blues", fmt="d", cbar=True)
plt.title('Number of Common Features Between SHAP and LIME by Model')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
