import pandas as pd

# Load the dataset
data = pd.read_csv('250122_인천시데이터.csv')
data['confirm_date'] = pd.to_datetime(data['confirm_date'])

# Step 1: Filter clusters without new cases within 4 days of the earliest date
filtered_cluster_ids = []
for cluster_id in data['transmission_cluster'].unique():
    cluster = data[data['transmission_cluster'] == cluster_id]
    min_date = cluster['confirm_date'].min()
    four_days_later = min_date + pd.Timedelta(days=4)
    mask = (cluster['confirm_date'] > min_date) & (cluster['confirm_date'] <= four_days_later)
    if cluster[mask].shape[0] > 0:
        filtered_cluster_ids.append(cluster_id)

data_filtered = data[data['transmission_cluster'].isin(filtered_cluster_ids)].copy()

# Step 2: Reindex clusters to consecutive integers
unique_clusters = sorted(data_filtered['transmission_cluster'].unique())
cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters, 1)}  # Start from 1
data_filtered['transmission_cluster'] = data_filtered['transmission_cluster'].map(cluster_mapping)

# Process each cluster
all_clusters_processed = []
for cluster_id in sorted(data_filtered['transmission_cluster'].unique()):
    cluster_data = data_filtered[data_filtered['transmission_cluster'] == cluster_id].copy()
    cluster_data = cluster_data.sort_values('confirm_date')
    
    # Convert gender to numeric
    cluster_data['gender'] = cluster_data['gender'].map({'남': 1, '여': 0})
    
    # Calculate days from the earliest date
    min_date = cluster_data['confirm_date'].min()
    cluster_data['diff_days'] = (cluster_data['confirm_date'] - min_date).dt.days
    
    current_cut_date = min_date
    cut_date_value = 1
    
    while cut_date_value <= 5:
        # Select data up to current_cut_date
        cut_date_data = cluster_data[cluster_data['confirm_date'] <= current_cut_date].copy()
        cut_date_data['cut_date'] = cut_date_value
        
        # Calculate y (cases after current_cut_date)
        y_count = (cluster_data['confirm_date'] > current_cut_date).sum()
        cut_date_data['y'] = y_count
        
        # Calculate d (days to last case)
        d_value = (cluster_data['confirm_date'].max() - current_cut_date).days if y_count > 0 else 0
        cut_date_data['d'] = d_value
        
        all_clusters_processed.append(cut_date_data)
        
        if y_count == 0 and d_value == 0:
            break
        
        current_cut_date += pd.Timedelta(days=1)
        cut_date_value += 1

# Combine and save
final_data = pd.concat(all_clusters_processed).drop(columns=['confirm_date'])
final_data.to_csv('not_preprocessed_clusters.csv', index=False)