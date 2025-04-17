import pandas as pd

# Load the dataset
df = pd.read_csv('250122_인천시데이터.csv')
# confirm_date를 datetime 형식으로 변환
df['confirm_date'] = pd.to_datetime(df['confirm_date'])

# Step 1: 기존 필터링 방법 적용
filtered_cluster_ids = []
for cluster_id in df['transmission_cluster'].unique():
    cluster = df[df['transmission_cluster'] == cluster_id]
    min_date = cluster['confirm_date'].min()
    four_days_later = min_date + pd.Timedelta(days=4)

    # 첫날 이후 4일 이내에 새로운 데이터가 있는지 확인
    mask = (cluster['confirm_date'] > min_date) & (cluster['confirm_date'] <= four_days_later)

    # 4일 내 데이터가 있거나, 클러스터에 데이터가 첫날만 있는 경우 유지
    if cluster[mask].shape[0] > 0 or cluster.shape[0] == 1:
        filtered_cluster_ids.append(cluster_id)

# 필터링된 데이터셋 생성
filtered_data = df[df['transmission_cluster'].isin(filtered_cluster_ids)].copy()

# Step 2: 클러스터 크기가 192 이상인 것 제거
cluster_sizes = filtered_data.groupby('transmission_cluster').size().reset_index(name='row_count')
clusters_to_remove = cluster_sizes[cluster_sizes['row_count'] >= 100]['transmission_cluster'].unique()

filtered_data = filtered_data[~filtered_data['transmission_cluster'].isin(clusters_to_remove)]

# Step 3: y, d 할당
all_data = []

for orig_cluster_id in sorted(filtered_data['transmission_cluster'].unique()):
    cluster = filtered_data[filtered_data['transmission_cluster'] == orig_cluster_id].copy()
    cluster = cluster.sort_values('confirm_date').reset_index(drop=True)

    min_date = cluster['confirm_date'].min()
    max_date = cluster['confirm_date'].max()

    cluster['diff_days'] = (cluster['confirm_date'] - min_date).dt.days 
    cluster['gender'] = cluster['gender'].map({'남': 1, '여': 0})

    cut_step = 1
    current_cut_date = min_date

    while cut_step <= 5 and current_cut_date <= max_date:
        subset = cluster[cluster['confirm_date'] <= current_cut_date].copy()
        if subset.empty:
            break

        future_cases = cluster[cluster['confirm_date'] > current_cut_date]
        y_count = len(future_cases)
        d_value = (future_cases['confirm_date'].max() - current_cut_date).days if not future_cases.empty else 0

        subset['cut_date'] = cut_step
        subset['y'] = y_count
        subset['d'] = d_value

        all_data.append(subset)

        current_cut_date += pd.Timedelta(days=1)
        cut_step += 1

        if y_count == 0 and d_value == 0:
            break

final_df = pd.concat(all_data, ignore_index=True).drop('confirm_date', axis=1)

# Step 4: transmission_cluster 재할당 (Re-indexing)
unique_clusters = final_df['transmission_cluster'].unique()
cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_clusters), start=1)}
final_df = final_df.sort_values(['transmission_cluster', 'cut_date'])
final_df['transmission_cluster'] = final_df.groupby(['transmission_cluster', 'cut_date']).ngroup()

# 최종 결과 확인
print(final_df.head(10))


final_df.to_csv('preprocessed_incheon.csv', index=False)