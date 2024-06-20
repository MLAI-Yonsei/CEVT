import wandb
import math
from datetime import datetime

wandb.login()
# 프로젝트와 엔터티 이름 설정
project_name = "causal-effect-vae"
entity_name = "mlai_medical_ai"

# WandB API 객체 생성
api = wandb.Api()

timestamp_threshold = datetime.fromisoformat('2023-09-18T00:00:00+00:00').timestamp()
# timestamp_upper = datetime.fromisoformat('2023-09-18T00:00:00+00:00').timestamp()
# 프로젝트의 모든 실행을 불러옴
runs = api.runs(f"{entity_name}/{project_name}")
# 각 실행에서 특정 두 값의 합을 계산하고 저장
run_sums = []

for run in runs:
    if isinstance(run, wandb.apis.public.Run):
        if run.state in ['finished', 'success']:
            elbo_lambda1 = run.config.get("elbo_lambda1", None)
            lambda1 = run.config.get("lambda1", None)
            created_at = datetime.fromisoformat(run.created_at.replace("Z", "+00:00")).timestamp()  # created_at을 Unix timestamp로 변환
            # if timestamp_upper > created_at > timestamp_threshold:
            if created_at > timestamp_threshold and elbo_lambda1 == 1 and lambda1 == 1:
                metric1 = run.summary.get("best_val_d_mae_loss", None)
                metric2 = run.summary.get("best_val_y_mae_loss", None)
                if metric1 is not None and metric2 is not None and not (math.isnan(metric1) or math.isnan(metric2)):
                    run_sums.append((run, metric1 + metric2))

# 합이 가장 낮은 순으로 정렬
sorted_runs = sorted(run_sums, key=lambda x: x[1])

# 가장 작은 값을 가진 run을 가져옴
best_run = sorted_runs[0][0]

# 해당 run의 다른 값을 출력
print(f"Best Run ID: {best_run.name}")
print(f"Created At: {best_run.created_at}")
print("Other values:")
for key, value in best_run.summary.items():
    if key in [ 'best_test_d_mae_loss', 'best_test_d_rmse_loss', 'best_test_y_mae_loss', 'best_test_y_rmse_loss']:
        print(f"{key}: {value:.4f}")
    # if key in ['best_test_ceelbo_loss', 'best_test_d_mae_loss', 'best_test_d_rmse_loss', 'best_test_y_mae_loss', 'best_test_y_rmse_loss', 'best_train_ceelbo_loss', 'best_train_d_mae_loss', 'best_train_d_rmse_loss', 'best_train_y_mae_loss', 'best_train_y_rmse_loss', 'best_val_ceelbo_loss', 'best_val_d_mae_loss', 'best_val_d_rmse_loss', 'best_val_y_mae_loss', 'best_val_y_rmse_loss']:
    #     print(f"{key}: {value:.4f}")
    # print(f"{key}: {value}")
print("\n")
# # 해당 run의 config 값을 테이블 형태로 출력합니다.
print("\nBest Run Config:")
print("-" * 30)
print(f"{'Config Key':<15} | {'Config Value':<15}")
print("-" * 30)
for key, value in best_run.config.items():
    print(f"{key:<15} | {value:<15}") 