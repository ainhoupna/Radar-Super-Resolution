import mlflow
from mlflow.tracking import MlflowClient

def update_best_metrics():
    client = MlflowClient()
    experiments = client.search_experiments()
    
    for exp in experiments:
        print(f"Checking experiment: {exp.name} (ID: {exp.experiment_id})")
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        
        for run in runs:
            run_id = run.info.run_id
            metrics_history = client.get_metric_history(run_id, "val_psnr")
            
            if metrics_history:
                best_psnr = max(m.value for m in metrics_history)
                print(f"  Updating run {run_id} ({run.data.tags.get('mlflow.runName', 'unnamed')}): best_val_psnr = {best_psnr:.4f}")
                
                # Log the best metric. We use a step that doesn't conflict or just log it once.
                # MLflow metrics are usually logged with a timestamp and step.
                client.log_metric(run_id, "best_val_psnr", best_psnr)

if __name__ == "__main__":
    update_best_metrics()
