import mlflow
import logging
from typing import Optional
from mlflow.tracking.client import MlflowClient
from mlflow.config import MLFLOW_SERVER_URI 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_latest_run(client: MlflowClient) -> Optional[mlflow.entities.Run]:
    """
    Fetches the latest MLflow run.

    Args:
    - client: MlflowClient instance connected to the MLflow server.

    Returns:
    - mlflow.entities.Run: Latest MLflow run object.
    """
    try:
        # Fetch latest run details
        latest_run = client.list_experiments(view_type=mlflow.tracking.ViewType.ACTIVE_ONLY, max_results=1)[0].runs[0]
        return latest_run
    except IndexError:
        logging.error("No active runs found in the MLflow server.")
        return None
    except Exception as e:
        logging.error(f"Error fetching latest run: {e}")
        return None

def generate_report(run: mlflow.entities.Run) -> str:
    """
    Generates a report based on the metrics and parameters of the given MLflow run.

    Args:
    - run: mlflow.entities.Run object representing the MLflow run.

    Returns:
    - str: Report content.
    """
    try:
        run_id = run.info.run_id
        metrics = run.data.metrics
        params = run.data.params

        report_content = f"Run ID: {run_id}\nMetrics: {metrics}\nParameters: {params}"
        return report_content
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        return ""

def write_report(report_content: str, output_file: str = "mlflow_report.txt") -> None:
    """
    Writes the report content to a file.

    Args:
    - report_content: Content of the report to write.
    - output_file: File path to write the report content. Default is "mlflow_report.txt".
    """
    try:
        with open(output_file, "w") as f:
            f.write(report_content)
        logging.info(f"Report written to {output_file}")
    except Exception as e:
        logging.error(f"Error writing report to {output_file}: {e}")

def main():
    try:
        # Initialize MLflow client
        mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
        client = mlflow.tracking.MlflowClient()

        # Fetch latest run
        latest_run = fetch_latest_run(client)
        if latest_run:
            # Generate report
            report_content = generate_report(latest_run)

            # Write report to file
            write_report(report_content)
            print("CML report generated successfully.")
    except Exception as e:
        logging.error(f"Error in main process: {e}")

if __name__ == "__main__":
    main()

