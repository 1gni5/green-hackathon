import typer
import pandas as pd
import papermill as pm

app = typer.Typer()


@app.command()
def run_notebook(csv_path: str, notebook_path: str):
    """
    Run a notebook with different parameters specified in a CSV file.

    Parameters:
    - csv_path: Path to the CSV file containing parameter values.
    - notebook_path: Path to the notebook to be executed.
    """
    # Load parameters from CSV
    parameters_df = pd.read_csv(csv_path)
    parameters_list = parameters_df.to_dict(orient="records")

    # Execute the notebook with different parameters
    for parameters in parameters_list:
        pm.execute_notebook(
            input_path=notebook_path,
            output_path=notebook_path,
            parameters=parameters,
        )


if __name__ == "__main__":
    app()
