## Pipeline for Evaluation

The evaluation pipeline is a set of scripts that are used to evaluate the performance of the method. The pipeline is composed of the following steps:
1. Update .env file with the necessary information.

The .env file contains the necessary information to run the evaluation pipeline. The file should contain the following information:
```
# .env file
INTERWEB_APIKEY=api_key
CRAG_MOCK_API_URL=url
```

2. Check the data path and result output in main.py
```
# main.py
dataset_path = "example_data/crag_task_1_dev_v4_release.jsonl.bz2"
.
.
.
# Save the predictions
if ":" in model_name:
    model_name = model_name.replace(":", "_")
output_path = f"results/{model_name}_predictions_task2.jsonl"
```

3. Submit the job to the cluster
```
sbatch jobscript.sh
```

4. Update the result output in the evaluation.py and run the evaluation script
```
# evaluation.py
predictions_path = "results/llama3.3_70b_predictions_task1.jsonl"
```
- Run the evaluation script (should be ran in interactive session)
```
python evaluation.py 
```

### Note
- The pipeline should be ran in the virtual environment with the necessary dependencies installed.