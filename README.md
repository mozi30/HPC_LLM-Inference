# LLM Inference
by Moritz Anton Zideck

## Documentation
[LLM-Inference](LLM-Inference.pdf) contains the full documentation of the LLM Inference Experiment.

## Models
All models can be downloaded from HuggingFace with different quantisations.

### Gemma 2

https://huggingface.co/google/gemma-2-2b

### SmolLM2

https://huggingface.co/ngxson/SmolLM2-1.7B-Instruct-Q4_K_M-GGUF
https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF

### Meta LLama

https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF

## Scripts
All scripts can be found in the folder [scripts](scripts/).

Tests have been made on the Deucalion supercomputer.
For the reporduction of the results access to this system is required.

On a worker note run [run_bench.sh](scripts/run_bench.sh) with its preset settings to get the full evalutation equal to Track A in the documentation. It will handle module load and complete setup of environment.
If it cant find llama-cli change the path in the code to the correct one.
For result generation use the python file [single_llm_csv_generation.py](scripts/single_llm_csv_generation.py) to generate final full csv files and [single_llm_csv_visualisation.ipynb](scripts/single_llm_csv_visualisation.ipynb) to visualise them.


To get the model evaluation based on 2.1 LLM Analysis run [llama-bench-sweep.sh](scripts/llama-bench-sweep.sh)

`llama-bench-sweep -m {model_name.gguf} -t{thread_number}`

For visualisation use the notebook [multi_llm_inference_visualisation.ipynb](scripts/multi_llm_inference_visualisation.ipynb)

## Data

The `data/` folder contains multiple setup and result files used throughout the evaluation.

The [`prompts.json`](data/prompts.json) file contains all prompts used for the deep model evaluation, while [`results.json`](data/results.json) contains the results from multiple model evaluation runs.

The [`experiment_x86.tar.gz`](data/experiment_x86.tar.gz) file contains the full documentation folder for the **Gemma-2-2B** model inference experiments on the x86 setup.

The CSV file [`all_runs_with_response.csv`](data/all_runs_with_response.csv) provides a compressed overview of all individual runs. It includes both calculated and measured metrics, as well as the corresponding prompt and model response for each run.

The CSV file [`configuration_category_and_overall_mean_std.csv`](data/configuration_category_and_overall_mean_std.csv) contains summarized results across the different configuration runs. It reports the mean and standard deviation for each configuration, category, and overall result.
