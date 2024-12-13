# CS592MLS-Project

Project report: [report.pdf](report.pdf)

The jupyter notebooks used for performing the evaluations is present in the eval directory of the repository. The notebooks are named as follows:
1. eval_gpt4o.ipynb: Performs eval on GPT4o for MMLU
2. eval_llama_vanilla.ipynb: Performs eval on vanilla LLAMA for MMLU
3. eval_llama_finetune.ipynb: Performs eval on finetuned LLAMA for MMLU
4. eval_llama_vanilla_math.ipynb: Performs eval on vanilla LLAMA for the k-digit addition task
5. eval_llama_finetune_math.ipynb: Performs eval on finetuned LLAMA for the k-digit addition task

The jupyter notebooks present in data_gen/ directory are used for generating the datasets for fine-tuning of LLaMA depending on task.
The outputs are present in test_runs folder. That also contains the logic for creating the plots as present in the report.