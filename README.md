# Frame_recSys Dataset

Find the data available in the data folder 

# Zero-shot experiments

Install all necessary libraries: 

    pip install -r requirements.txt

For running the experiments, either run:

    bash send_job.sh

Or if you want to run models and prompts individually, run: 

    python3 classification_with_model_prompting.py -m "Qwen/Qwen2.5-7B-Instruct" -p "zero1"