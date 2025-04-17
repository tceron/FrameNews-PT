# FrameRecSys Dataset

Find the data available in the data folder 

# Zero-shot experiments

Install all necessary libraries: 

    pip install -r requirements.txt

The script for these experiments are inside the ./prompting. For running the classification, either run:

    bash send_job.sh

Or if you want to run models and prompts individually, run: 

    python3 classification_with_model_prompting.py -m "Qwen/Qwen2.5-7B-Instruct" -p "zero1"