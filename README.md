# FrameNews-PT Dataset

Find the data available in the data folder 

# Zero-shot experiments

Install all necessary libraries: 

    pip install -r requirements.txt

The script for these experiments are inside the ./prompting. For running the classification, either run:

    bash send_job.sh

Or if you want to run models and prompts individually, run: 

    python3 classification_with_model_prompting.py -m "Qwen/Qwen2.5-7B-Instruct" -p "zero1"


More details of the dataset are found in the [paper](https://arxiv.org/pdf/2506.16337). 

If you refer to our dataset, please cite: 

Agnese Daffara, Sourabh Dattawad, Sebastian Padó, Tanise Ceron. 2025. Generalizability of Media Frames: Corpus creation and analysis across countries. In Proceedings of the 14th Joint Conference on Lexical and Computational Semantics (startSEM). Co-located with EMNLP, Suzhou, China.
