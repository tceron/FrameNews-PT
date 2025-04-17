# add the parent directory to the path
import copy
import sys
from utils import get_experiment_start_date, format_prompt, get_prompt_guidelines_manifestos
from llm_utils import ChatBot
from prompts import ALL_PROMPTS
import argparse
import yaml
from tqdm import tqdm
import pdb
import csv
import time
from pathlib import Path
from munch import Munch
import pandas as pd
import os
from collections import defaultdict
import re


def map_response(text):
    label = "na"
    all_found_labels = []
    # Define categories (using lowercase for case-insensitive matching)
    categories = ['Economic', 'Capacity and resources', 'Morality', 'Fairness and equality', 'Legality, Constitutionality, Jurisdiction', 'Crime and punishment', 
                  'Security and defense', 'Health and safety', 'Quality of life', 'Cultural identity', 'Public opinion', 'Political', 'Policy prescription and evaluation', 
                  'External regulation and reputation', 'Other'
                  ]
    # Define letter mappings
    letter_mappings = {1: 'Economic', 2: 'Capacity and resources', 3: 'Morality', 4: 'Fairness and equality', 5: 'Legality, Constitutionality, Jurisdiction', 6: 'Crime and punishment', 7: 'Security and defense', 8: 'Health and safety', 9: 'Quality of life', 10: 'Cultural identity', 11: 'Public opinion', 12: 'Political', 13: 'Policy prescription and evaluation', 14: 'External regulation and reputation', 15: 'Other'}

    # Check for category keywords in the text
    text_lower = text.lower().strip()
    for category in categories:
        if category in text_lower:
            all_found_labels.append(category)
    # If only one category is found, use that
    if len(all_found_labels) == 1:
        label = all_found_labels[0]
    # Otherwise check if text starts with category name or letter
    else:
        # Check if there's digit and extract digit in text_lower
        digit_match = re.search(r'\d+', text_lower)
        if digit_match:  # Ensure digit_match is not None
            digit = int(digit_match.group())
            # Check if the digit is in letter_mappings
            if digit in letter_mappings:
                label = letter_mappings[digit]
        else:
            # Check if the text starts with a letter mapping
            for letter, category in letter_mappings.items():
                if text_lower.startswith(str(letter)):  # Ensure letter is converted to string
                    label = category
                    break
    return label


def main(all_prompts, model, prompt_description, prompt_template, api_key, seed=42, output_folder='results'):

    guidelines = get_prompt_guidelines_manifestos()

    chatbot_args = [model]
    if api_key:
        print(f"Using API key: {api_key}")
        chatbot = chatbot_args.append(api_key)

    chatbot = ChatBot(*chatbot_args).chatbot

    generation_config = {
        'do_sample': False,
        'max_new_tokens': 10,
        'temperature': 0.0,
        'top_p': None,
    }

    os.makedirs(output_folder, exist_ok=True)

    filename = Path(output_folder, f"{model.replace('/', '--')}_{get_experiment_start_date()}.csv")

    with open(filename, 'w') as file:
        file.write(
            "prompt_id,ground_truth1,response_mapped,response,prompt_description\n")

    for index, row in tqdm(all_prompts.iterrows(), desc="Classifying prompts", total=len(all_prompts)):
        prompt_id = row['prompt_id']
        ground_truth1 = row['label']
        # ground_truth2 = row['label1']
        content = row['text']

        prompt = format_prompt(
            prompt_template, content, guidelines)

        try:
            response = chatbot(
                prompt,
                generation_config,
                seed=seed,
            )
        except Exception as e:
            print(f"Error while processing prompt {prompt_id}: {e}")
            response = f" Could not generate response for prompt {prompt_id} due to error: {e}"

        if args.model == "gpt-4o-2024-08-06":
            response = response[1]
        response_mapped = map_response(response)

        # Append the cleaned row along with the empty 'reformulations' column to the new CSV
        with open(filename, mode='a', newline='', encoding='utf-8') as new_file:
            writer = csv.writer(new_file)
            writer.writerow(
                [prompt_id,ground_truth1, response_mapped,response,prompt_description])

    print(
        f"Finished running all {len(all_prompts)} queries with model {model} :)")


def load_all_prompts_from_file(all_prompts_path):
    all_prompts_df = pd.read_csv(all_prompts_path)
    return all_prompts_df


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run statements on offline models using Hugging Face.")
    parser.add_argument('-m', '--model', required=True, type=str,
                        help="Model to run.")
    parser.add_argument('-a', '--api_key', required=False, type=str, default=0,
                        help="Api key (for gemini or openai) or hf token. If not specified, will be loaded from .env file.")
    parser.add_argument('-p', '--prompt_description', required=False, type=str, default=0,
                        help="Prompt description.")
    parser.add_argument('-o', '--output_folder', required=False, type=str, default='results',
                        help="Name of the output folder.")
    args = parser.parse_args()

    all_prompts_path = "../data/mfc_stratified_sample.csv" # '../data/frames_recsys_test.csv'   

    # account for possibility of key not to be present
    prompt_template = ALL_PROMPTS.get(args.prompt_description, None)
    if prompt_template is None:
        raise ValueError(
            f"Prompt description {args.prompt_description} not found in ALL_PROMPTS. Keys:\n{ALL_PROMPTS.keys()}")

    all_prompts = load_all_prompts_from_file(all_prompts_path)

    main(all_prompts, args.model, args.prompt_description,
         prompt_template, args.api_key, output_folder=args.output_folder)
    print("Done!")

    ##############################
    # RUN EXAMPLES
    ##############################

    # python classification_with_model_prompting.py -m "meta-llama/Llama-3.2-3B-Instruct" -p "zero1"
    # python classification_with_model_prompting.py -m "gpt-4o-2024-08-06" -p "zero1"
