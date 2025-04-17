import copy
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, MistralForCausalLM, AutoModelForSequenceClassification, pipeline, set_seed, BitsAndBytesConfig
import torch
from openai import OpenAI
import tiktoken
import google.generativeai as genai
import anthropic
import pdb

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
os.environ['LOCAL_MODEL_PATH'] = '/data1/shared_models/'

def free_gpu_memory(items_to_delete):
    for item in items_to_delete:
        del item
    torch.cuda.empty_cache()


def move_to_device(model):
    # Move model to the correct device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Moving model to device: {device}")
    if model.device.type != device:
        model.to(device)


def get_model_size_B(model_name: str, default: int = 2) -> int:
    """Get the model size from the model name, in Billions of parameters.
    """
    regex = re.search(r"((?P<times>\d+)[xX])?(?P<size>\d+)[bB]", model_name)
    if regex:
        return int(regex.group("size")) * int(regex.group("times") or 1)
    else:
        print(f"Could not infer model size from name '{model_name}'")
    return default


def load_nli_model_and_tokenizer(model_name_or_path, config):
    downloaded_nli_model_path = Path(
        config.DOWNLOADED_MODELS_PATH, model_name_or_path.replace("/", "--"))
    # check if nli_model_name path exists
    if downloaded_nli_model_path.exists():
        model_path = downloaded_nli_model_path
    else:
        model_path = model_name_or_path.replace("--", "/")

    if model_name_or_path in ["google--t5_xxl_true_nli_mixture"]:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    elif model_name_or_path in ["potsawee--deberta-v3-large-mnli"]:
        from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
    elif model_name_or_path in ["MoritzLaurer--DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        raise ValueError(f"nli model '{model_name_or_path}' not supported")

    model.eval()  # TODO: is this necessary?
    # Move model to cuda if available
    move_to_device(model)

    return model, tokenizer


class ChatBot:
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        args = [model_name_or_path, config, api_key]
        if model_name_or_path in [
            # "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.2-90B-Vision-Instruct"
        ]:
            self.chatbot = _FMAPISwissAI(*args)
        elif model_name_or_path in [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct-4bit", "meta-llama/Llama-3.1-70B-Instruct-8bit",
            "meta-llama/Llama-3.1-70B-Instruct",
            # "mistralai/Mistral-Large-Instruct-2407",
            "mistralai/Mistral-Nemo-Instruct-2407", "mistralai/Mistral-Small-Instruct-2409",
            "mistralai/Mixtral-8x7B-Instruct-v0.1-4bit",
            # "mistralai/Mixtral-8x7B-Instruct-v0.1-8bit", # not enough GPU memory
            "mistralai/Mixtral-8x22B-Instruct-v0.1-4bit", "mistralai/Mixtral-8x22B-Instruct-v0.1-8bit",
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it-4bit", "google/gemma-2-9b-it-8bit", "google/gemma-2-9b-it",
            "google/gemma-2-27b-it-4bit", "google/gemma-2-27b-it-8bit", "google/gemma-2-27b-it",
            "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            # "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", # cuda error :(
            "CohereForAI/aya-23-8B", "CohereForAI/aya-23-35B", "CohereForAI/aya-expanse-8b", "CohereForAI/aya-expanse-32b",
            "CohereForAI/c4ai-command-r-plus-4bit", "CohereForAI/c4ai-command-r-plus-8bit", "CohereForAI/c4ai-command-r-plus",
            "microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-small-128k-instruct", "microsoft/Phi-3-medium-128k-instruct",
        ]:
            self.chatbot = _HFModel(*args)
        elif model_name_or_path in ["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-08-06"]:
            self.chatbot = _OpenAI(*args)
        elif model_name_or_path in ["gemini-1.5-flash", "gemini-1.5-flash-002", "gemini-1.5-pro", "gemini-1.5-pro-002"]:
            self.chatbot = _Gemini(*args)
        elif model_name_or_path in ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]:
            self.chatbot = _Anthropic(*args)
        else:
            raise ValueError(f"chatbot '{model_name_or_path}' not supported")

    # def __call__(self, *args, **kwargs):
    #     return self.chatbot(*args, **kwargs)

    def update_seed(self, seed):
        if seed is not None:
            self.seed = seed

    def initialize_prompt_history(self, prompt):
        # check if prompt is a string or a list of messages
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            return [self.initialize_prompt_history(p) for p in prompt]
        elif isinstance(prompt, dict):
            if "role" not in prompt:
                prompt["role"] = "user"
            elif prompt["role"] not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role: {prompt['role']}")
            if set(prompt.keys()) != {"role", "content"}:
                raise ValueError(f"Prompt has invalid keys: {list(prompt.keys())}. Only 'role' and 'content' are allowed.")
            return prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def get_nr_of_tokens(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses.")

    def get_max_context_length(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses.")

    def get_temperature_range(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses.")

    def get_temperature(self, temperature):
        if temperature is None:
            return None
        min_temperature, max_temperature = self.get_temperature_range()
        if temperature == 'min':
            return min_temperature
        elif temperature == 'max':
            return max_temperature
        elif temperature > max_temperature:
            raise ValueError(f"Temperature {temperature} is higher than the maximum allowed temperature {max_temperature} (min temperature: {min_temperature}).")
        else:
            return temperature


class _HFModel(ChatBot):
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        if seed is not None:
            set_seed(seed)
        elif config is not None and config.SEED is not None:
            set_seed(config.SEED)
        if api_key is None:
            api_key = os.getenv('HF_ACCESS_TOKEN')
        local_model_directory = os.getenv('LOCAL_MODEL_PATH')
        print(">>>> LOCAL model directory:",local_model_directory)
        # self.pipe = pipeline(
        #     "text-generation",
        #     **pipeline_args
        #     )
        self.model, self.tokenizer = self.load_model_and_tokenizer(
            model_name_or_path, token=api_key, local_model_directory=local_model_directory)

    def load_model_and_tokenizer(self, model_name, token, local_model_directory, **kwargs):
        # print(f"\n-------\n  Loading model and tokenizer {model_name}\n-------\n")
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if local_model_directory is not None:
            model_path = local_model_directory + model_name.replace("/", "--")
        else:
            model_path = model_name
        print(">>>> Model path:", model_path)   
        model_kwargs.update(kwargs)
        if token is not None:
            model_kwargs.update({"token": token})
        tokenizer_kwargs = copy.deepcopy(model_kwargs)
        model, tokenizer = None, None

        if 'meta-llama/Llama-3.1-70B-Instruct-' in model_name:

            if '4bit' in model_name:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif '8bit' in model_name:
                print("Not enough GPU memory to load 8bit model?")
                pdb.set_trace()
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_quant_type="nf8",
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                )

            model_kwargs.update({
                'quantization_config': quantization_config,
                'torch_dtype': torch.bfloat16,
            })
            # model_name = model_name.replace('-4bit', '').replace('-8bit', '')
            model_path = model_path.replace('-4bit', '').replace('-8bit', '')

        elif model_name == 'mistralai/Mistral-Small-Instruct-2409':

            tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token

            model = MistralForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16)

        elif 'mistralai/Mixtral-8x7B-Instruct-v0.1' in model_name or 'mistralai/Mixtral-8x22B-Instruct-v0.1' in model_name:
            # if '4bit' in model_name:
            #     model_kwargs.update({'load_in_4bit': True})
            # elif '8bit' in model_name:
            #     model_kwargs.update({'load_in_8bit': True})
            # else:
            #     print("check self._is_bf16_compatible()")
            #     pdb.set_trace()
            #     model_kwargs.update({'torch_dtype': torch.float16})

            # specify how to quantize the model
            if '4bit' in model_name:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif '8bit' in model_name:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_quant_type="nf8",
                    bnb_8bit_compute_dtype=torch.float16,
                )

            model_kwargs.update({'quantization_config': quantization_config})
            # model_name = model_name.replace('-4bit', '').replace('-8bit', '')
            model_path = model_path.replace('-4bit', '').replace('-8bit', '')

        elif "google/gemma-3-" in model_name:
            # fixing gemma 3 error https://github.com/google-deepmind/gemma/issues/169
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            # an alternative would be to add padding ()
            # padding="max_length", max_length=4096

            model_kwargs.update({'torch_dtype': torch.bfloat16})

        elif "google/gemma-2-" in model_name:
            if '4bit' in model_name:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs.update(
                    {'quantization_config': quantization_config})
            elif '8bit' in model_name:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs.update(
                    {'quantization_config': quantization_config})
            else:
                model_kwargs.update({'torch_dtype': torch.bfloat16})

            # model_name = model_name.replace('-4bit', '').replace('-8bit', '')
            model_path = model_path.replace('-4bit', '').replace('-8bit', '')

        elif model_name == "CohereForAI/c4ai-command-r-plus-8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs.update({'quantization_config': quantization_config})
            # model_name = model_name.replace('-8bit', '')
            model_path = model_path.replace('-8bit', '')

        elif "Qwen" in model_name:
            model_kwargs.update({'torch_dtype': 'auto'})
        elif model_name in ["microsoft/Phi-3-mini-128k-instruct", "microsoft/Phi-3-small-128k-instruct", "microsoft/Phi-3-medium-128k-instruct"]:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        else:
            # print("No special model handling")
            model_kwargs.update(
                {'torch_dtype': torch.bfloat16 if self._is_bf16_compatible() else torch.float16})

        if tokenizer is None:
            # print(f"\n ## Loading tokenizer {model_name} with kwargs {tokenizer_kwargs}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, **tokenizer_kwargs)
            except Exception as e:
                print(
                    f"Error loading tokenizer {model_name} with kwargs {tokenizer_kwargs}: {e}")
                adjusted_path = model_path.replace(local_model_directory, '').replace("--", "/")
                print(f"Trying to load tokenizer from '{adjusted_path}' instead.")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path.replace(local_model_directory, '').replace("--", "/"), **tokenizer_kwargs)
        if model is None:
            try:
                # print(f"\n ## Loading model {model_name} with kwargs {model_kwargs}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, **model_kwargs)
            except Exception as e:
                print(
                    f"Error loading model {model_name} with kwargs {model_kwargs}: {e}")
                print(f"Trying to load model from '{model_path.replace(local_model_directory, '').replace('--', '/')}' instead.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path.replace(local_model_directory, '').replace("--", "/"), **model_kwargs)

        # Add pad token to the tokenizer if it doesn't already exist
        self._set_pad_token_id(tokenizer)

        # use model for inference
        model.eval()
        # Move model to cuda if available
        move_to_device(model)

        return model, tokenizer

    def _set_pad_token_id(self, tokenizer):
        """Add end-of-sentence pad token to the model and tokenizer if it doesn't already exist.
        """
        if tokenizer.pad_token_id is None:
            # If there's no pad_token_id, default to eos_token_id or any value of your choice
            self.pad_token_id = tokenizer.eos_token_id
        else:
            self.pad_token_id = tokenizer.pad_token_id
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def update_seed(self, seed):
        if seed is not None:
            set_seed(seed)

    def __call__(self, prompt, decoding_params, num_return_sequences=1, seed=None):
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        prompt = self.ensure_prompt_is_compatible_with_template(prompt)
        tokens = self.tokenizer.apply_chat_template(
            prompt, return_dict=True, return_tensors="pt", add_generation_prompt=True)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        prompt_length = tokens['input_ids'].shape[1]

        generation_config = {
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        generation_config.update(decoding_params)

        temperature = generation_config.get("temperature", None)
        if temperature is not None:
            temperature = self.get_temperature(temperature)
            if float(temperature) == 0.0:
                generation_config["do_sample"] = False
                generation_config["temperature"] = None
                generation_config["top_p"] = None
            else:
                generation_config["do_sample"] = True

        decoding_params = generation_config  # ensure that final parameters are saved

        # run the following to check the exact prompt being used:
        # print(self.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=False))

        # Generate new tokens
        generated_ids = self.model.generate(
            **tokens,
            **generation_config,
        )

        # print(f"\n\n\n\n   Generated text:\n{self.tokenizer.decode(generated_ids[0])}\n\n\n\n")

        if num_return_sequences == 1:
            return self.tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True)
        else:
            return [
                self.tokenizer.decode(
                    generated_id[prompt_length:], skip_special_tokens=True)
                for generated_id in generated_ids
            ]

    def ensure_prompt_is_compatible_with_template(self, prompt):
        """
        Ensure the input prompt is compatible with the chat template.
        Fixes issues with unsupported roles or improper role alternation.
        """
        def is_compatible(test_prompt):
            """
            Check if the given prompt is compatible with the chat template.
            Returns True if compatible, False otherwise.
            """
            try:
                _ = self.tokenizer.apply_chat_template(
                    test_prompt, return_dict=True, return_tensors="pt", add_generation_prompt=True
                )
                return True
            except Exception as e:
                # print(f"Compatibility check failed: {e}")
                return False

        # Step 1: Replace 'system' role with 'user' if needed
        test_prompt1 = [{'role': 'system', 'content': 'Hello!'}]
        if not is_compatible(test_prompt1):
            # print("Replacing system role with user role in the prompt.")
            # print(f"Example prompt before: {test_prompt1}")
            # print(f"Example prompt after: {self.replace_system_with_user(test_prompt1)}")
            prompt = self.replace_system_with_user(prompt)

        # Step 2: Concatenate consecutive roles if needed
        test_prompt2 = [{'role': 'user', 'content': 'Hello!'},
                        {'role': 'user', 'content': 'Hello again!'}]
        if not is_compatible(test_prompt2):
            # print("Concatenating consecutive user/assistant roles in the prompt.")
            # print(f"Example prompt before: {test_prompt2}")
            # print(f"Example prompt after: {self.concat_consecutive_roles(test_prompt2)}")
            prompt = self.concat_consecutive_roles(prompt)

        return prompt

    def replace_system_with_user(self, prompt):
        prompt = [{"role": "user" if p["role"] == "system" else p["role"],
                   "content": p["content"]} for p in prompt]
        return prompt

    def concat_consecutive_roles(self, prompt):
        """
        Concatenate consecutive user or assistant roles in the prompt.
        """
        concatenated_prompt = []
        current_role = None
        accumulated_content = ""

        for message in prompt:
            if message["role"] == current_role:
                # Accumulate content if the role is the same as the previous one
                accumulated_content += "\n" + message["content"]
            else:
                if current_role is not None:
                    # Append the previous accumulated message
                    concatenated_prompt.append(
                        {"role": current_role, "content": accumulated_content.strip()})
                # Start accumulating content for the new role
                current_role = message["role"]
                accumulated_content = message["content"]

        # Append the last accumulated message
        if current_role is not None:
            concatenated_prompt.append(
                {"role": current_role, "content": accumulated_content.strip()})

        return concatenated_prompt

    def get_nr_of_tokens(self, input_text):
        return len(self.tokenizer(input_text, return_tensors="pt").input_ids.squeeze())

    def get_max_context_length(self):
        return get_context_window(self.model)

    def get_temperature_range(self):
        # 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability.
        # default: 1.0
        return (0.0, 100.0)

    def _is_bf16_compatible(self) -> bool:
        """Checks if the current environment is bfloat16 compatible."""
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


class _OpenAI(ChatBot):
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        if seed is not None:
            self.seed = seed
        elif config is not None and config.SEED is not None:
            self.seed = config.SEED
        self.model_name = model_name_or_path
        self.api_key = api_key or self.get_api_key()
        self.client = self.initialize_client()
        self.tokenizer = None

    def get_api_key(self):
        return os.getenv('OPENAI_API_KEY')

    def initialize_client(self):
        return OpenAI(api_key=self.api_key)

    def __call__(self, prompt, do_sample=False, temperature=None, top_p=None, max_new_tokens=10, num_return_sequences=1, seed=None):
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        prompt = [
            {
                "role": p["role"],
                "content": [{"type": "text", "text": p["content"]}],
            }
            for p in prompt]
        generation_config = {
            "model": self.model_name,
            "messages": prompt,
            "seed": seed,
            "max_tokens": max_new_tokens,
            "n": num_return_sequences,
        }
        if temperature is not None:
            generation_config["temperature"] = self.get_temperature(
                temperature)
        if do_sample:
            if top_p is not None:
                generation_config["top_p"] = top_p

        response = self.client.chat.completions.create(
            **generation_config
        )
        # print(f'temperature: {generation_config["temperature"]}')
        # TODO: return the full response object?
        if num_return_sequences == 1:
            full_response_object = response.to_dict()
            response_text = response.choices[0].message.content
            return full_response_object, response_text
        else:
            # TODO
            pass

    def get_nr_of_tokens(self, input_text, model=None):
        """Return the number of tokens used by a list of messages."""
        if model is None:
            model = self.model_name
        # make sure the input_text is of type list and format the prompts
        input_text = self.initialize_prompt_history(input_text)
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_message = 4
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print(
                "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.get_nr_of_tokens(input_text, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print(
                "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.get_nr_of_tokens(input_text, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""get_nr_of_tokens() is not implemented for model {model}."""
            )
        num_tokens = 0
        for message in input_text:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def get_max_context_length(self):
        if self.model_name in [
            'gpt-4o',
            'gpt-4o-2024-05-13',
            'gpt-4o-2024-08-06',
            'chatgpt-4o-latest',
            'gpt-4o-mini',
            'gpt-4o-mini-2024-07-18',
            'gpt-4-turbo',
            'gpt-4-turbo-2024-04-09',
            'gpt-4-turbo-preview',
            'gpt-4-0125-preview',
            'gpt-4-1106-previe',
        ]:
            return 128000
        elif self.model_name in [
            'gpt-4',
            'gpt-4-0613',
            'gpt-4-0314',
        ]:
            return 8192
        elif self.model_name in [
            'gpt-3.5-turbo-0125',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-1106',
        ]:
            return 16385
        elif self.model_name in [
            'gpt-3.5-turbo-instruct',
        ]:
            return 4096
        else:
            raise NotImplementedError(
                f"""get_max_context_length() is not implemented for model {
                    self.model_name}."""
            )

    def get_temperature_range(self):
        # default: 1.0
        return (0.0, 2.0)


class _FMAPISwissAI(_OpenAI):
    def get_api_key(self):
        return os.getenv('FMAPI_SWISSAI_API_KEY')

    def initialize_client(self):
        return OpenAI(api_key=self.api_key, base_url="https://fmapi.swissai.cscs.ch")


class _Anthropic(ChatBot):
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        if seed is not None:
            self.seed = seed
        elif config is not None and config.SEED is not None:
            self.seed = config.SEED
        self.model_name = model_name_or_path
        if api_key is None:
            api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=api_key)

    def __call__(self, prompt, do_sample=False, temperature=None, top_p=None, max_new_tokens=10, num_return_sequences=1, seed=None):
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        prompt = [
            {
                "role": p["role"],
                "content": [{"type": "text", "text": p["content"]}],
            }
            for p in prompt]
        generation_config = {
            "model": self.model_name,
            "messages": prompt,
            "max_tokens": max_new_tokens,
        }
        if num_return_sequences > 1:
            raise ValueError("num_return_sequences>1 is not supported")
        if temperature is not None:
            generation_config["temperature"] = self.get_temperature(
                temperature)
        if do_sample:
            if top_p is not None:
                generation_config["top_p"] = top_p
        system_instruction = [p["content"]
                              for p in prompt if p["role"] == "system"]
        if len(system_instruction) > 0:
            # concatenate all system instructions
            system_prompt = " ".join(system_instruction)
            print(f"  System instruction: {system_prompt}")
            generation_config["system"] = system_prompt
        message = self.client.messages.create(
            **generation_config
        )
        # TODO: return the full response object?
        if num_return_sequences == 1:
            return message.content
        else:
            # TODO
            pass

    def get_nr_of_tokens(self, input_text, model=None):
        """Return the number of tokens used by a list of messages."""
        # TODO: check if this is correct
        pdb.set_trace()
        if model is None:
            model = self.model_name
        # make sure the input_text is of type list and format the prompts
        input_text = self.initialize_prompt_history(input_text)
        response = self.client.beta.messages.count_tokens(
            betas=["token-counting-2024-11-01"],
            model=self.model_name,  # "claude-3-5-sonnet-20241022",
            system="You are a scientist",
            messages=input_text,
        )
        token_count_response = response.json()
        print(f"Token count response: {token_count_response}")
        return token_count_response["input_tokens"]

    def get_max_context_length(self):
        # TODO: implement this
        raise NotImplementedError("Not yet implemented for Anthropic models.")

    def get_temperature_range(self):
        # default: 1.0
        return (0.0, 1.0)


class _Gemini(ChatBot):
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        if seed is not None:
            self.seed = seed
        elif config is not None and config.SEED is not None:
            self.seed = config.SEED
        self.model_name = model_name_or_path
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        self.model_info = genai.GenerativeModel(model_name_or_path)
        self.model, self.temperature, self.max_new_tokens, self.system_instruction = self.initialize_model_and_generation_config(
            self.model_name)

    def initialize_model_and_generation_config(self, model_name, temperature=None, top_p=None, top_k=None, max_new_tokens=10, system_instruction=[]):
        generation_config = {
            "top_p": top_p,
            "max_output_tokens": max_new_tokens,
            # "candidate_count": num_return_sequences # num_return_sequences>1 is not supported in the current version
        }
        if temperature is not None:
            temperature = self.get_temperature(temperature)
            generation_config.update({"temperature": temperature})
        if top_k is not None:
            generation_config["top_k"] = top_k
        if top_p is not None:
            generation_config["top_p"] = top_p
        if model_name in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro-002"]:
            # if model_name in ["gemini-1.5-flash-001", "gemini-1.5-pro-001", "gemini-1.0-pro-002", "gemini-1.5-flash-002"]:
            # seed is a preview feature that is only available for few models
            # by default, a random seed value is used
            generation_config['seed'] = self.seed

        # TODO: mode generation_config to inferencing
        if len(system_instruction) == 0:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
            )
        else:
            # concatenate all system instructions
            system_instruction = " ".join(system_instruction)
            # print(f"  System instruction: {system_instruction}")
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                system_instruction=system_instruction
            )
        return model, temperature, max_new_tokens, system_instruction

    def _role_mapping(self, role):
        if role == "assistant":
            return "model"
        else:
            return role

    def __call__(self, prompt, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=10, num_return_sequences=1, seed=None):
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        system_instruction = [p["content"]
                              for p in prompt if p["role"] == "system"]
        temperature = self.get_temperature(temperature)
        # if do_sample:
        #     # TODO: is this necessary or is the default top_p value better?
        #     top_p = 1.0
        #     top_k = 1000000
        # else:
        #     top_p = self.get_default_top_p()
        #     top_k = None
        if temperature != self.temperature or max_new_tokens != self.max_new_tokens or system_instruction != self.system_instruction:
            # re-initialize the model with the new generation config
            self.model, self.temperature, self.max_new_tokens, self.system_instruction = self.initialize_model_and_generation_config(
                self.model_name, temperature, top_p, top_k, max_new_tokens, system_instruction)

        user_message = prompt.pop()["content"]
        history = [
            {
                "role": self._role_mapping(p["role"]),
                "parts": p["content"],
            }
            for p in prompt if p["role"] != "system"]

        responses = []
        for _ in range(num_return_sequences):
            if len(history) == 0:
                response = self.model.generate_content(user_message)
            else:
                chat_session = self.model.start_chat(history=history)
                response = chat_session.send_message(user_message)
            responses.append(response.text)

        if num_return_sequences == 1:
            return responses[0]
        else:
            return responses

    def get_nr_of_tokens(self, input_text):
        return self.model.count_tokens(input_text).total_tokens

    def get_max_context_length(self):
        if self.model_name in ["gemini-1.5-flash", "gemini-1.5-flash-002"]:
            input_token_limit = 1048576
        elif self.model_name in ["gemini-1.5-pro", "gemini-1.5-pro-002"]:
            input_token_limit = 2097152
        else:
            raise NotImplementedError(
                f"get_max_context_length() is not implemented for model {self.model_name}.")
        return input_token_limit

    def get_temperature_range(self):
        # A temperature of 0 means that the highest probability tokens are always selected. In this case, responses for a given prompt are mostly deterministic, but a small amount of variation is still possible. Higher temperatures can lead to more diverse or creative results.
        if self.model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-002", "gemini-1.5-pro-002"]:
            # default: 1.0
            return (0.0, 2.0)
        elif self.model_name in ["gemini-1.0-pro-vision"]:
            # default: 0.4
            return (0.0, 1.0)
        elif self.model_name in ["gemini-1.0-pro-002"]:
            # default: 1.0
            return (0.0, 2.0)
        elif self.model_name in ["gemini-1.0-pro-001"]:
            # default: 0.9
            return (0.0, 1.0)
        else:
            raise NotImplementedError(
                f"""get_temperature_range() is not implemented for model {
                    self.model_name}."""
            )

    def get_default_top_p(self):
        # The default top_p value depends on the model
        # top_p value range: 0.0 - 1.0
        if self.model_name in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-002", "gemini-1.5-pro-002"]:
            return 0.95
        elif self.model_name in ["gemini-1.0-pro", "gemini-1.0-pro-vision"]:
            return 1.0
        else:
            raise NotImplementedError(
                f"""get_default_top_p() is not implemented for model {
                    self.model_name}."""
            )


def post_process_response(response, options, default_option="n/a"):
    """
    Post-processing function to map a response to a predefined set of options.

    Args:
    response (str): The text response to process.
    options (dict): A dictionary where keys are possible string options and values are corresponding scores.
    default_option (str): The default option to map to if the response does not match any option.

    Returns:
    float: The score corresponding to the processed response.
    """
    # Normalize the response
    response = response.replace("\n", "").lower().strip()

    # Sort options keys by length in descending order
    # check if options is a dictionary
    if isinstance(options, dict):
        sorted_keys = sorted(options.keys(), key=len, reverse=True)

        # Check if the response matches any of the sorted options keys
        for key in sorted_keys:
            if response.startswith(key.lower().strip()):
                return options[key]

        # If no match, return the default option score
        # print(f"warning: {response} not defined")
        # Default score if default_option not in options
        return options.get(default_option, 0.5)

    elif isinstance(options, list):
        options = [o.lower().strip() for o in options]
        if response in options:
            return response
        else:
            return default_option


def get_context_window(model):
    # Different model families use different names for the same field
    typical_fields = ["max_position_embeddings", "n_positions",
                      "seq_len", "seq_length", "n_ctx", "sliding_window"]

    # Check which attribute a given model object has
    context_windows = [getattr(model.config, field)
                       for field in typical_fields if field in dir(model.config)]

    # remove None values
    context_windows = [cw for cw in context_windows if cw is not None]

    # Grab the last one in the list; usually there's only 1 anyway
    if len(context_windows) > 0:
        return context_windows[-1]
    else:
        raise ValueError("Could not find context window size in model config.")


set_seed(42)
