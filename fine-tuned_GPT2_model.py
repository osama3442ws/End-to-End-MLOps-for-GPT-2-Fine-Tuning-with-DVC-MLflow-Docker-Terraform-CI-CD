!pip install transformers
!pip install datasets
!pip install mlflow
!pip install torch
!pip install pyngrok -q
!pip install gradio

import subprocess
from pyngrok import ngrok, conf
import getpass 

import os
import mlflow
import mlflow.pytorch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from transformers import EarlyStoppingCallback
import torch

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI])

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("duration-prediction-experiment")

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = getpass.getpass()
port=5000
public_url = ngrok.connect(port).public_url
print(f' * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"')

# Import necessary libraries
import os
import math
import mlflow
import mlflow.pytorch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict
import torch

def main():
    # ---------------------- MLflow Setup ----------------------
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Define MLflow backend database
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Set tracking URI
    mlflow.set_experiment("duration-prediction-experiment")  # Define experiment name
    os.makedirs("mlruns", exist_ok=True)  # Ensure MLflow run directory exists

    # ---------------------- Load and Prepare Dataset ----------------------
    data_files = 'cleaned_creative_writing_dataset.csv'  # Path to dataset file
    dataset = load_dataset('csv', data_files=data_files)  # Load CSV dataset

    # Remove unnecessary columns if exist and rename target column
    if 'text' in dataset['train'].column_names:
        dataset = dataset['train'].remove_columns(['text'])
    else:
        dataset = dataset['train']
    dataset = dataset.rename_column('cleaned_text', 'text')

    # ---------------------- Load Pretrained GPT-2 and Tokenizer ----------------------
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Load GPT-2 tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')    # Load GPT-2 model
    tokenizer.pad_token = tokenizer.eos_token          # Set pad token

    # ---------------------- Tokenization Function ----------------------
    def tokenize_function(examples):
        input_ids = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128,
        )
        input_ids['labels'] = input_ids['input_ids'].copy()  # Use same input as labels
        return input_ids

    # Apply tokenization and split dataset into train/validation
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
    tokenized_datasets = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })

    # ---------------------- Training Configuration ----------------------
    learning_rate = 2e-5
    per_device_train_batch_size = 1
    num_train_epochs = 1
    max_length = 128

    training_args = TrainingArguments(
        output_dir='./results',                   # Directory for saving results
        evaluation_strategy='epoch',              # Evaluate at the end of each epoch
        save_strategy='epoch',                    # Save checkpoint after each epoch
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,                        # Regularization
        load_best_model_at_end=True,
        metric_for_best_model=None,               # No specific metric to determine best model
        no_cuda=not torch.cuda.is_available(),    # Use GPU if available
    )

    # ---------------------- Start Training with MLflow Logging ----------------------
    with mlflow.start_run():
        # Log training parameters
        mlflow.log_param("data_files", data_files)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("per_device_train_batch_size", per_device_train_batch_size)
        mlflow.log_param("num_train_epochs", num_train_epochs)
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("model_name", "gpt2")

        # Initialize Trainer with early stopping
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            compute_metrics=compute_metrics,  # (You must define this function elsewhere)
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()  # Begin model fine-tuning

        # ---------------------- Save Model and Tokenizer ----------------------
        model.save_pretrained('./fine_tuned_gpt2')
        tokenizer.save_pretrained('./fine_tuned_gpt2')
        mlflow.pytorch.log_model(model, "fine_tuned_gpt2")  # Log model to MLflow

        # ---------------------- Evaluate Model and Log Metrics ----------------------
        eval_metrics = trainer.evaluate()
        train_loss = trainer.state.log_history[-1]['loss'] if 'loss' in trainer.state.log_history[-1] else None
        mlflow.log_metric("Training Loss", train_loss if train_loss is not None else 0.0)
        mlflow.log_metric("Validation Loss", eval_metrics['eval_loss'])

        # Compute and log perplexity
        perplexity = math.exp(eval_metrics["eval_loss"])
        print("Perplexity:", perplexity)
        mlflow.log_metric("Perplexity", perplexity)

    print("Model training and saving completed.")

    # ---------------------- Load and Use Fine-Tuned Model for Text Generation ----------------------
    model_name = "./fine_tuned_gpt2"
    if os.path.exists(model_name):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        raise FileNotFoundError(f"{model_name} not found. Please train and save the model first.")

    model.eval()  # Set model to evaluation mode

    # Define function to generate story from prompt
    def generate_story(prompt, max_length=1000, temperature=1.5, top_k=100):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_story = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_story

    # ---------------------- Generate and Display a Sample Story ----------------------
    prompt = "Write a story about a girl's adventures in a magical forest where she finds strange creatures"
    generated_text = generate_story(prompt, max_length=1000)
    print(generated_text)

if __name__ == "__main__":
    main()













# ---------------------- MLflow Setup ----------------------
import os
import math
import mlflow
import mlflow.pytorch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, DatasetDict
import torch

def main():
    # MLflow setup
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("duration-prediction-experiment")
    os.makedirs("mlruns", exist_ok=True)

    # Load data
    data_files = 'cleaned_creative_writing_dataset.csv'
    dataset = load_dataset('csv', data_files=data_files)
    if 'text' in dataset['train'].column_names:
        dataset = dataset['train'].remove_columns(['text'])
    else:
        dataset = dataset['train']
    dataset = dataset.rename_column('cleaned_text', 'text')

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        input_ids = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128,
        )
        input_ids['labels'] = input_ids['input_ids'].copy()
        return input_ids

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
    tokenized_datasets = DatasetDict({
        'train': train_test_split['train'],
        'validation': train_test_split['test']
    })

    def compute_metrics(eval_pred):
        # No need to compute accuracy here, just return an empty dict
        return {}

    learning_rate = 2e-5
    per_device_train_batch_size = 1
    num_train_epochs = 1
    max_length = 128

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=None,
        no_cuda=not torch.cuda.is_available(),
    )

    with mlflow.start_run():
        mlflow.log_param("data_files", data_files)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("per_device_train_batch_size", per_device_train_batch_size)
        mlflow.log_param("num_train_epochs", num_train_epochs)
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("model_name", "gpt2")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()

        model.save_pretrained('./fine_tuned_gpt2')
        tokenizer.save_pretrained('./fine_tuned_gpt2')
        mlflow.pytorch.log_model(model, "fine_tuned_gpt2")

        eval_metrics = trainer.evaluate()
        train_loss = trainer.state.log_history[-1]['loss'] if 'loss' in trainer.state.log_history[-1] else None
        mlflow.log_metric("Training Loss", train_loss if train_loss is not None else 0.0)
        mlflow.log_metric("Validation Loss", eval_metrics['eval_loss'])
        # Calculate and log Perplexity
        perplexity = math.exp(eval_metrics["eval_loss"])
        print("Perplexity:", perplexity)
        mlflow.log_metric("Perplexity", perplexity)

    print("Model training and saving completed.")

    # Generate a sample text
    model_name = "./fine_tuned_gpt2"
    if os.path.exists(model_name):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        raise FileNotFoundError(f"{model_name} not found. Please train and save the model first.")
    model.eval()

    def generate_story(prompt, max_length=1000, temperature=1.5, top_k=100):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_story = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_story

    prompt = "Write a story about a girl's adventures in a magical forest where she finds strange creatures"
    generated_text = generate_story(prompt, max_length=1000)
    print(generated_text)

if __name__ == "__main__":
    main() 


import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned GPT-2 model and tokenizer
model_name = "./fine_tuned_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()

# Define the generation function
def generate_story(prompt, max_length=1000, temperature=1.5, top_k=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_story = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_story

# Define Gradio interface
def gradio_generate(prompt):
    generated_text = generate_story(prompt)
    return generated_text

# Create Gradio interface
gradio_interface = gr.Interface(
    fn=gradio_generate,
    inputs="text",
    outputs="text",
    title="Story Generator",
    description="Enter a prompt to generate a story using the fine-tuned GPT-2 model.",
)

# Launch the interface
gradio_interface.launch(share=True)    