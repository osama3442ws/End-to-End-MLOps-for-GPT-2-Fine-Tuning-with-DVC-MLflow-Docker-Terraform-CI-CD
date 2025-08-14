import os
import json
import math
import mlflow
import mlflow.pytorch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
import config
from data_processing import get_processed_data

def compute_metrics(eval_pred):
    """
    A placeholder for computing metrics. Perplexity is calculated separately from the loss.
    """
    return {}

def train_model():
    """
    Trains the GPT-2 model, evaluates it, and logs metrics with MLflow.
    """
    # Setup MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    os.makedirs(config.MLRUNS_DIR, exist_ok=True)

    # Load data
    processed_data = get_processed_data()
    train_dataset = processed_data['train']
    eval_dataset = processed_data['validation']

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token # Important for padding

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.RESULTS_DIR,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model=None, # Perplexity is based on loss, not a specific metric
        no_cuda=config.NO_CUDA,
    )

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("data_files", config.DATA_FILE)
        mlflow.log_param("learning_rate", config.LEARNING_RATE)
        mlflow.log_param("per_device_train_batch_size", config.PER_DEVICE_TRAIN_BATCH_SIZE)
        mlflow.log_param("num_train_epochs", config.NUM_TRAIN_EPOCHS)
        mlflow.log_param("max_length", config.MAX_LENGTH)
        mlflow.log_param("model_name", config.MODEL_NAME)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
        )

        # Train
        trainer.train()

        # Save model and tokenizer
        model.save_pretrained(config.MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)
        mlflow.pytorch.log_model(model, "fine_tuned_gpt2")

        # Evaluate and log metrics
        eval_metrics = trainer.evaluate()
        train_loss = trainer.state.log_history[-1]['loss'] if 'loss' in trainer.state.log_history[-1] else None

        training_loss_value = float(train_loss) if train_loss is not None else 0.0
        validation_loss_value = float(eval_metrics.get('eval_loss', 0.0))
        perplexity = float(math.exp(validation_loss_value)) if validation_loss_value else 0.0

        mlflow.log_metric("Training Loss", training_loss_value)
        mlflow.log_metric("Validation Loss", validation_loss_value)
        mlflow.log_metric("Perplexity", perplexity)

        # Write DVC-friendly metrics file
        try:
            metrics_payload = {
                "training_loss": training_loss_value,
                "validation_loss": validation_loss_value,
                "perplexity": perplexity
            }
            with open(config.METRICS_FILE, 'w', encoding='utf-8') as f:
                json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to write metrics file: {e}")

    print("Model training and saving completed.")

if __name__ == '__main__':
    train_model() 