import os
import torch
import yaml

def _load_params():
    try:
        with open('params.yaml', 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if data else {}
    except FileNotFoundError:
        return {}

_PARAMS = _load_params()

# Data configuration (defaults)
DATA_FILE = 'cleaned_creative_writing_dataset.csv'
TEXT_COLUMN = 'cleaned_text'
TARGET_COLUMN = 'text'

# Model configuration (defaults)
MODEL_NAME = 'gpt2'
MODEL_OUTPUT_DIR = './fine_tuned_gpt2'
RESULTS_DIR = './results'

# Training configuration (defaults)
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 1
NUM_TRAIN_EPOCHS = 1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 3
MAX_LENGTH = 128
TEST_SIZE = 0.2
NO_CUDA = not torch.cuda.is_available()
METRICS_FILE = 'metrics.json'

# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "duration-prediction-experiment"
MLRUNS_DIR = "mlruns"

# Inference configuration
GENERATION_MAX_LENGTH = 1000
TEMPERATURE = 1.5
TOP_K = 100
GENERATION_PROMPT = "Write a story about a girl's adventures in a magical forest where she finds strange creatures"

# Override defaults from params.yaml when present
_data = _PARAMS.get('data', {})
DATA_FILE = _data.get('data_file', DATA_FILE)
TEXT_COLUMN = _data.get('text_column', TEXT_COLUMN)
TARGET_COLUMN = _data.get('target_column', TARGET_COLUMN)

_model = _PARAMS.get('model', {})
MODEL_NAME = _model.get('name', MODEL_NAME)
MODEL_OUTPUT_DIR = _model.get('output_dir', MODEL_OUTPUT_DIR)

_training = _PARAMS.get('training', {})
LEARNING_RATE = _training.get('learning_rate', LEARNING_RATE)
PER_DEVICE_TRAIN_BATCH_SIZE = _training.get('per_device_train_batch_size', PER_DEVICE_TRAIN_BATCH_SIZE)
NUM_TRAIN_EPOCHS = _training.get('num_train_epochs', NUM_TRAIN_EPOCHS)
WEIGHT_DECAY = _training.get('weight_decay', WEIGHT_DECAY)
EARLY_STOPPING_PATIENCE = _training.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
MAX_LENGTH = _training.get('max_length', MAX_LENGTH)
TEST_SIZE = _training.get('test_size', TEST_SIZE)
METRICS_FILE = _training.get('metrics_file', METRICS_FILE)

_inference = _PARAMS.get('inference', {})
GENERATION_MAX_LENGTH = _inference.get('generation_max_length', GENERATION_MAX_LENGTH)
TEMPERATURE = _inference.get('temperature', TEMPERATURE)
TOP_K = _inference.get('top_k', TOP_K)
GENERATION_PROMPT = _inference.get('prompt', GENERATION_PROMPT)