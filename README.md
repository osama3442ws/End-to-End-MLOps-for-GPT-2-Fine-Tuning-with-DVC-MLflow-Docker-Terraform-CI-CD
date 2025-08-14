# Fine-Tuned GPT-2 Story Generator

This project fine-tunes a GPT-2 model on a creative writing dataset to generate short stories based on a given prompt. The project is structured to be modular and easy to maintain, following common practices in software engineering.

## Project Structure

```
.
├── config.py               # All configurations and hyperparameters
├── data_processing.py      # Scripts for data loading and preprocessing
├── model.py                # Model training and evaluation logic
├── inference.py            # Script for generating text with the trained model
├── main.py                 # Main entry point to run training or inference
├── requirements.txt        # Project dependencies
├── fine-tuned_gpt2/        # Output directory for the saved model
├── results/                # Output directory for training results
├── mlruns/                 # MLflow tracking data
└── README.md               # This file
```

## Features

- **Fine-tuning GPT-2**: Leverages the `transformers` library to fine-tune the GPT-2 model.
- **Perplexity Evaluation**: Uses Perplexity as the primary metric for model evaluation, which is more suitable for language models than accuracy.
- **MLflow Integration**: Tracks experiments, parameters, and metrics using MLflow.
- **Modular Structure**: Code is organized into separate modules for configuration, data processing, training, and inference.
- **Command-Line Interface**: A simple CLI to switch between training and generation modes.

## DVC Integration

This project uses DVC with an S3 remote to version datasets and model artifacts and to describe the pipeline via `dvc.yaml` and hyperparameters in `params.yaml`.

Quick start:

```
pip install -r requirements.txt
dvc init
dvc remote add -d s3remote s3://<your-bucket>/dvcstore
dvc add creative_writing_dataset.csv
git add creative_writing_dataset.csv.dvc .gitignore
git commit -m "Track raw data with DVC"
dvc push

# Reproduce the pipeline
dvc repro
dvc push
```

## Setup

1.  **Clone the repository (if applicable)**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

The `main.py` script is the main entry point for the project.

### 1. Train the Model

To fine-tune the GPT-2 model on the provided dataset, run the following command:

```bash
python main.py --mode train
```

This will:
- Process the data from `cleaned_creative_writing_dataset.csv`.
- Train the model using the parameters in `config.py`.
- Save the fine-tuned model to the `./fine_tuned_gpt2` directory.
- Log all parameters and metrics to MLflow.

### 2. Generate a Story

Once the model is trained, you can generate a story using a prompt.

```bash
python main.py --mode generate
```

To use a custom prompt, use the `--prompt` argument:

```bash
python main.py --mode generate --prompt "In a world where the sky is made of glass,"
```

## Configuration

All project settings can be modified in the `config.py` file. This includes file paths, model parameters, and training hyperparameters.

## Running the Project with Docker and Docker Compose

### 1. Build images and run services

```bash
cd New\ folder
# Build images and run all services (training, Gradio, MLflow UI)
docker-compose up --build
```

### 2. Run each service separately

- **Train the model only:**
  ```bash
  docker-compose run train
  ```
- **Run Gradio interface only:**
  ```bash
  docker-compose up gradio
  ```
- **Run MLflow UI only:**
  ```bash
  docker-compose up mlflow
  ```

### 3. Accessing the services
- **Gradio interface:** [http://localhost:7860](http://localhost:7860)
- **MLflow UI:** [http://localhost:5000](http://localhost:5000)

### 4. Important Notes
- You can modify environment variables (such as AWS credentials) in the `docker-compose.yml` file as needed.
- Make sure your data and DVC files are in the same project directory or configure a suitable remote.
- You can add additional commands or customize the services as required.

## AWS S3 and GitHub Actions Integration with Docker

### 1. Set AWS Secrets in GitHub
- In your repository settings on GitHub, add the following secrets:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - (Optional) `AWS_DEFAULT_REGION` (e.g., us-east-1)

### 2. Pass secrets to Docker via GitHub Actions
- In your workflow file (example):

```yaml
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    env:
      AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
      AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      AWS_DEFAULT_REGION: us-east-1
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      - name: Build and run Docker Compose
        run: |
          docker-compose up --build --detach
```

### 3. Configure DVC remote (only once)
- You can run the following commands manually to set up the remote:

```bash
dvc remote add -d myremote s3://your-bucket/path
dvc remote modify myremote region us-east-1
```

After that, DVC will automatically use the environment variables to connect to S3 when running inside Docker or GitHub Actions.

### 4. Security
- Never commit credentials or AWS secrets to the code or repository.
- Always use GitHub Secrets and environment variables. 

## CI/CD Pipeline: GitHub Actions, AWS ECR, and EC2 Deployment

### Overview
This project uses a professional CI/CD pipeline to automate building, testing, and deploying your application. The pipeline is triggered on every push to the `main` branch and consists of the following steps:

1. **Build Docker Image:** GitHub Actions builds a new Docker image from your code.
2. **Push to AWS ECR:** The image is tagged and pushed to your AWS Elastic Container Registry (ECR).
3. **Deploy to EC2:** GitHub Actions connects to your AWS EC2 instance via SSH, pulls the new image from ECR, and restarts the Docker Compose services.

### Requirements
- **AWS ECR repository** for storing Docker images.
- **AWS EC2 instance** with Docker and docker-compose installed.
- **IAM user** with permissions to push to ECR and access EC2.
- **SSH key** for secure access to EC2.

### Required GitHub Secrets
Add these secrets to your repository settings:
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`: For AWS programmatic access.
- `EC2_HOST`: Public IP or DNS of your EC2 instance.
- `EC2_USER`: Username for SSH (e.g., `ubuntu` for Ubuntu AMIs).
- `EC2_SSH_KEY`: Private SSH key for accessing EC2 (use the contents, not the file path).

### How It Works
- On every push to `main`, GitHub Actions will:
  1. Build the Docker image and tag it with the commit SHA.
  2. Push the image to your ECR repository.
  3. SSH into your EC2 instance and run `docker-compose` to pull and restart the services with the new image.

### docker-compose.yml
- The `docker-compose.yml` file is configured to use the image from ECR using the variables `${ECR_REPOSITORY}` and `${IMAGE_TAG}` for flexibility.

### Customization
- You can extend the workflow to include tests, notifications, or additional deployment steps as needed.

--- 