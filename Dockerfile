# Use official Python image
FROM python:3.10-slim

# Install essential tools
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Set environment variables (example for AWS and MLflow)
ENV MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
ENV AWS_DEFAULT_REGION=us-east-1

# Copy project files
WORKDIR /app
COPY . /app

# Install requirements
RUN pip install --upgrade pip && pip install -r requirements.txt

# Initialize DVC (if remote data is used)
RUN dvc config core.no_scm true

# Default Gradio port
EXPOSE 7860
# Default MLflow UI port
EXPOSE 5000

# Flexible entrypoint (set via CMD)
CMD ["python", "main.py", "--mode", "train"] 