# ✨ Fine-Tune GPT-2 with DVC, MLflow, Docker, Terraform & CI/CD

## 📜 Project Overview

Fine-tune GPT-2 on a creative-writing dataset, track experiments, and version data/models with production-friendly tooling:

* 📦 **DVC** for data/model versioning & reproducible pipelines
* 📊 **MLflow** for experiment tracking
* ☁️ **AWS S3** for remote storage of large artifacts
* 🐳 **Docker & Docker Compose** for consistent runtime
* 🏗️ **Terraform** to provision AWS infrastructure
* 🤖 **GitHub Actions** for CI/CD automation

**Result:**
📂 `fine_tuned_gpt2/` → fine-tuned model
📄 `metrics.json` → DVC metrics
📝 `artifacts/samples/` → sample generations

---

## 🏗️ Architecture (High Level)

* 📂 Code & pipeline definitions → **Git** (this repo)
* ⚙️ **DVC** orchestrates the pipeline & stores big files in **S3**
* 📊 **MLflow** logs metrics/params for every training run
* 🐳 **Docker** ensures identical environments locally & in CI
* 🏗️ **Terraform** provisions S3, ECR, EC2, IAM roles, and security groups
* 🔄 **GitHub Actions** runs `dvc repro` on push → pushes artifacts to S3

📌 Diagram: `Screenshot 2025-07-23 090851.png`

---

## 💻 Tech Stack

| Purpose           | Tools                         |
| ----------------- | ----------------------------- |
| **Model**         | 🤗 `transformers` (GPT-2)     |
| **Data**          | 🗂 `datasets`, `pandas`       |
| **Tracking**      | 📊 `mlflow`                   |
| **Orchestration** | ⚙️ `dvc`                      |
| **Runtime**       | 🐳 `docker`, `docker-compose` |
| **Storage**       | ☁️ AWS S3                     |
| **Infra**         | 🏗️ Terraform                 |
| **CI/CD**         | 🤖 GitHub Actions             |

---

## 📂 Repository Structure

```text
.
├── config.py                 # Loads config, overrides from params.yaml
├── params.yaml               # Single source of hyperparameters and paths
├── dvc.yaml                  # Pipeline definition
├── data_processing.py        # Data cleaning, tokenization, split
├── model.py                  # Training + MLflow logging
├── inference.py              # CLI text generation
├── main.py                   # Entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container setup
├── docker-compose.yml        # Services for training/MLflow/Gradio
├── creative_writing_dataset.csv
├── cleaned_creative_writing_dataset.csv
├── terraform/                # Infrastructure as code
└── .github/workflows/dvc-repro.yml  # CI workflow
```

---

## 🔄 How the Pipeline Works

**Stages in `dvc.yaml`:**

1. **🧹 preprocess**

   * Cleans dataset → `cleaned_creative_writing_dataset.csv` (DVC-tracked)

2. **🔤 tokenize**

   * Tokenizes with GPT-2 tokenizer → `artifacts/processed/` (DVC-tracked)

3. **🧠 train**

   * Fine-tunes GPT-2, logs to MLflow, creates `metrics.json`

4. **✍️ generate**

   * Generates sample text → `artifacts/samples/sample.txt`

💡 **Why DVC?** Guarantees reproducibility & versions large files.
💡 **Why MLflow?** Logs hyperparams & metrics for comparison.

---

## ⚙️ Configuration

**`params.yaml`** = single source of truth:

* 📂 Data: file paths & column names
* 🧠 Model: name, output directory
* 🎯 Training: learning rate, batch size, epochs, weight decay, early stopping, max length, test size
* ✍️ Inference: generation length, temperature, top\_k, prompt

**`config.py`** loads defaults & overrides from `params.yaml`.

---

## 🚀 Getting Started (Local)

**Prerequisites:**

* Python 3.10+
* Git
* AWS CLI (optional)
* S3 bucket (optional)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# DVC init
dvc init

# Optional: configure S3
$env:AWS_ACCESS_KEY_ID="..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_DEFAULT_REGION="eu-west-1"
dvc remote add -d s3remote s3://<your-bucket>/dvcstore
dvc remote modify s3remote region eu-west-1
```

Run pipeline:

```powershell
dvc repro
dvc push
```

---

## 🐳 Docker & Docker Compose

```powershell
docker build -t creative-gpt2 .
$env:ECR_REPOSITORY="creative-gpt2"; $env:IMAGE_TAG="local"
docker compose up -d
```

* **train** → runs `dvc pull → train → dvc push`
* **gradio** → serves model on `http://localhost:7860`
* **mlflow** → UI on `http://localhost:5000`

---

## 🏗️ Infrastructure with Terraform

### 📦 What Terraform Provisions

* ☁️ **S3 Bucket** (`<project_name>-dvc-bucket`) — Remote storage for DVC artifacts
* 📦 **ECR Repository** — Stores Docker images for deployment
* 🔐 **Security Group** — Opens 22 (SSH), 7860 (Gradio), 5000 (MLflow)
* 🖥 **EC2 Instance (Ubuntu)** — Runs Docker & Docker Compose
* 👤 **IAM Role** — Grants EC2 access to S3 and ECR without hardcoding keys
* 📤 **Outputs**:

  * `dvc_bucket_name` — DVC remote
  * `ecr_repository_url` — Docker image tag
  * `ec2_public_ip` — SSH access

### 🔗 How It Fits the Pipeline

* **DVC remote** → S3 bucket created by Terraform
* **Docker image** → Built locally/CI and pushed to ECR
* **EC2 runtime** → Pulls image and runs pipeline via Docker Compose
* **Ports** → Open for accessing Gradio & MLflow from your machine

### 🚀 Deploy with Terraform

```bash
cd terraform
terraform init
terraform plan -var "aws_region=eu-west-1" -var "project_name=llmops-gpt2" -var "key_name=<your-keypair>"
terraform apply -auto-approve -var "aws_region=eu-west-1" -var "project_name=llmops-gpt2" -var "key_name=<your-keypair>"
```

Copy the outputs:

* `dvc_bucket_name` → `s3://<dvc_bucket_name>/dvcstore`
* `ecr_repository_url` → for tagging/pushing image
* `ec2_public_ip` → `ssh ubuntu@<ec2_public_ip>`

### 🐳 Build & Push Image to ECR

```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t creative-gpt2 .
docker tag creative-gpt2:latest <ecr_repository_url>:latest
docker push <ecr_repository_url>:latest
```

### ▶️ Run on EC2

```bash
ssh ubuntu@<ec2_public_ip>
sudo apt-get update && sudo apt-get install -y git
git clone <your-repo.git> app && cd app
export ECR_REPOSITORY="<ecr_repository_url>"
export IMAGE_TAG="latest"
docker compose up -d
```

### 🔒 Security Notes

* Security group currently allows `0.0.0.0/0` on 22/7860/5000 — restrict for production
* `force_destroy = true` on S3 eases cleanup but deletes all objects — remove for safety

---

## 🤖 CI/CD with GitHub Actions

* Runs on every push to `main`
* Installs deps + `dvc[s3]`
* Configures AWS from secrets
* `dvc pull → dvc repro → dvc push`
* Uploads `metrics.json` + samples

**Required Secrets:**

* `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
* `DVC_REMOTE_URL` = `s3://<dvc_bucket_name>/dvcstore`

---

## 📈 Typical End-to-End Flow

1. Edit `params.yaml`
2. Commit & push
3. CI runs pipeline, uploads artifacts to S3
4. Gradio pulls newest model
5. Check MLflow/metrics.json for results

---

## 🛠 Tips & Troubleshooting

* Don’t commit large files — let DVC track them
* Verify AWS credentials if S3 auth fails
* Ensure internet access for Hugging Face downloads
* Quick experiments:

  ```powershell
  dvc exp run -S training.num_train_epochs=5
  ```

---

📌 Diagram available at: `Screenshot 2025-07-23 090851.png`

---

