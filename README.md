# AI Text Flagger backend

FastAPI backend and model training for AI Text Flagger, a classifier that tells human-written text from AI-generated text. You send it a piece of text, or a .txt, .docx or .pdf file, and it answers with a label (Human or AI), the probability that the text was generated and how confident the model is. The model is BERT Large Cased, fine-tuned on about 521,000 samples: 160,000 written by people and about 361,000 written by language models.

This repo holds the data pipeline, the training script and the API that serves the trained model. The React client lives in [ai_text_flagger_ui](https://github.com/MartinSG98/ai_text_flagger_ui). The AWS infrastructure is a Terraform module, [tf-module-ai_text_flagger](https://github.com/MartinSG98/tf-module-ai_text_flagger), which [tf-stack-ai_text_flagger](https://github.com/MartinSG98/tf-stack-ai_text_flagger) deploys.

## How it works

1. **Collect.** `src/data_loader.py` walks `Datasets/` and reads every .txt, .csv, .json and .docx file in it. Files under `Datasets/Human written text/` are labelled Human. Files under `Datasets/Ai generated text/<model>/` are labelled with their folder's name, one folder per model and twelve in all. CSV and JSON files are read from their `text`, `generated`, `content` or `article` field. Everything goes into `output/dataset.json`.
2. **Clean and split.** `src/preprocess.py` collapses whitespace, folds every model label into a single AI label, shuffles, and writes an 80/10/10 split to `output/train.json`, `output/val.json` and `output/test.json`.
3. **Train.** `src/model.py` tokenizes the training and validation splits to 512 tokens and fine-tunes `bert-large-cased` as a two-class classifier with the Hugging Face Trainer. The best checkpoint is saved to `output/model/final` with a `label_map.json` next to it.
4. **Predict.** For each request the API loads the tokenizer and the saved model, scores the first 512 tokens of the text, softens the logits with a temperature of 4 and calls the text AI when the AI probability is above 0.65.

The per-model folders are left over from a first version that tried to name the model behind a text. The data was too small to tell the models apart, so the classifier became binary.

## Running locally

Requires Python 3.11 or newer. The API runs on a CPU. Training needs a CUDA GPU.

```bash
python -m venv venv
venv\Scripts\activate         # Windows
source venv/bin/activate      # macOS / Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn transformers python-dotenv python-docx pypdf accelerate datasets safetensors huggingface-hub python-multipart
cp .env.example .env          # then set API_KEY
uvicorn src.api:app --reload
```

The API starts on http://127.0.0.1:8000. Check it is alive at http://127.0.0.1:8000/health and browse the interactive API reference at http://127.0.0.1:8000/docs. The trained model is not in git (`output/` is ignored and the weights are about 1.3 GB), so predictions need one in `output/model/final`. Train it as described under Training, or copy `config.json`, `model.safetensors` and `label_map.json` from an earlier run. The tokenizer is not saved with the model. It is downloaded from the Hugging Face Hub (`bert-large-cased`) on the first prediction and cached after that. No other external service is needed.

The install above is the same set of packages the EC2 bootstrap in the Terraform module installs, with CPU PyTorch because inference does not need a GPU. `requirements.txt` is a freeze of the training machine instead. It pins a nightly PyTorch build for CUDA 12.8 (`torch==2.10.0.dev20251204+cu128`), which is not on PyPI, so a plain `pip install -r requirements.txt` cannot resolve it. For training, install a CUDA build of PyTorch for your GPU in place of the CPU one.

## Configuration

Copy `.env.example` to `.env`. `src/api.py` loads it with python-dotenv when it starts.

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | none | The key clients must send in the `x-api-key` header. If it is not set, both predict routes answer 500 |

Everything else is set in code. The CORS origins in `src/api.py` are the UI's Vite dev server (`http://localhost:5173`) and the CloudFront domain of the hosted version. The AI threshold (`AI_THRESHOLD = 0.65`), the temperature (4.0) and the model path (`output/model/final`) are in `src/model.py`.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness check, returns `{"status": "ok"}`. Needs no key |
| POST | `/predict` | Classify the text in `{"text": "..."}` |
| POST | `/predict/file` | Classify an uploaded .txt (UTF-8), .docx or .pdf, sent as the multipart form field `file` |

Both predict routes need the `x-api-key` header and answer with the same shape:

```json
{ "prediction": "Human", "confidence": 0.94, "ai_probability": 0.06 }
```

`ai_probability` is the probability of the AI class, and `prediction` is AI only when it is above 0.65. `confidence` is the probability of whichever class is more likely. So for an AI probability between 0.5 and 0.65 the label says Human while the confidence is the AI side's.

A missing `x-api-key` header returns 422 and a wrong key 401. Empty text and a file with another extension return 400. A file that is empty or cannot be read comes back as 500, because the 400 the handler raises for it is caught by its general error handler. A `FileNotFoundError` while loading the model, such as a missing `label_map.json`, returns 503, and any other failure returns 500 with the reason in `detail`. The model is loaded from disk on every request, so each prediction includes the load time.

`TestingResources/human1.txt` is a short human-written review and `TestingResources/ai1.txt` an AI rewrite of it, for trying the file route.

## Training data

Human-written text, 160,000 samples:

| Source | What it is | Samples |
|---|---|---|
| [Human Written Text](https://www.kaggle.com/datasets/youssefelebiary/human-written-text) | CNN/DailyMail, Wikipedia and Project Gutenberg | 60,000 |
| [Pushshift Reddit](https://huggingface.co/datasets/fddemarco/pushshift-reddit) | Casual posts and discussions | 100,000 |

AI-generated text, 360,979 samples:

| Source | What it is | Samples |
|---|---|---|
| [AI Text Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile) | GPT-2, GPT-3, ChatGPT and GPT-J outputs, spread evenly over the twelve model folders | 240,000 |
| A second generated set | 10,000 more samples in each of the twelve model folders | 120,000 |
| Custom prompts | Documents and essays generated directly with GPT-4, Claude, Gemini, LLaMA and Mistral | about 1,000 |

That makes 520,979 samples. The 80/10/10 split gives 416,783 for training and 52,098 each for validation and test. All AI samples share one label, so the model tells human text from generated text and not one model from another. `test.json` is written, but no script in the repo scores the model on it yet.

## Training

```bash
mkdir output
python src/data_loader.py   # Datasets/ into output/dataset.json
python src/preprocess.py    # clean, fold labels to Human/AI, write the splits
python src/model.py         # fine-tune and save to output/model/final
```

Run them from the repo root, because the paths are relative. The raw data sits in `Datasets/`, which is not in git.

| Setting | Value |
|---|---|
| Base model | `bert-large-cased`, 24 layers, hidden size 1024 |
| Input | First 512 tokens, padded to 512 |
| Epochs | Up to 6, stopping after 2 evaluations without a better validation loss, best checkpoint kept |
| Batch size | 32 for training, 64 for evaluation |
| Learning rate | 2e-5, with weight decay 0.01 |
| Evaluation and checkpoints | Every epoch, logs in `output/model/logs` |
| Prediction | Softmax over the logits divided by 4.0, AI above 0.65 |

The temperature and the threshold were set after training. The temperature keeps the probabilities from sitting at 0 or 100 percent, and the threshold makes the model slower to call a text AI, which trades false positives for false negatives.

The model was trained on an AMD Ryzen 9 9900X with an RTX 5090 and 48 GB of DDR5. Tokenizing took about 90 minutes and training about 18 hours: 416,783 samples at a batch of 32 make 13,025 steps an epoch and 78,150 over six epochs. The trained model runs on any CUDA GPU or, more slowly, on a CPU.

## Deployment

The hosted version ran on AWS. CloudFront sits in front of an S3 bucket for the UI and an EC2 instance for this API, with a WAF web ACL on the distribution. CloudFront sends `/predict*` and `/health` to the instance on port 8000 without caching and forwards every header, so `x-api-key` gets through. Everything else comes from S3. The WAF uses the AWS Common Rule Set with `SizeRestrictions_BODY`, `CrossSiteScripting_BODY`, `GenericLFI_BODY` and `GenericRFI_BODY` set to Count, because long texts and file uploads tripped them. All of it is defined in [tf-module-ai_text_flagger](https://github.com/MartinSG98/tf-module-ai_text_flagger) and applied from [tf-stack-ai_text_flagger](https://github.com/MartinSG98/tf-stack-ai_text_flagger).

The instance runs Ubuntu 24.04 on an m7i-flex.large (2 vCPU, 8 GB) with a 20 GB gp3 volume in eu-west-2. Its user data clones this repo to `/home/ubuntu/ai_text_flagger`, installs CPU PyTorch and the API's packages into a venv, pulls the model from the model bucket into `output/model/final`, and installs a systemd service called `atf` that runs `uvicorn src.api:app --host 0.0.0.0 --port 8000`. The service reads `API_KEY` from `/home/ubuntu/ai_text_flagger/.env`, which Terraform does not create, so you write it on the instance once.

`.github/workflows/deploy.yml` updates the running instance on every push to main, except pushes that only touch `output/`, `Datasets/` or Markdown files. It connects over SSH as `ubuntu` with the `EC2_HOST` and `EC2_SSH_KEY` secrets, pulls main, runs `pip install -r requirements.txt --quiet` in the venv and restarts `atf`.

On the instance:

```bash
sudo systemctl restart atf   # after changing .env
sudo journalctl -u atf -f    # follow the logs
```

## Structure

```
src/
├── data_loader.py     walks Datasets/ and writes output/dataset.json
├── preprocess.py      whitespace cleanup, Human/AI labels, the 80/10/10 split
├── model.py           fine-tuning, the threshold and temperature, predict()
└── api.py             FastAPI app, API key check, text and file routes
TestingResources/      a human-written review and an AI rewrite of it
.github/workflows/     deploy.yml: pull and restart on the EC2 instance
.env.example           API_KEY
requirements.txt       freeze of the training machine
Datasets/              raw training data (not in git)
output/                dataset.json, the splits and the trained model (not in git)
```

The hosted demo is switched off for billing reasons.
