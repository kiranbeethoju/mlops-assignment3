# ML Ops Assignment 3 - End-to-End MLOps Pipeline

This project implements a complete MLOps pipeline for California Housing price prediction.

## Project Structure
```
├── train.py          # Train scikit-learn model
├── predict.py        # Load and test model
├── quantize.py       # Manual quantization
├── Dockerfile        # Container configuration
├── requirements.txt  # Dependencies
└── .github/workflows/ci.yml  # CI/CD pipeline
```

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python3 train.py`
3. Test model: `python3 predict.py`
4. Quantize model: `python3 quantize.py`

## Branches
- `main`: Initial setup
- `dev`: Model development
- `docker_ci`: Docker and CI/CD
- `quantization`: Model quantization

## Results
- Original Sklearn Model R2 Score: 0.5758
- Quantized Model R2 Score: -0.1799
- Model Size - unquant_params.joblib: 0.40 KB
- Model Size - quant_params.joblib: 0.59 KB

## Team
- Kirankumar Beethoju: G24Ai1115 