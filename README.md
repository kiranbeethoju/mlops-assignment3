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
2. Train model: `python train.py`
3. Test model: `python predict.py`
4. Quantize model: `python quantize.py`

## Branches
- `main`: Initial setup
- `dev`: Model development
- `docker_ci`: Docker and CI/CD
- `quantization`: Model quantization

## Team
- Kirankumar Beethoju: G24Ai1115 