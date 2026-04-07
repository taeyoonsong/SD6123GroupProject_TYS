# SD6123GroupProject_TYS
This project implements and evaluates three Privacy-Preserving Machine Learning paradigms:

Federated Learning (FL): Train a shared model across 20 simulated clients without sharing raw data (FedAvg, FedProx, SCAFFOLD)
Differential Privacy (DP): Add calibrated Gaussian noise during training to limit individual data influence
Machine Unlearning: Remove the influence of specific training samples from a trained model (GDPR right to erasure)

Two datasets are used:

Fashion-MNIST — image classification benchmark (10 classes, 70,000 images)
Adult Income — US Census tabular data from HuggingFace (binary income classification, sensitive demographic data)

ppml_project/
├── models/
│   ├── mnist_model.py          # CNN for Fashion-MNIST (~420K params)
│   └── adult_model.py          # MLP for Adult Income (14-dim input)
├── central/
│   ├── train_central_mnist.py  # Fashion-MNIST centralised baseline
│   ├── train_central_adult.py  # Adult Income centralised baseline
│   ├── train_dp_adult.py       # Differential privacy on Adult Income
│   └── train_overfit_mnist.py  # Overfit control model for MIA validation
├── fl/
│   ├── data_utils.py           # IID/non-IID partitioning for Fashion-MNIST
│   ├── adult_data_utils.py     # IID/non-IID partitioning for Adult Income
│   ├── mnist_client.py         # FedAvg/FedProx client (Fashion-MNIST)
│   ├── scaffold_client.py      # SCAFFOLD client (Fashion-MNIST)
│   ├── adult_client.py         # FedAvg/FedProx client (Adult Income)
│   ├── run_fl.py               # FL entry point for Fashion-MNIST
│   └── run_fl_adult.py         # FL entry point for Adult Income
├── privacy/
│   ├── mia.py                  # Loss-threshold membership inference attack
│   ├── run_mia.py              # Run MIA across all models
│   └── measure_overhead.py     # Communication and compute timing
├── unlearning/
│   └── run_unlearning.py       # Machine unlearning experiment
├── results/                    # Saved model checkpoints (auto-created)
└── data/                       # Downloaded datasets (auto-created)
