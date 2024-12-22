# Heterogeneous Federated Learning with Quantization

## Project Overview

This project, **Heterogeneous Federated Learning with Quantization**, aims to address the challenges of Federated Learning (FL) in heterogeneous environments, including data heterogeneity, device variability, and limited computational resources. The project leverages **quantization techniques** to reduce model size and improve computational efficiency while employing the **FedCS client selection protocol** to optimize training in resource-constrained scenarios. 

The MNIST dataset is used as the benchmark for evaluating the performance of the proposed Federated Learning framework.

---

## Problem Statement

Federated Learning (FL) enables decentralized collaborative training of machine learning models without sharing raw data, ensuring data privacy and security. However, real-world FL implementations face several key challenges:

- **Data Heterogeneity**: Clients' datasets are often non-IID (non-independent and identically distributed), causing slower convergence and degraded global model performance.
- **Device Heterogeneity**: Participating devices vary in computational power, memory, and communication capabilities, leading to uneven resource usage and inefficient training.
- **Resource Constraints**: Limited bandwidth and storage demand optimized solutions to reduce communication overhead and model size.

This project addresses these challenges by incorporating **quantization techniques** and **client selection protocols** to improve the scalability, efficiency, and adaptability of FL systems.

---

## Features

- **Simulated FL Environment**: The project includes a simplified FL setup in `parallel_fedcs.ipynb`, which trains a model on five clients using the MNIST dataset.
- **Simulated Clients**: In `src/client.py` we create clients and simulate their heterogeneity.
- **Quantization Techniques**: Implements quantization methods such as:
  - **int8 Quantization** for efficient inference and reduced model size.
  - **Kurtosis Regularization (KURE)** to make weights more robust to quantization.
  - **MQAT + APQN** to handle varying bit-widths and simulate quantization noise.
- **Client Selection Protocol**: Adopts the **FedCS protocol** to prioritize clients based on resource availability, reducing training time and increasing efficiency.
- **Dataset Handling**: Introduces a **shared dataset strategy** to mitigate the effects of non-IID data distributions.

---

## Installation and Setup

Follow these steps to set up the environment and run the project:

### Install Dependencies
Set up the environment using the provided Conda environment file:
```bash
conda env create -f env.yml
```

### Dataset Usage in This Project
1. For **non-IID setups**, each client is assigned data corresponding to only 2 specific classes, simulating a real-world scenario where local data distributions are imbalanced.
2. For the **shared dataset approach**, 5% of the global MNIST dataset is used as a shared subset and distributed equally among clients. This introduces a degree of uniformity while maintaining client privacy.

The MNIST dataset can be downloaded from the official page: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
