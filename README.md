# Mini-Project: Human Activity Recognition (HAR)

This repository contains the source code for the UE23CS352A Machine Learning mini-project Human Activity Recognition ( HAR ). The goal of this project is to classify human activities from smartphone sensor data using an LSTM model and compare the performance of the Adam and SGD optimizers.

---

## Team Members

* **Name:** Harsha Madev Hegde
* **SRN:** PES2UG23CS212

*(Note: Add your teammate's details here if applicable)*
* **Name:** Gudihalli Kiran
* **SRN:** PES2UG23CS207

---

## How to Set Up and Run the Project

### 1. Setup

**Clone the Repository**
```bash
git clone <your-repository-url>
cd <repository-name>
```

**Create a Virtual Environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
```

**Install Dependencies**
All required libraries are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**Dataset**
This project uses the "Human Activity Recognition with Smartphones" dataset from UCI. Ensure the `train.csv` and `test.csv` files are located in the `data/` directory.

### 2. Run the Experiment

To train the models and generate the comparison plot, run the main script from the root directory:
```bash
python src/train.py
```

The script will train the LSTM model twice (once with Adam, once with SGD), print the final test accuracies, and save the results in a plot named `optimizer_comparison.png`.