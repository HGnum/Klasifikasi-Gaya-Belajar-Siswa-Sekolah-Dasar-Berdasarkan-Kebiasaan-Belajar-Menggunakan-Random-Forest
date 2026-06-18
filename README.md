# Learning Style Classification

This repository contains a machine learning project for classifying elementary school students' learning styles based on learning habit questionnaire data.

The project focuses on the machine learning modeling process, including data preprocessing, model training, evaluation, and result analysis. The model has not yet been deployed into an application, API, or dashboard.

## Project Overview

The main objective of this project is to build a classification model that can identify students' learning styles based on questionnaire responses.

This project was developed as part of an academic research project in the field of machine learning.

## Research Objective

The objective of this project is to classify students' learning styles using machine learning based on learning habit data collected through questionnaires.

The workflow includes:

1. Data collection
2. Data preprocessing
3. Feature encoding
4. Model training
5. Model evaluation
6. Result analysis

## Dataset

The dataset used in this project is stored in:

```text
data/Data_siswa.csv
```

The dataset consists of student questionnaire data related to learning habits. Sensitive personal identity information has been removed from the published dataset to maintain data privacy.

## Methodology

The machine learning workflow used in this project includes:

1. Data understanding
2. Data cleaning
3. Data preprocessing
4. Label encoding
5. Train-test split
6. Model training
7. Model evaluation

The model was evaluated using classification metrics such as accuracy and F1-score.

## Model Performance

The final model achieved the following evaluation results:

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 85.91% |
| F1-Score | 0.8341 |

These results indicate that the model was able to classify students' learning styles with good performance based on the available questionnaire data.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook
* Google Colab
* Git & GitHub

## Repository Structure

```text
.
├── data/
│   └── Data_siswa.csv
├── notebooks/
│   └── learning_style_classification.ipynb
├── src/
│   └── learning_style_classification.py
├── results/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/HGnum/Klasifikasi-Gaya-Belajar-Siswa-Sekolah-Dasar-Berdasarkan-Kebiasaan-Belajar-Menggunakan-Random-Forest.git
```

2. Navigate to the project directory:

```bash
cd Klasifikasi-Gaya-Belajar-Siswa-Sekolah-Dasar-Berdasarkan-Kebiasaan-Belajar-Menggunakan-Random-Forest
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Python script:

```bash
python src/learning_style_classification.py
```

## Project Status

This repository currently focuses on the machine learning modeling stage.

Future development may include:

* Building a prediction application
* Creating a simple web-based interface
* Deploying the model as an API
* Adding Docker support for easier environment setup
* Improving model documentation and reproducibility

## Author

**Teguh Mulya Lesmana**
Informatics Engineering Fresh Graduate
Universitas Muhammadiyah Sukabumi
