# Diabetes Prediction with Machine Learning

Welcome to the Diabetes Prediction project! This machine learning project aims to predict the likelihood of an individual having diabetes based on various health features.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Tuning](#tuning)

## Project Overview

![Diabetes](images/diabetes.jpg)

The project uses machine learning algorithms to predict diabetes based on a dataset of health-related features. We explore the data, preprocess it, build predictive models, and evaluate their performance.

The frameworks/libraries used in this project:

- Python 3.11
- Libraries: NumPy, Pandas, Matplotlib, scikit-learn

## Dataset

![Dataset](images/data.png)

The dataset used for this project contains health-related features, including age, BMI, and glucose levels. It's available in the data directory.

## Exploratory Data Analysis

![Number](images/number.png)

![gender](images/gender.png)

![HBA1C](images/hbc.png)

![heatmap](images/heatmap.png)


We perform exploratory data analysis (EDA) to gain insights into the dataset, visualize feature distributions, and identify correlations.

## Model Building

![accuracy](images/accuracy.png)

We build machine learning models, including Logistic Regression, Decision Tree, and Random Forest, to predict diabetes outcomes.

## Evaluation

![metric](images/metric.png)

We evaluate model performance using accuracy, precision, recall, and F1-score metrics. The ROC curve and AUC are also visualized.

## Tuning
We fine-tune the models using RandomizedSearchCV and GridSearchCV to optimize hyperparameters.
