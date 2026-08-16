# Week-10
# AutoML Framework Comparison: PyCaret vs H2O AutoML

## Introduction

This project explores and compares two Automated Machine Learning (AutoML) frameworks, PyCaret and H2O AutoML, using the California Housing Dataset. The objective was to investigate how each framework simplifies the machine learning workflow, evaluate their compatibility with Google Colab, and assess their ability to automatically train predictive models for a regression problem.

## Dataset

The California Housing Dataset was obtained from Scikit-learn and contains information about housing districts in California. The target variable for this project is `MedHouseVal`, which represents the median house value.

### Features

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

### Loading the Dataset

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing(as_frame=True)
df = housing.frame

df.head()
```

---

## Framework 1: PyCaret

### Overview

PyCaret is a low-code machine learning library designed to automate many steps of the machine learning process, including preprocessing, model selection, and evaluation.

### Installation

```python
!pip install pycaret
```

### Implementation

```python
from pycaret.regression import *

setup(
    data=df,
    target="MedHouseVal",
    session_id=42,
    verbose=False
)

best_model = compare_models()
```

### Challenges Encountered

Although PyCaret was installed successfully, the framework could not be executed in the Google Colab environment because the runtime was using Python 3.12. PyCaret 3.3.2 only supports Python 3.9, 3.10, and 3.11. As a result, model training could not be completed.

#### Error Message

```text
RuntimeError:
Pycaret only supports python 3.9, 3.10, 3.11.
Your actual Python version: 3.12
```

### Strengths

- Easy to use
- Requires minimal code
- Automatic model comparison
- Suitable for rapid prototyping

### Limitations

- Limited support for newer Python versions
- Dependency conflicts during installation
- Unable to complete training in the current environment

---

## Framework 2: H2O AutoML

### Overview

H2O AutoML is an open-source machine learning platform that automatically trains and ranks multiple machine learning models, including ensemble models.

### Installation

```python
!pip install h2o
```

### Initialization

```python
import h2o

h2o.init()
```

### Data Preparation

```python
hf = h2o.H2OFrame(df)

train, test = hf.split_frame(
    ratios=[0.8],
    seed=42
)
```

### AutoML Training

```python
from h2o.automl import H2OAutoML

aml = H2OAutoML(
    max_models=10,
    seed=42
)

aml.train(
    y="MedHouseVal",
    training_frame=train
)
```

### Leaderboard Generation

```python
leaderboard = aml.leaderboard
print(leaderboard)
```

H2O AutoML was successfully installed and executed. It trained multiple models and generated a leaderboard ranking model performance automatically.

### Strengths

- Compatible with Python 3.12
- Automatically builds multiple models
- Generates performance leaderboards
- Scales well for larger datasets
- Supports ensemble learning

### Limitations

- Larger installation size
- Longer setup process
- Higher computational requirements

---

## Comparison Summary

| Criteria | PyCaret | H2O AutoML |
|-----------|----------|------------|
| Installation | Successful | Successful |
| Python 3.12 Compatibility | No | Yes |
| Model Training Completed | No | Yes |
| Automatic Model Comparison | Available | Available |
| Leaderboard Generation | Not Completed | Completed |
| Ease of Use | High | Moderate |
| Scalability | Moderate | High |

---

## Results

The experiment demonstrated significant differences in compatibility and execution.

PyCaret provided a simple and user-friendly workflow; however, the Python version requirements prevented the framework from being used successfully in the Google Colab environment.

H2O AutoML successfully completed the full machine learning workflow, including dataset conversion, model training, and leaderboard generation. This made it the only framework that could be fully evaluated during the experiment.

### Experimental Leaderboard

| Rank | Framework | Status |
|------|-----------|---------|
| 1 | H2O AutoML | Successfully installed, trained models, and generated leaderboard |
| 2 | PyCaret | Installed successfully but could not train models due to Python version incompatibility |

---

## Challenges Encountered

Several challenges arose during the implementation process:

- Python 3.12 was incompatible with PyCaret 3.3.2.
- Installing AutoML frameworks introduced dependency conflicts involving Pandas, NumPy, and Scikit-learn.
- Runtime restarts were required after package installations.
- Different frameworks required different library versions, creating compatibility issues within the same Colab environment.

---

## Technologies Used

- Python
- Google Colab
- Pandas
- Scikit-learn
- PyCaret
- H2O AutoML

---

## Conclusion

This project compared PyCaret and H2O AutoML using the California Housing Dataset for a regression task.

While PyCaret offers a streamlined and beginner-friendly approach to machine learning, it could not be evaluated fully due to Python version incompatibility in Google Colab. H2O AutoML successfully completed the entire workflow and demonstrated strong automation capabilities through automatic model training and leaderboard generation. Overall, H2O AutoML proved to be the more practical framework for the environment used in this project.

---

## Author

**Chandani Gupta**

Week 10 - AutoML Framework Comparison
