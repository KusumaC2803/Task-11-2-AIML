# 🎯 Student Suspicious Behavior Detection using Machine Learning

A Machine Learning project that detects suspicious student behavior during online examinations using a **Random Forest Classifier**. The model analyzes behavioral features such as face detection, gaze direction, head movement, hand movement, and phone presence to classify whether a student's behavior is **Normal** or **Suspicious**.

---

## 📌 Project Overview

Online examinations require effective proctoring to maintain academic integrity. This project uses machine learning to identify suspicious behavior by analyzing various behavioral features captured during an online examination.

The project performs the following tasks:

- Load the dataset
- Preprocess the data
- Train a Random Forest model
- Evaluate the model
- Save the trained model
- Predict new samples
- Visualize feature importance

---

## 📂 Dataset

**Dataset Name:** Student Suspicious Behaviour Detection Dataset

### Input Features

The dataset contains multiple behavioral features, including:

- Face Presence
- Number of Faces
- Face Coordinates
- Eye Coordinates
- Nose Position
- Mouth Position
- Face Confidence
- Hand Count
- Hand Coordinates
- Hand Object Interaction
- Head Pose
- Head Pitch
- Head Yaw
- Head Roll
- Phone Presence
- Phone Location
- Phone Confidence
- Gaze Direction
- Gaze Point
- Pupil Coordinates

### Target Variable

| Label | Meaning |
|-------|---------|
| 0 | Normal Behaviour |
| 1 | Suspicious Behaviour |

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

# 🤖 Machine Learning Algorithm

**Random Forest Classifier**

### Why Random Forest?

- Handles high-dimensional datasets effectively
- Reduces overfitting using multiple decision trees
- Provides high classification accuracy
- Supports feature importance analysis
- Performs well on classification tasks

---

# 📁 Project Structure

```
Task11-AIML/
│
├── data/
│   └── suspicious_behaviour.csv
│
├── models/
│   └── random_forest.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── predict.py
│   └── feature_importance.py
│
├── main.py
├── predict.py
├── feature_importance.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/KusumaC2803/Task11-AIML.git
```

## Navigate to the project folder

```bash
cd Task11-AIML
```

## Install the required libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

## Train and Evaluate the Model

```bash
python main.py
```

This will:

- Load the dataset
- Preprocess the data
- Train the Random Forest model
- Evaluate model performance
- Save the trained model
- Generate a confusion matrix

---

## Predict New Data

```bash
python predict.py
```

Example Output

```
Prediction: Suspicious Behaviour Detected
```

or

```
Prediction: Normal Behaviour
```

---

## Generate Feature Importance

```bash
python feature_importance.py
```

This generates:

- Top 10 important features
- Feature importance graph

Saved inside:

```
outputs/
```

---

# 📊 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 99% |
| Precision | 100% |
| Recall | 99% |
| F1 Score | 99% |

---

# 📈 Output Files

After execution, the following files are generated:

```
models/random_forest.pkl
outputs/confusion_matrix.png
outputs/feature_importance.png
```

---

# 📷 Project Workflow

```
Dataset
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Encoding
    │
    ▼
Train-Test Split
    │
    ▼
Random Forest Training
    │
    ▼
Model Evaluation
    │
    ▼
Save Model
    │
    ▼
Prediction
    │
    ▼
Feature Importance Visualization
```

---

# 🚀 Future Improvements

- Real-time webcam integration
- Live online exam monitoring
- Face recognition
- Deep Learning based behavior detection
- Flask/Django web deployment
- Automatic alert generation
- Live dashboard for invigilators

---

# 📚 Learning Outcomes

Through this project, I learned:

- Data preprocessing techniques
- Handling missing values
- Encoding categorical variables
- Training machine learning classification models
- Evaluating classification performance
- Saving and loading trained models
- Making predictions on new data
- Visualizing feature importance
- Organizing a complete machine learning project

---

# 📄 Requirements

```
pandas
numpy
scikit-learn
matplotlib
joblib
```

---

# 👩‍💻 Author

**Kusuma C**

Computer Science Engineering Student

- **GitHub:** https://github.com/KusumaC2803
- **LinkedIn:** https://linkedin.com/in/kusumac28

---

