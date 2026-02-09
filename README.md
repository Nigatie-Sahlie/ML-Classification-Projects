
# ML-Classification-Projects

A collection of Machine Learning classification models and experiments implemented in Python and Jupyter Notebooks.  
This repository explores different classification algorithms on imbalanced datasets (fraud detection) and compares their performance using confusion matrices and precision–recall curves.

---

## 🚀 Project Overview

This project demonstrates end-to-end ML classification workflows, including:

- Data loading & preprocessing  
- Model training  
- Model evaluation using:
  - Confusion Matrix  
  - Precision, Recall  
  - Precision–Recall Curve  
  - Average Precision (AP)
- Performance comparison between models

The main focus is understanding how different classifiers behave on **imbalanced classification problems**.

---

## 📂 Repository Structure

ML-Classification-Projects/
├── notebooks/        # Jupyter notebooks for experiments and visualizations
├── src/              # Helper scripts
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation


## 🧠 Models Implemented

- **K-Nearest Neighbors (KNN)**
  - Strong precision and recall trade-off
  - Performs well on non-linear patterns

- **Logistic Regression**
  - Baseline linear classifier
  - Interpretable and simple
  - Used for comparison with KNN
---

## 📈 Evaluation Metrics

Each model is evaluated using:

| Metric                 | Description |
|------------------------|-------------|
| Confusion Matrix       | TP, FP, TN, FN breakdown |
| Precision              | How many predicted frauds are truly fraud |
| Recall                 | How many fraud cases are caught |
| Precision–Recall Curve | Performance across thresholds |
| Average Precision (AP) | Area under PR curve (good for imbalanced data) |

These metrics are especially important for **fraud detection and anomaly detection** problems.

---

## 🛠 Installation & Setup

1️⃣ Clone the repository and download the dataset on Kaggle(Link on License section):
```bash
git clone https://github.com/Nigatie-Sahlie/ML-Classification-Projects.git
cd ML-Classification-Projects
````

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Run Jupyter Notebook:

```bash
jupyter notebook
```

4️⃣ Open notebooks inside the `notebooks/` folder and run the cells.

---

## 🧪 Example Results

* KNN achieved stronger performance in terms of **precision–recall trade-off**.
* Logistic Regression serves as a baseline model.
* Results are visualized using:

  * Confusion Matrices
  * Precision–Recall curves (Train vs Test)

These comparisons help understand **model behavior on imbalanced datasets**.

---

## 🔧 Future Improvements

* Add cross-validation and hyperparameter tuning
* Add class weighting / SMOTE for imbalance handling
* Add model deployment examples (Flask / FastAPI)
* Add experiment tracking

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch
3. Add your improvements
4. Submit a pull request

---

## 📌 License

This project is open-source and intended for learning and experimentation.
The data is from Kaggle Competition: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

---

## 📬 Author

**Nigatie Sahlie**
Machine Learning Classification Projects
GitHub: [https://github.com/Nigatie-Sahlie](https://github.com/Nigatie-Sahlie)

```

