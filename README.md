# Customer Churn Prediction & Predictive Analytics

**Author:** Pranay  
**Course / Program:** AICTE Data Analytics Internship  
**Project Name:** Customer Churn Prediction and Risk Analysis  

---

## 1. Project Overview
This project delivers a comprehensive predictive analytics solution designed to evaluate customer loyalty and forecast churn risk using Recency, Frequency, and Monetary (RFM) behavioral indicators. 

By aggregating order-level transactional data into customer-centric behavioral metrics, the machine learning pipeline identifies at-risk accounts, evaluates predictive model accuracy, and generates structured risk categorizations. This equips management with actionable insights to deploy targeted customer retention strategies before revenue loss occurs.

---

## 2. Problem Statement
Customer acquisition costs are significantly higher than retaining existing clients. Businesses face key challenges:
* Inability to identify early signs of customer disengagement.
* High revenue decay caused by unpredicted customer churn.
* Difficulty in differentiating high-value active customers from high-risk disengaged customers.

This project solves these issues by training a supervised classification algorithm (Random Forest) that calculates risk probabilities based on historical transaction behaviors.

---

## 3. Dataset Description
The analysis utilizes customer transaction records transformed into customer-level RFM features:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `CustomerID` | String | UniqueIdentifier assigned to each customer account. |
| `Recency` | Integer | Number of days elapsed since the customer's last purchase[cite: 3]. |
| `Frequency` | Integer | Total count of completed transactions made by the customer[cite: 3]. |
| `Monetary` | Float | Cumulative total monetary value spent across all orders ($)[cite: 3]. |
| `Churn` | Binary | Target label (1 = At Risk / Churned, 0 = Active / Retained)[cite: 3]. |

---

## 4. Technologies & Tools Used
* **Programming Language:** Python 3.x[cite: 3]
* **Development Environment:** Google Colab / Jupyter Notebook
* **Data Processing & Analytics:** `pandas`, `numpy`[cite: 3]
* **Machine Learning & Modeling:** `scikit-learn` (RandomForestClassifier, train_test_split, evaluation metrics)[cite: 3]
* **Documentation & Submission Formats:** `.ipynb`, `.docx`, `.txt`, `.md`

---

## 5. Project Architecture & Workflow