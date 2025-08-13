# ⚽ Strikers Performance Analysis

This project focuses on **comprehensive analysis, modeling, and clustering of football strikers’ performance data**.  
The goal is to **understand performance patterns, classify strikers based on key attributes, and predict future performance types** using statistical and machine learning techniques.  

It combines **data cleaning, exploratory data analysis (EDA), statistical hypothesis testing, clustering, and supervised learning** to deliver actionable insights.

---

## 📂 Project Files
- **Strikers_performance.xlsx** – Primary dataset containing individual striker statistics and categorical details such as nationality and footedness.
- **striker_analysis.py** – Python script that executes the complete workflow from data preprocessing to model building.
- **README.md** – Documentation explaining project structure, methodology, and outputs.
- **Outputs/** – Folder containing generated visualizations:
  - `output_7_0.png` – Pie chart of footedness distribution.
  - `output_7_1.png` – Footedness across nationalities.
  - `output_10_0.png` – Elbow method plot for KMeans clustering.
  - `output_11_1.png` – Confusion matrix for model evaluation.

---

## 📊 Features of the Project

### 1. **Data Preprocessing**
Before any analysis, the raw dataset undergoes **cleaning and preparation**:
- **Missing Value Imputation:**  
  - For numerical features → Imputed using **median** strategy.  
  - For categorical features → Imputed using **most frequent** strategy.  
- **Data Type Correction:**  
  - Specific columns like 'Goals Scored', 'Assists', 'Shots on Target', etc., are converted to integer types for accurate processing.

---

### 2. **Descriptive Analysis**
Statistical summaries and visualization are used to understand the dataset:
- **Descriptive Statistics:**  
  - Mean, median, standard deviation, min, max, etc., rounded to 2 decimal places.
- **Pie Chart:**  
  - Shows the percentage distribution of players’ **footedness**.
- **Countplot:**  
  - Displays distribution of footedness across different nationalities.

---

### 3. **Statistical Analysis**
To extract deeper insights, several statistical methods are applied:
- **Top Scoring Nationality:**  
  - Identifies which nationality produces strikers with the highest average goals scored.
- **Average Conversion Rate by Footedness:**  
  - Compares finishing efficiency between left-footed and right-footed players.
- **ANOVA Test:**  
  - Checks whether **consistency rates** significantly differ among strikers from different nationalities.
- **Correlation Test:**  
  - Evaluates if **Hold-up Play** is significantly correlated with **Consistency**.
- **Regression Analysis:**  
  - Assesses if Hold-up Play significantly influences Consistency.

---

### 4. **Feature Engineering**
Enhancements are made to create new useful features:
- **Total Contribution Score:**  
  - Sum of performance metrics like Goals, Assists, Shots, Dribbling, Aerial Duels, Defensive Contribution, Big Game Performance, and Consistency.
- **Label Encoding:**  
  - Encodes categorical variables such as **Footedness** and **Marital Status**.
- **Dummy Variables:**  
  - Converts **Nationality** into one-hot encoded features.

---

### 5. **Clustering Analysis**
Groups strikers into performance-based segments:
- **KMeans Clustering:**  
  - Uses the elbow method to determine the optimal number of clusters (**2** in this case).
- **Cluster Tagging:**  
  - Cluster 0 → **Best Strikers** 🏆  
  - Cluster 1 → **Regular Strikers** ⚽
- **Feature Mapping:**  
  - Creates a 'Strikers types' column using mapping for easy classification.

---

### 6. **Machine Learning**
Builds a classification model to predict striker type:
- **Data Preparation:**
  - Feature scaling with `StandardScaler`
  - Train-test split (80% training, 20% testing)
- **Model:**
  - Logistic Regression for classification.
- **Evaluation:**
  - Accuracy score (%)  
  - Confusion matrix (visualized with heatmap)

---
## ❓ Questions
1. What is the maximum goal scored by an individual striker?  
2. What is the portion of Right-footed strikers within the dataset?  
3. Which nationality strikers have the highest average number of goals scored?  
4. What is the average conversion rate for left-footed player?  
5. How many left-footed players are from France?  
6. What is the correlation co-efficient between hold up play and consistency score?  
7. What is the p-value for the Shapiro-Wilk test of consistency score? Is it normally distributed?  
8. What is the p-value for the Levene's test of ANOVA analysis? Is the heteroscedasticity assumed?  
9. Is there any significant correlation between strikers' Hold-up play and consistency rate?  
10. Describe the beta value of Hold-up Play you have found in your regression analysis.  
11. What is the average Total contribution score you get for the best strikers?  
12. What is the accuracy score of your LGR model? How many regular strikers your model predicted correctly? How many best strikers your model predicted incorrectly?  






















































































































