# Mini Project 2 - Machine Learning for Analysis and Prediction

## Questions and Answers

### 1) Which are the most decisive factors forming the price of a car?

Based on the regression model and feature analysis, the most decisive factors are:

- **Mileage** – the more a car has been driven, the lower the price
- **Age** – older cars are generally cheaper, except classic/luxury cars
- **Engine size (ccm2)** – larger engines typically mean higher price
- **Make and model** – brand has a significant impact on price

The regression model achieved an R² of ~0.60, meaning these features explain about 60% of the price variation.
it could maybe be more optimized if we put more categories in the model.
---

### 2) Which make and model is most popular according to this data source?

**Most popular make: Ford** with 1,582 listings, followed by Toyota (990) and Volvo (258).

**Most popular model: Ford Fiesta 1.0 EcoBoost Titanium 5d** with 63 listings.

Technical characteristics of the Ford Fiesta:
- Small 1.0L turbocharged petrol engine
- Known for fuel efficiency and low running costs

---

### 3) Is there any obvious tendency in the preference of car models during the past 5-10 years?

Yes. The data shows that the majority of cars are between 0–10 years old, with a peak around 3–7 years old. This suggests that Danish buyers prefer relatively recent used cars rather than brand new or very old ones.

The dominance of small, fuel-efficient models like the Ford Fiesta and Toyota Aygo suggests a clear trend towards compact, economical cars over the past decade — likely driven by rising fuel prices and urban living.

---

### 4) Do people in Denmark prefer big or small cars? How reliable is your answer?

The data strongly suggests Danes prefer **small cars**. The most listed models are all compact:
- Ford Fiesta (511 listings)
- Toyota Aygo (447 listings)
- Ford Focus (431 listings)

Furthermore, **86% of all cars have 5 doors**, which is typical for small hatchbacks.

---

### 5) Are there any locations in Denmark where expensive cars are preferred?

Yes. The average car price by region shows a clear pattern:

| Region | Average Price |
|---|---|
| Region Hovedstaden | 158,929 kr |
| Region Syddanmark | 151,535 kr |
| Region Midtjylland | 141,831 kr |
| Region Sjælland | 135,770 kr |
| Region Nordjylland | 135,132 kr |

**Region Hovedstaden (Copenhagen area)** has the highest average car price, which aligns with higher income levels in the capital region.

---

### 6) Which machine learning methods did you choose and why?

**Task 2 - Regression: Linear Regression**
Chosen because we want to predict a continuous numeric value (price). Linear Regression is simple, interpretable, and a good baseline model.

**Task 3 - Clustering: KMeans**
Chosen because it is the most widely used unsupervised algorithm for grouping similar data points. We used the silhouette score to find the optimal number of clusters (k=3).

**Task 4 - Classification: Random Forest Classifier**
Chosen because it handles non-linear relationships well and is robust against overfitting.

---

### 7) How accurate are the prediction models? What do the quality measures mean?

**Regression (Task 2):**
- **MAE (Mean Absolute Error):** The average error in kroner. Easy to understand — "the model is on average X kr off"
- **RMSE (Root Mean Squared Error):** Similar to MAE but penalises large errors more heavily
- **R² (R-squared):** How much of the price variation the model explains. Our R² of ~0.60 means the model explains 60% of price differences

Our regression model had an RMSE of ~65,000–73,000 kr, meaning predictions are typically off by this amount.

**Classification (Task 4):**
- **Accuracy: 91%** — the model correctly classifies 9 out of 10 cars
- **Precision:** When the model predicts a category, how often is it correct?
- **Recall:** Of all cars in a category, how many did the model find?
- **F1-score:** The balance between precision and recall (0–1, higher is better)

Luxury cars scored slightly lower (F1: 0.80) due to fewer training examples.

---

### 8) What could be done to further improve accuracy?

- **Add more features** such as make and model (encoded) to the regression model
- **Use more data** — especially more luxury car examples to improve classification
- **Try more advanced models** such as Gradient Boosting or XGBoost for regression
- **Hyperparameter tuning** — optimise the number of trees in Random Forest
- **Remove outliers** more carefully to reduce noise in the training data
---

### 9) Which were the challenges in the project development?

- **Data quality:** Missing values in mpg and ccm2 required careful handling. Some cars had negative age values (unreleased cars) that needed to be removed.
- **Encoding categorical variables:** Machine learning models only understand numbers, so text columns like make and type needed to be converted.
- **Imbalanced classes:** The luxury category had very few examples (173 out of 2,830), making it harder for the classification model to learn.
- **Choosing the right number of clusters:** Finding the optimal k for KMeans required testing multiple values and evaluating with silhouette scores.
- **Model interpretability:** Explaining what the model has learned in a meaningful way required both visualisation and written analysis.
