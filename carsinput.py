import pandas as pd
import requests as rq
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

#------------------------------------------------
#Task 1
#------------------------------------------------

df_cars = pd.read_csv("./CarsSP2/cars.csv")

# 1. Drop unnecessary columns (IDs or index columns if they exist)
columns_to_drop = ["Unnamed: 0", "navn"]
df_cars = df_cars.drop(columns=[col for col in columns_to_drop if col in df_cars.columns])

# 2. Remove duplicate rows
df_cars = df_cars.drop_duplicates()

print(df_cars.isna().sum())
print(df_cars.isna().mean())  #Percentage missing

# Fill numeric columns with median
df_cars["mpg"] = df_cars["mpg"].fillna(df_cars["mpg"].median())
df_cars["ccm2"] = df_cars["ccm2"].fillna(df_cars["ccm2"].median())

# Fill categorical column with "Unknown"
df_cars["region"] = df_cars["region"].fillna("Unknown")

# Removing d and making doors numerical
df_cars["doors"] = df_cars["doors"].str.replace("d", "", regex=False)
df_cars["doors"] = pd.to_numeric(df_cars["doors"])

# Verify
print(df_cars.isna().sum()) #Should now be 0
# Missing numeric values were replaced with the median.
# Missing categorical values were replaced with "Unknown".

bins = [0, 3, 15, 25, float("inf")]
labels = ["Moderne", "Brugt", "Klassisk", "Veteran"]

df_cars["aldercat"] = pd.cut(df_cars["alder"], bins=bins, labels=labels, right=True)
# In aldercat - everything was called veteran, now that is change from age.

#------------------------------------------------
#Task 2
#------------------------------------------------

# 1. Select independent variables (inputs)
X = df_cars[["milage", "alder", "mpg", "ccm2"]]

# 2. Select dependent variable (what we predict)
y = df_cars["price"]     # could also be "mpg" for economy

# 3. Split data into train (70%), validation (15%), test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# 4. Create model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Validate the model
val_predictions = model.predict(X_val)

# 7. Calculate validation errors
print("Validation MAE:", mean_absolute_error(y_val, val_predictions))
print("Validation RMSE:", np.sqrt(mean_squared_error(y_val, val_predictions)))

# 8. Test the final model
test_predictions = model.predict(X_test)

# 9. R² score - how good is the model?
from sklearn.metrics import r2_score
val_r2  = r2_score(y_val,  val_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("Validation R²:", round(val_r2, 3))
print("Test R²:",       round(test_r2, 3))

# 10. Calculate test errors
print("Test MAE:",  mean_absolute_error(y_test, test_predictions))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_predictions)))

# 11. Plot - actual vs predicted price
pd.DataFrame({"Actual": y_test.values, "Predicted": test_predictions}).plot.scatter(x="Actual", y="Predicted", alpha=0.3, title="Actual vs Predicted Price")

#------------------------------------------------
# Model evaluation summary
#------------------------------------------------
# R² (R-squared) shows how well the model explains the variation in car prices.
# An R² of 0.606 on the validation data and 0.548 on the test data means the model
# explains about 55–60% of the variation in prices using the selected features.
# RMSE (Root Mean Squared Error) measures the average prediction error in the same
# units as the price. The RMSE values (~64k–73k) mean the predicted car prices are
# typically off by around 65,000–70,000.
# Since the validation and test results are similar, the model generalizes reasonably
# well and does not appear to be strongly overfitting.

#------------------------------------------------
# Task 3
#------------------------------------------------

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# We use these features to group the cars
X_cluster = df_cars[["price", "milage", "alder", "ccm2"]]

print(X_cluster.describe())

# Remove negative age (cars not yet released)
X_cluster = X_cluster[X_cluster["alder"] >= 0]

# Scale data so all columns have equal weight
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Find the best number of clusters
from sklearn.metrics import silhouette_score

km2 = KMeans(n_clusters=2, random_state=42).fit(X_scaled)
print(f"k=2  score={round(silhouette_score(X_scaled, km2.labels_), 3)}")

km3 = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
print(f"k=3  score={round(silhouette_score(X_scaled, km3.labels_), 3)}")

km4 = KMeans(n_clusters=4, random_state=42).fit(X_scaled)
print(f"k=4  score={round(silhouette_score(X_scaled, km4.labels_), 3)}")

km5 = KMeans(n_clusters=5, random_state=42).fit(X_scaled)
print(f"k=5  score={round(silhouette_score(X_scaled, km5.labels_), 3)}")

# Plot clusters - change km3 to whichever k had the highest silhouette score
X_cluster.assign(cluster=km3.labels_).plot.scatter(x="price", y="milage", c="cluster", colormap="viridis", alpha=0.5, title="Car Clusters - Price vs Milage")

# Add cluster labels back to the dataframe
df_cars_clustered = X_cluster.copy()
df_cars_clustered["cluster"] = km3.labels_  # change km3 to your best k

# Describe each cluster
print(df_cars_clustered.groupby("cluster").mean().round(0))

# Cluster analysis results:
# Cluster 0 - Modern mid-range cars: ~172,000 kr, 4 years old, 65,000 km
#             Relatively new, moderately driven. Typical used family car.
#
# Cluster 1 - Budget older cars: ~61,000 kr, 16 years old, 189,000 km
#             Old and heavily driven. Typical budget purchase.
#
# Cluster 2 - Expensive luxury/classic cars: ~487,000 kr, 21 years old, 4111 ccm
#             High price despite age - likely classic or exclusive cars
#             that retain their value due to large engine and brand.
#
# Recommendation: k=3 provides the best segmentation based on silhouette score
# and produces three clearly distinguishable and meaningful car segments.

#------------------------------------------------
# Task 4
#------------------------------------------------
# Create price categories
bins  = [0, 100000, 300000, float("inf")]
labels = ["budget", "mid-range", "luxury"]
df_cars["price_cat"] = pd.cut(df_cars["price"], bins=bins, labels=labels)

print(df_cars["price_cat"].value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# X = same features as before, Y = price category
X_clf = df_cars[["milage", "alder", "mpg", "ccm2"]]
y_clf = df_cars["price_cat"]

# Split into train and test
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# Life biggest question :D
clf.fit(X_train_clf, y_train_clf)

# Test the model
y_pred_clf = clf.predict(X_test_clf)

# Evaluate
print(classification_report(y_test_clf, y_pred_clf))

# Classification results:
# Overall accuracy: 91% - the model correctly classifies 9 out of 10 cars
# Budget and mid-range perform well (F1: 0.93 and 0.91)
# Luxury performs slightly worse (F1: 0.80) due to fewer training examples (173 cars)
# The model is reliable for budget and mid-range classification

#------------------------------------------------
# Task 5
#------------------------------------------------

import joblib

# Save the regression model (Task 2)
joblib.dump(model, "model_regression.pkl")

# Save the clustering model (Task 3)
joblib.dump(km3, "model_clustering.pkl")  # change km3 to your best k

# Save the classification model (Task 4)
joblib.dump(clf, "model_classification.pkl")

print("All models saved!")

# Load and test the models
loaded_regression = joblib.load("model_regression.pkl")
loaded_classification = joblib.load("model_classification.pkl")

# Predict price for a test car: milage=50000, alder=3, mpg=20, ccm2=1500
test_car = pd.DataFrame([[50000, 3, 20, 1500]], columns=["milage", "alder", "mpg", "ccm2"])

print("Predicted price:", loaded_regression.predict(test_car).round(0))
print("Predicted category:", loaded_classification.predict(test_car))
