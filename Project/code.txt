# Step 1: Import library and function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Sample movie data
data = {
    'Movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy'],
    'Rating': [4.5, 3.8, 4.2, 4.0, 3.5],
    'Runtime': [120, 90, 110, 130, 100],
    'Budget': [100, 50, 70, 120, 60]
}
df = pd.DataFrame(data)

#Display Dataset
df

# Convert categorical features (Genre) into numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Genre'], drop_first=True)

# Step 3: Define features (X) and target (y)
X = df.drop(['Movie', 'Rating'], axis=1)
y = df['Rating']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Coefficients and Intercept
model.intercept_ , model.coef_

# Step Make predictions on the test set
y_pred = model.predict(X_test)
y_pred

# Example recommendation:
# Create a new movie profile
new_movie = pd.DataFrame({
    'Runtime': [115],
    'Budget': [80],
    'Genre_Comedy': [0],
    'Genre_Drama': [1]
})

# Step 8: Calculate Mean Squared Error and R-squared
mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

predicted_rating = model.predict(new_movie)
print(f"Predicted Rating for the new movie: {predicted_rating[0]}")

plt.scatter(X_test['Runtime'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Runtime'], y_pred, color='red', label='Predicted')
plt.xlabel('Runtime') 
plt.ylabel('Rating')
plt.title('Rating Prediction') 
plt.legend()
plt.show()
