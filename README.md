# Predicting_Movie_Rental_Durations
Built a regression model for a DVD rental firm to predict rental duration. Evaluate models to recommend the best one.

## Movie Rental Duration Prediction: Lasso, Linear Regression & Tuned Random Forest (MSE 2.23) | Python, scikit-learn

Built and compared regression models to predict how many days a customer rents a DVD. A tuned random forest reached a test MSE of 2.23 (about 1.5 days RMSE), beating the 3.0 target and a Lasso-selected linear regression at 4.81.

# CONTEXT
A DVD rental company wanted to plan inventory better and asked for a model of rental length from rental and movie features, with a test MSE of 3 or below.

# MY ROLE
I handled preprocessing, feature selection, modelling, tuning and the model recommendation.

# WHAT I DID
• Created the target (rental_length_days) from rental and return timestamps and built dummy variables for Deleted Scenes and Behind the Scenes from the special features text
• Excluded the date columns and the target-derived columns from the features to avoid leakage
• Split the data 80/20 with random_state 9
• Used Lasso (alpha 0.3) to select features, then fitted a linear regression on them 14
• Tuned a random forest with 5-fold RandomizedSearchCV over number of trees (1 to 100) and depth (1 to 10); best was 51 trees and depth 10
• Compared both models on the held-out test set with mean squared error

# RESULTS
• Linear regression on Lasso-selected features: test MSE 4.81
• Random forest: test MSE 2.23, about 54% lower
• Recommended the random forest as best_model
• Baseline MSE from predicting the mean 1.5 days

# TECH STACK
Python, pandas, NumPy, scikit-learn (Lasso, LinearRegression, RandomForestRegressor, RandomizedSearchCV)
