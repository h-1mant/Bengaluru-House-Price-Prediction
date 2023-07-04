# House Price Prediction using Elastic Net Regression

Robust Model Development for House Price Prediction in Bangalore,India

### Process Flow
- Data preprocessing and Feature Selection
- Elastic Net Regularization 
- GridSearch Cross Validation
- Model Testing (RMSE, MAE, Residual Plot, Normality Check)

### Libraries Used
- Numpy
- Pandas
- Matplotlib
- Seaborn

Description
---
This project report presents the development of a machine learning model that utilizes ElasticNet regression to predict house prices in Bangalore for the first nine months of the year 2020. The model's performance was evaluated using 5-fold crossvalidation, and the R2 score and RMSE were used as performance metrics. The original dataset used in this study contained approximately 13,300 instances. The model was trained using Gradient Descent to learn the model parameters. Before applying the model, all assumptions for linear regression were tested.

Data
---
The dataset contains information on house prices in Bengaluru, India, from January 2020 to September 2020. 
Attributes include
- **area_type**: The type of area where the property is located (e.g., built-up area, super built-up area).
- **availability**: Whether the property is currently available for purchase or not.
- **location**: The name of the area or locality where the property is located.
- **size**: The size of the property in terms of the number of bedrooms.
- **society**: The name of the housing society or apartment complex.
- **total_sqft**: The total square footage of the property.
- **bath**: The number of bathrooms in the property.
- **balcony**: The number of balconies in the property.
- **price**: The price of the property in Indian rupees (INR). 

Summary
---
The study identified the factors that significantly influence house prices in Bangalore and developed a robust model for accurate price prediction. The model's performance was evaluated using the RMSE benchmark, with a result of approximately 15%, indicating that the predicted prices were within 20% of the actual price, satisfying the benchmark. Despite the promising results, the study identified issues with Heteroskedasticity in the model output. Further development to address the identified issues and improve the model's performance may be performed considering the following:
- Heteroskedasticity issue with dataset, outliers
- Training data limited to samples from 2020
- Decision Tree based approach exploration, with Bagging
- A more refined Benchmark
