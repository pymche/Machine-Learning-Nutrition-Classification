# Nutrition Facts Classification

[Link to Report](https://github.com/pymche/Machine-Learning-Nutrition-Classification/blob/master/Nutrition_Analysis.ipynb)

Train models using nutritional information to classify into various food groups.

Data obtained from [My Food Data](https://tools.myfooddata.com/nutrition-facts-database-spreadsheet.php)

#### Preview of raw data
![Preview](https://i.imgur.com/mPxcRaQ.png)

#### About the Data

Nutritional information such as Carbohydrates, Sugar, Saturated Fats, Protein, Calories are used as features, while Food Group, i.e. Meat, Beverages, Baked Goods etc. is the target value. The size of the data is 14167 rows × 117 columns.

#### Preprocessing

All features (except Food Group) are of numerical values, so there was no need to encode any attributes for classification models. Missing values were dealth with removal and imputation (depending on how many missing values there are in each column), scaling is employed before classification.

#### Analysis

Exploratory analysis and data visualisation are also available to explore the dataset.
![Example](https://i.imgur.com/K19LOWm.png)

#### Modelling

Techniques including Feature Selection, Preprocessing with Pipeline, Cross Validation and Parameter Tuning are used in this report. 

#### Models used: 
Decision Tree, Random Forest, Stochastic Gradient Descent, Naive Bayes, K-Nearest Neighbors, Gradient Boosting, Multi-Layer Perception.

#### Results
![Chart](https://i.imgur.com/RPNQj6u.png)
![Results](https://i.imgur.com/Gh8PUXV.png)
