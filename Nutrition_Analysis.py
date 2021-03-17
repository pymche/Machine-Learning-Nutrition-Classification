import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from collections import Counter

df = pd.read_excel('nut_data.xlsx')
df

df.columns = df.loc[2].values

df.drop(index=[0, 1, 2], inplace=True)

df.reset_index(inplace=True)

df.drop(columns=['index', 'ID'], inplace=True)

df.isna()

# Dropping columns which do not have any values

df.dropna(axis='columns', how='all', inplace = True)

# check number of columns that have at least 25% missing values 

count = 0
bad_features = []
for x in df.columns:
    if df[x].isna().sum() > 0.23*df.shape[0]:
        count += 1
        bad_features.append(x)
        
print(f'Number of columns that have at least 25% missing values: {count}')
print(f'Columns that have at least 25% missing values: {bad_features}')

bad_features.append('name')

# Dropping columns that have at least 25% missing values

df = df.drop(columns=bad_features)

df.columns

# dropping columns with description

df = df.drop(columns=['Serving Weight 1 (g)',
       'Serving Description 1 (g)'])

X = df.drop(columns='Food Group')
y = df['Food Group']

df[X.columns] = df[X.columns].astype(float)

# The data is first split into training and testing data before feature selection, to prevent data leakage

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Imputer to fill in missing values

imputed_X_train = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X_train))
imputed_X_train.columns = X_train.columns

y_train.fillna('Missing', inplace=True)

# Feature Selector

selector = SelectKBest(f_classif, k=20)
X_fit = selector.fit(imputed_X_train, y_train)
X_new = selector.fit_transform(imputed_X_train, y_train)

# Check score of each selected feature

df_scores = pd.DataFrame(X_fit.scores_)
df_columns = pd.DataFrame(X_train.columns)

feature_scores = pd.concat([df_columns, df_scores],axis=1)
feature_scores.columns = ['Feature Name','Score']

feature_scores.set_index('Feature Name', inplace=True)
feature_scores.head(20).sort_values(by='Score', ascending=False)

# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_new), columns=X_train.columns)

# Dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0]
print(selected_columns)
print(selected_columns.shape)

# Create new dataframe from original data set, but with selected features

new_df = pd.concat([df['Food Group'], df[selected_columns]], axis=1)
new_df

# Dropping rows with missing target value (Food Group)

index = new_df.loc[df['Food Group'].isnull()].index

new_df = new_df.drop(index=index)

# Splitting into training and test sets

X = new_df.drop(columns='Food Group')
y = new_df['Food Group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

new_df

print(new_df['Food Group'].unique())

plt.figure(figsize=(10,10))
new_df['Food Group'].value_counts().plot(kind='pie')

new_df.describe()

top = new_df.columns[1:9]
print(top)

plt.figure()
new_df[top].hist(figsize=(20, 20), bins=30)

plt.figure(figsize=(20,10))
ax = sns.boxplot(x="Food Group", y="Calories", data=new_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

for group in new_df['Food Group'].unique():
    nut_info = []
    mean = []
    filt = (new_df['Food Group'] == group)
    filtered = new_df.loc[filt]
    for item in ['Fat (g)', 'Protein (g)', 'Carbohydrate (g)',
       'Sugars (g)', 'Fiber (g)', 'Saturated Fats (g)']:
        nut_info.append(item)
        mean.append(filtered[item].mean())
    
    plt.pie(mean, labels=nut_info, autopct='%1.1f%%') 
    plt.title(f'Nutritional Information of {group}')
    plt.tight_layout() 
    plt.show()

# Setting up pipeline

# Preprocessing
process_cols = X.columns
process_t = Pipeline(steps=[
    ('Impute', SimpleImputer()), 
    ('Scaling', MinMaxScaler())])

preprocessor = ColumnTransformer(transformers=[('preprocess', process_t, process_cols)])

# Models
classifiers = [
    SGDClassifier(),
    GaussianNB(),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier()
]

# Pipeline

# Find out best models

models = []
scores = []
best_models = []


for model in classifiers:
    print(model)
    pipeline = Pipeline(steps=[('preprocess', preprocessor), ('m', model)])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(score)
    
    models.append(model)
    scores.append(score)
    
    if score >= 0.75:
        best_models.append(model)

best_models

models = [str(x).strip('()') for x in models]
# models

df_scores = pd.DataFrame(scores)
df_models = pd.DataFrame(models)

model_scores = pd.concat([df_models, df_scores],axis=1)
model_scores.columns = ['Model','Score']

model_scores.set_index('Model', inplace=True)
model_scores = model_scores.sort_values(by='Score', ascending=False)
model_scores

plt.figure(figsize=(12,5))
plt.barh(models, scores) 
plt.title('Performance of Models') 
plt.ylabel('Models') 
plt.xlabel('Performance accuracy score') 
plt.tight_layout()
plt.show()

print(f'Models with 75% or above accuracy: {best_models}')

# Fill missing values

imputed_X_train = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X_train))
preprocessed_X_train = pd.DataFrame(MinMaxScaler().fit_transform(imputed_X_train))
preprocessed_X_train.columns = X_train.columns

# Cross Validation

cross_vali = []

for model in best_models:
    pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', model)])
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    mean = cv_scores.mean()
    cross_vali.append(mean)
    print(f'Accuracy score mean of {model}: {mean}')

best_models_names = [str(x).strip('()') for x in best_models]
# best_models_names
before = model_scores.loc[best_models_names]
# before

after = pd.DataFrame(cross_vali)
after_name = pd.DataFrame(best_models_names)
after = pd.concat([after_name, after], axis=1)
after.columns = ['Model', 'Score']
after.set_index('Model', inplace=True)
# after

crossval_compare = pd.concat([before, after], axis=1)
crossval_compare.columns = ['Before Cross Validation','After Cross Validation']
crossval_compare

crossval_compare.plot(kind='bar')

para = [x for x in range(5, 10)]
print(para)

# Parameter Tuning

classifier = RandomForestClassifier()

parameter_grid = {'max_depth': para,
                  'max_features': para,
                 'criterion': ['gini', 'entropy']}

cross_validation = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(preprocessed_X_train, y_train)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

classifier = DecisionTreeClassifier()

parameter_grid = {'max_depth': para,
                  'max_features': para,
                 'criterion': ['gini', 'entropy'],
                 'splitter': ['best', 'random'] }

cross_validation = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(preprocessed_X_train, y_train)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

classifier = KNeighborsClassifier()

parameter_grid = {'n_neighbors': para,
                  'weights': ['uniform', 'distance'],
                 'algorithm': ['ball_tree', 'kd_tree', 'brute']}

cross_validation = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(preprocessed_X_train, y_train)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

classifier = GradientBoostingClassifier()

parameter_grid = {'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
                  'n_estimators':[100,250,500,750,1000,1250,1500,1750]}

cross_validation = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(preprocessed_X_train, y_train)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

new_df.isnull()

X = new_df.drop(columns='Food Group')
y = new_df['Food Group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess training set

imputed_X_train = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X_train))
preprocessed_X_train = pd.DataFrame(MinMaxScaler().fit_transform(imputed_X_train))
preprocessed_X_train.columns = X_train.columns

preprocessed_X_train

# Preprocess test set

imputed_X_test = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X_test))
preprocessed_X_test = pd.DataFrame(MinMaxScaler().fit_transform(imputed_X_test))
preprocessed_X_test.columns = X_test.columns

y_train.isnull().sum()

# Fitting the model and generating its accuracy score

classifier = RandomForestClassifier(criterion='entropy', max_depth=9, max_features=9)
classifier.fit(preprocessed_X_train, y_train)
score = classifier.score(preprocessed_X_test, y_test)
print(score)

# Predicting the first 20 classes with the classifier

predictions = classifier.predict(preprocessed_X_test)
y_test = np.array(y_test)
for x in range(20):
    print(f'Prediction: {predictions[x]} ------ Actual Value: {y_test[x]}')

pred_count = Counter(predictions)
print(pred_count)
pred_count = pd.Series(pred_count)

actual_count = Counter(y_test)
print(actual_count)
actual_count = pd.Series(actual_count)

# Results comparison - Counts of Prediction vs Actual classification

results_comp = pd.concat([pred_count, actual_count], axis = 'columns')
results_comp.columns = ['Prediction', 'Actual Food Group']
results_comp

results_comp.plot(kind='bar', figsize=(20,20))
