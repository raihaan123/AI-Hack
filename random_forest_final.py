# Step 0: importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pprint import pprint

### PHASE 1: CREATING THE BASE MODEL WITH ALL PARAMETERS

# Step 1: Reading in the data
df = pd.read_csv('boston_corrected_final.csv')
X1 = df[['RM', 'PTRATIO', 'LSTAT', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'AGE', 'DIS', 'RAD', 'TAX', 'B']]
y1 = df['CMEDV']

# Step 2: Splitting the data into a training and test set
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=0.1, random_state=42)

# Step 3: Training the random forest regression model on the whole dataset
regressor1 = RandomForestRegressor(n_estimators=13,random_state=42)
regressor1.fit(X_train1, y_train1)

# Step 4: Predicting the results
y_pred1 = regressor1.predict(X_test1)

# Step 5: Assessing the model's performance
score1 = r2_score(y_test1, y_pred1)
print("The  r squared score is %.2f"  % score1 )

n_estimators = [int(x) for x in np.arange(start = 1, stop = 20, step = 0.5)]
max_features = [0.5,'auto', 'sqrt','log2']
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# First create the base model to tune
m = RandomForestRegressor()
# Fit the random search model
m_random = RandomizedSearchCV(estimator = m, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
m_random.fit(X_train1, y_train1)
pprint(m_random.best_params_)

regressor2 = RandomForestRegressor(n_estimators=13, random_state=42, max_features=0.5, min_samples_leaf=1, bootstrap=False)
regressor2.fit(X_train1, y_train1)
score2 = r2_score(y_test1, y_pred1)
print("The  r squared score is %.2f"  % score2 )

# Step 6: Showing the most important features

#get correlations of each features in dataset
fig1 = plt.figure(1)
corrmat = df.corr()
top_corr_features = corrmat.index
#plt.figure(figsize=(20, 20))
sns.heatmap(df[top_corr_features].corr(), annot=True)

# summarize feature importance
def feat_importance(regressor1,df):
    importance = regressor1.feature_importances_
    importance = pd.DataFrame(importance, index=df.columns, columns=["Importance"])
    return importance.sort_values(by=['Importance'], ascending=False)
importance = feat_importance(regressor1,X_train1)
print(importance)
fig2 = plt.figure(2)
importance.plot(kind='barh')

fig3 = plt.figure(4)
plt.scatter(y_pred1, y_test1)
plt.xlabel('Prediction')
plt.ylabel('Real value')
# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test1), 100)
plt.plot(diagonal, diagonal, '-r')

plt.show()