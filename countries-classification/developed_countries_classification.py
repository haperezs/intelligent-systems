import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('data.csv', header = 0)

# Remove space in column names
df.rename(columns = lambda x: x.strip().replace(' ', '_').lower(), inplace=True)

# Fill nulls with a mean
df = df.fillna(df.mean())

# Split into X and y
y = df[['status']]
X = df.drop(columns=['status', 'year'])

# Remove the non-numeric columns (Country)
X = X._get_numeric_data()

print(y)
print(X)
print(df.isnull().sum())

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train)
print(y_train)

# Correlation
corr = df[['status', 'life_expectancy', 'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure', 'hepatitis_b',
           'measles', 'bmi', 'under-five_deaths', 'polio', 'total_expenditure', 'diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness__1-19_years',
           'thinness_5-9_years', 'income_composition_of_resources', 'schooling']].corr()

plt.figure(figsize=(15,12))
sns.heatmap(corr, square=True, annot=True, cmap='viridis');

plt.figure(figsize=(8,6))

plt.subplot(2,2,1)
ax1 = sns.scatterplot(x='life_expectancy', y='hiv/aids', data=df, color='blue')

plt.subplot(2,2,2)
ax1 = sns.scatterplot(x='life_expectancy', y='adult_mortality', data=df, color='magenta')

plt.subplot(2,2,3)
ax1 = sns.scatterplot(x='life_expectancy', y='income_composition_of_resources', data=df, color='green')

plt.subplot(2,2,4)
ax1 = sns.scatterplot(x='life_expectancy', y='schooling', data=df, color='orange')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))

plt.subplot(2,2,1)
ax1 = sns.scatterplot(x='income_composition_of_resources', y='schooling', data=df, color='blue')

plt.subplot(2,2,2)
ax1 = sns.scatterplot(x='percentage_expenditure', y='gdp', data=df, color='magenta')

plt.subplot(2,2,3)
ax1 = sns.scatterplot(x='under-five_deaths', y='infant_deaths', data=df, color='green')

plt.subplot(2,2,4)
ax1 = sns.scatterplot(x='thinness__1-19_years', y='thinness_5-9_years', data=df, color='orange')

plt.tight_layout()
plt.show()

# Standard scalar normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA
pca = PCA(n_components=19)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(
    'The percentage of total variance in the dataset explained by each component from Sklearn PCA: \n',
    pca.explained_variance_ratio_,

)

plt.figure(figsize=(20,10))

plt.subplot(122)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
ax1.set_ylabel('% of Variance Explained')
ax1.set_xlabel('Components')
ax1.set_ylim(0,1);

plt.show()

# Single tree
classifier = tree.DecisionTreeClassifier(max_depth=5, random_state=0)
classifier.fit(X_train, y_train.values.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy
print(f'Single tree\nAccuracy {accuracy_score(y_test, y_pred)*100}')

plt.subplots(figsize=(24, 12))
tree.plot_tree(classifier, fontsize=10)
plt.show()

# Random forest with all features
classifier = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=12)
classifier.fit(X_train, y_train.values.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy
print(f'Random forest\nAccuracy {accuracy_score(y_test, y_pred)*100}')

# Random forest with PCA
pca = PCA(n_components=12)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

classifier = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=12)
classifier.fit(X_train_pca, y_train.values.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test_pca)

# Accuracy
print(f'Random forest with PCA\nAccuracy {accuracy_score(y_test, y_pred)*100}')

# Queries
X_query = None
'''
	life_expectancy', 'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure', 'hepatitis_b',
  'measles', 'bmi', 'under-five_deaths', 'polio', 'total_expenditure', 'diphtheria', 'hiv/aids', 'gdp', 'population', 'thinness__1-19_years',
  'thinness_5-9_years', 'income_composition_of_resources', 'schooling'
'''

# Developing
#X_query = [[65,	263,	62,	0.01,	71.27962362,	65,	1154,	19.1,	83,	6,	8.16,	65,	0.1,	584.25921,	33736494,	17.2,	17.3,	0.479,	10.1]]
X_query = [[63,	26,	8,	0.01,	80.92679802,	84,	14,	3.1,	12,	84,	3.77,	84,	0.9,	1326.66882,	46392,	8,	7.7,	0.509,	8.5]]

# Developed
#X_query = [[83,	8,	0,	10.11,	713.5297354,	97,	576,	62.3,	1,	98,	1.42,	98,	0.1,	4772.77415,	1147744,	0.9,	0.9,	0.884,	16.1]]
#X_query = [[79.3,	86,	0,	12.1,	5316.877456,	83,	15,	52.2,	0,	83,	1.56,	83,	0.1,	36693.4262,	8171966,	1.7,	1.9,	0.841,	14.7]]

X_query = sc.transform(X_query)
X_query = pca.transform(X_query)

y_query = classifier.predict(X_query)
print(y_query)
