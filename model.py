
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x=pd.read_csv("concrete_data.csv")
x.head(12)

print(x.shape)
x.describe()

"""## Correlation Data measure"""

sns.heatmap(x.corr(),annot=True)

"""## Handling Missing values"""

print(x.isnull().sum())
print("")
print("-----------Missing values in %------------")
print((x.isnull().sum()/x.shape[0])*100)

"""## Handling Outliers

* blast_furnace_slag
"""

plt.rcParams['figure.figsize']=(18,8)
sns.set(style='whitegrid')
sns.boxplot(y=x['blast_furnace_slag'])
plt.xlabel('Distribution of blast_furnace_slag',fontsize=20)
plt.title('Boxplot for blast_furnace_slag')
plt.show()

x['blast_furnace_slag'].values[x['blast_furnace_slag'].values>345]=x['blast_furnace_slag'].median()

plt.rcParams['figure.figsize']=(18,8)
sns.set(style='whitegrid')
sns.boxplot(y=x['blast_furnace_slag'])
plt.xlabel('Distribution of blast_furnace_slag',fontsize=20)
plt.title('Boxplot for blast_furnace_slag')
plt.show()

"""* cement
* fly_wash
"""

plt.rcParams['figure.figsize']=(18,8)

plt.subplot(1,2,1)
sns.set(style='whitegrid')
sns.boxplot(y=x['cement'])
plt.title('Distribution of cement',fontsize=20)

plt.subplot(1,2,2)
sns.boxplot(y=x['fly_ash'])
sns.set(style='whitegrid')
plt.title('Distribution of fly_ash',fontsize=20)
plt.show()

"""* Water
* superplasticizer
"""

plt.rcParams['figure.figsize']=(18,8)

plt.subplot(1,2,1)
sns.set(style='whitegrid')
sns.boxplot(y=x['water'])
plt.title('Distribution of water',fontsize=20)

plt.subplot(1,2,2)
sns.boxplot(y=x['superplasticizer'])
sns.set(style='whitegrid')
plt.title('Distribution of superplasticizer',fontsize=20)
plt.show()

x['water'].values[x['water'].values>230]=x['water'].median()
x['water'].values[x['water'].values<127]=x['water'].median()
x['superplasticizer'].values[x['superplasticizer'].values>25]=x['superplasticizer'].median()

plt.rcParams['figure.figsize']=(18,8)

plt.subplot(1,2,1)
sns.set(style='whitegrid')
sns.boxplot(y=x['water'])
plt.title('Distribution of water',fontsize=20)

plt.subplot(1,2,2)
sns.boxplot(y=x['superplasticizer'])
sns.set(style='whitegrid')
plt.title('Distribution of superplasticizer',fontsize=20)
plt.show()

"""* coarse_aggregate
* concrete_compressive_strength
"""

plt.rcParams['figure.figsize']=(18,8)

plt.subplot(1,2,1)
sns.set(style='whitegrid')
sns.boxplot(y=x['coarse_aggregate'])
plt.title('Distribution of coarse_aggregate',fontsize=20)

plt.subplot(1,2,2)
sns.boxplot(y=x['concrete_compressive_strength'])
sns.set(style='whitegrid')
plt.title('Distribution of concrete_compressive_strength',fontsize=20)
plt.show()

x['concrete_compressive_strength'].values[x['concrete_compressive_strength'].values>=79]=x['concrete_compressive_strength'].median()

plt.rcParams['figure.figsize']=(18,8)

plt.subplot(1,2,1)
sns.set(style='whitegrid')
sns.boxplot(y=x['coarse_aggregate'])
plt.title('Distribution of coarse_aggregate',fontsize=20)

plt.subplot(1,2,2)
sns.boxplot(y=x['concrete_compressive_strength'])
sns.set(style='whitegrid')
plt.title('Distribution of concrete_compressive_strength',fontsize=20)
plt.show()



"""## Data Visualization

* Distribution of blast_furnace_slag and fly_ash
"""

plt.rcParams['figure.figsize']=(18,8)
plt.subplot(1,2,1)
sns.set(style='whitegrid')
sns.distplot(x['blast_furnace_slag'])
plt.title('Distribution of blast_furnace_slag',fontsize=20)
plt.xlabel('Range of blast_furnace_slag')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.set(style='whitegrid')
sns.distplot(x['fly_ash'],color='red')
plt.title('Distribution of fly_ash',fontsize=20)
plt.xlabel('Range of fly_ash')
plt.ylabel('Count')
plt.show()

"""* Distribution of water and superplasticizer"""

plt.rcParams['figure.figsize']=(18,8)
plt.subplot(1,2,1)
sns.set(style='whitegrid')
sns.distplot(x['water'])
plt.title('Distribution of water',fontsize=20)
plt.xlabel('Range of water')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.set(style='whitegrid')
sns.distplot(x['superplasticizer'],color='red')
plt.title('Distribution of superplasticizer',fontsize=20)
plt.xlabel('Range of superplasticizer')
plt.ylabel('Count')
plt.show()

"""* Distribution of coarse_aggregate and concrete_compressive_strength"""

plt.rcParams['figure.figsize']=(18,8)
plt.subplot(1,2,1)
sns.set(style='whitegrid')
sns.distplot(x['coarse_aggregate'])
plt.title('Distribution of coarse_aggregate',fontsize=20)
plt.xlabel('Range of coarse_aggregate')
plt.ylabel('Count')

plt.subplot(1,2,2)
sns.set(style='whitegrid')
sns.distplot(x['concrete_compressive_strength'],color='red')
plt.title('Distribution of concrete_compressive_strength',fontsize=20)
plt.xlabel('Range of concrete_compressive_strength')
plt.ylabel('Count')
plt.show()

"""* Count of Age"""

plt.rcParams['figure.figsize']=(18,8)
plt.hist(x['age'],bins=10,edgecolor='black')
plt.xlabel("age")
plt.ylabel("Count")
plt.title("age vs Count")
plt.show()

"""* Distribution of water vs superplasticizer"""

plt.rcParams['figure.figsize']=(18,8)
sns.set(style='whitegrid')
plt.scatter(x['water'],x['superplasticizer'],s=80,edgecolor='black',linewidth=1,alpha=0.75)
plt.grid(True)
plt.xlabel("Distribution of Water")
plt.ylabel("Distribution of superplasticizer")
plt.title("water vs superplasticizer")
plt.show()

"""## Model Training"""

x.columns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

X=x.drop('concrete_compressive_strength',axis=1).values
Y=x['concrete_compressive_strength'].values

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=.2, random_state=42 )

ss=StandardScaler()
X_train_ss=ss.fit_transform(X_train)
X_test_ss=ss.fit_transform(X_test)

"""### Linear Regression"""

model=LinearRegression()
model.fit(X_train,y_train)
print(model.coef_)
model.intercept_
model.score(X_test,y_test)

"""## Gradient Boosting

"""

X=x.drop('concrete_compressive_strength',axis=1).values
Y=x['concrete_compressive_strength'].values

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=.2, random_state=0 )

ss=StandardScaler()
X_train_ss=ss.fit_transform(X_train)
X_test_ss=ss.fit_transform(X_test)

params = {
    'learning_rate' : [0.01, 0.1, 1.0],
    'n_estimators' : [100, 150, 200],
    'max_depth' : [3, 4, 5]
}

model=GradientBoostingRegressor()
model.fit(X_train,y_train)
model.score(X_test,y_test)

model2 = GridSearchCV(model, params)
model2.fit(X_train, y_train)

model2.score(X_test,y_test)

model2.predict([[233,86,63,84,9,930,676,270]]).reshape(-1,1).astype(int)

import pickle
pickle.dump(model2,open('model.pkl','wb'))