import numpy as np
import pandas as pd
import sklearn.svm
from sklearn import svm
import sklearn.ensemble 
import sklearn.tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error 
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 


results = {}
# fetch dataset 
#dataset = fetch_ucirepo(id=15)
column_names = ["ID", "Clump_Thickness", "Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape",
                "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
                "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"]

dataset = pd.read_csv('./breast cancer/breast-cancer-wisconsin.data', names= column_names)

dataset.replace("?", np.nan, inplace=True)
dataset['Bare_Nuclei'] = pd.to_numeric(dataset['Bare_Nuclei'])
dataset.fillna(dataset.mean(), inplace=True) 
x = dataset.loc[:, 'Clump_Thickness': 'Mitoses']
y = dataset['Class']

'''
scaler= StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled)
'''
'''
plt.figure(figsize=(10, 8))
x.boxplot()
plt.xticks(rotation=45, fontsize= 8)
plt.title("Box plot of numerical features")
plt.show()
'''

def remove_outliers_iqr(df ):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
    return df[~outliers]

removed_id = dataset.loc[:, 'Clump_Thickness': 'Class']

data_cleaned = remove_outliers_iqr(removed_id)
data_cleaned_x = data_cleaned.loc[:, 'Clump_Thickness': 'Mitoses']
data_cleaned_y = data_cleaned['Class']

x_train, x_test, y_train, y_test = train_test_split(data_cleaned_x, data_cleaned_y, test_size=0.2, random_state=42)

print('K-NN')
print('Default parameters')
model = KNeighborsClassifier() 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('------------------------------------------------------------------')
print('n_neighbours: 2  algorithm: ball tree  leaf_size: 20  p:2')
model = KNeighborsClassifier(n_neighbors= 2, algorithm='ball_tree', leaf_size= 20, p=2) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('------------------------------------------------------------------')
print('n_neighbours: 10  algorithm: ball tree  leaf_size: 20  p:2')
model = KNeighborsClassifier(n_neighbors= 2, algorithm='ball_tree', leaf_size= 20, p=1) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('------------------------------------------------------------------')
print('n_neighbours: 2  algorithm: ball tree  leaf_size: 50  p:2')
model = KNeighborsClassifier(n_neighbors= 2, algorithm='ball_tree', leaf_size= 50, p=2) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('------------------------------------------------------------------')
print('n_neighbours: 2  algorithm: ball tree  leaf_size: 20  p:1')
model = KNeighborsClassifier(n_neighbors= 2, algorithm='ball_tree', leaf_size= 20, p=1) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


print('------------------------------------------------------------------')
print('n_neighbours: 2  algorithm: kd tree  leaf_size: 20  p:2')
model = KNeighborsClassifier(n_neighbors= 2, algorithm='kd_tree', leaf_size= 20, p=2) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('------------------------------------------------------------------')
print('n_neighbours: 10  algorithm: kd tree  leaf_size: 20  p:2')
model = KNeighborsClassifier(n_neighbors= 10, algorithm='kd_tree', leaf_size= 20, p=2) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('------------------------------------------------------------------')
print('n_neighbours: 2  algorithm: kd tree  leaf_size: 50  p:2')
model = KNeighborsClassifier(n_neighbors= 10, algorithm='kd_tree', leaf_size= 50, p=2) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('------------------------------------------------------------------')
print('n_neighbours: 2  algorithm: algorithm: brute force  leaf_size: 20  p:1')
model = KNeighborsClassifier(n_neighbors= 2, algorithm='brute', leaf_size= 20, p=1) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


print('------------------------------------------------------------------')
print('n_neighbours: 10  algorithm: brute force  leaf_size: 20  p:2')
model = KNeighborsClassifier(n_neighbors= 10, algorithm='brute', leaf_size= 20, p=2) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('------------------------------------------------------------------')
print('n_neighbours: 2  algorithm: brute force  leaf_size: 50  p:2')
model = KNeighborsClassifier(n_neighbors= 10, algorithm='brute', leaf_size= 50, p=2) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print('------------------------------------------------------------------')
print('n_neighbours: 2  algorithm: brute force  leaf_size: 20  p:1')
model = KNeighborsClassifier(n_neighbors= 2, algorithm='brute', leaf_size= 20, p=1) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)