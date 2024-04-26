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


results = {} # It stores result of classification algorithms.
# Fetching dataset
#dataset = fetch_ucirepo(id=15)
column_names = ["ID", "Clump_Thickness", "Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape",
                "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
                "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"]

# Reading csv file
dataset = pd.read_csv('./breast cancer/breast-cancer-wisconsin.data', names= column_names)

dataset.replace("?", np.nan, inplace=True)
dataset['Bare_Nuclei'] = pd.to_numeric(dataset['Bare_Nuclei'])
dataset.fillna(dataset.mean(), inplace=True) 
x = dataset.loc[:, 'Clump_Thickness': 'Mitoses']
y = dataset['Class']

# in the command line below, it normalizes numeric data and draws outlier graph
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
# remove_outliers_iqr function removes outliers in the dataset
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

#Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(data_cleaned_x, data_cleaned_y, test_size=0.2, random_state=42)

# Using supervised algorithms to figure out which algoritm gives us the best accuracy result
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Random Forest Classifier')
print("Accuracy:", accuracy)
results['RandomForest'] = accuracy
print('------------------------------------------------------------------')

print('ExtraTreesClassifier')
model = sklearn.ensemble.ExtraTreesClassifier(100, random_state= 42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
results['ExtraTrees'] = accuracy
print('------------------------------------------------------------------')

print('Gradient Boosting Classifier')
model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate= 1.0, max_depth=1, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
results['GradientBoosting'] = accuracy

print('------------------------------------------------------------------')

print('Bagging Classifier')
decision_tree = sklearn.tree.DecisionTreeClassifier(criterion= 'gini')
model = sklearn.ensemble.BaggingClassifier(estimator=decision_tree, n_estimators=100 )
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
results['Bagging'] = accuracy
print('------------------------------------------------------------------')

print('Ada Boost Classifier')
model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
results['AdaBoost'] = accuracy

print('------------------------------------------------------------------')

print('Support Vector Machines')
model = svm.SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
results['SVM'] = accuracy
print('------------------------------------------------------------------')

print('Naive Bayes(Gaussian)')
model = GaussianNB() 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
results['NaiveBayes(Gaussian)'] = accuracy
print('------------------------------------------------------------------')

print('Naive Bayes(Bernoulli)')
model = BernoulliNB() 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
results['NaiveBayes(Bernoulli)'] = accuracy

print('------------------------------------------------------------------')

print('K-NN')
model = KNeighborsClassifier() 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
results['KNeighbors'] = accuracy

print('------------------------------------------------------------------')

print('MLP Classifier')
model = MLPClassifier(max_iter= 500)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
results['MLP'] = accuracy

print('------------------------------------------------------------------')

y_values = results.values()
y_values = [element * 100 for element in y_values]
max_percentage = y_values.index(max(y_values))
x_values = results.keys()
x_values = [element for element in x_values]

# Draw the accuracy table by using algorithms that we have used and find the algorithm
# that gives us the best accuracy result (Best accuracy result is represented red dot in the scatter plot)
plt.figure(figsize=(12, 8))  
plt.scatter(x_values, y_values)
plt.scatter(x_values[max_percentage], y_values[max_percentage], color='red')  
plt.ylabel('Percentage')
plt.xlabel('Algorithms')
plt.xticks(ticks=range(0, len(y_values)) , labels=x_values, rotation='vertical', fontsize= 10)
plt.show()



