from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV


file = "E:\\01 - FRANKFURT LECTURE SLIDE\\Machine Learning\\vehicale_baby_seat_detection\\Data Collection\\dataset.csv"
dataset = pd.read_csv(file)

X = dataset.drop('label', axis=1)
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

params = {
    'C': [0.00001, 0.001, 0.1, 1, 10],
    "kernel": ["rbf", "linear", "sigmoid"]
}

SVM = svm.SVC()

svm_cv = GridSearchCV(SVM, param_grid=params, cv=10)
svm_cv.fit(X_train, y_train)

print(SVM)


best_params = svm_cv.best_params_
best_estimator = svm_cv.best_estimator_

# Make predictions
predictions = best_estimator.predict(X_test)

print("Prediction on test dataset: {}".format(accuracy_score(y_test, predictions)))

