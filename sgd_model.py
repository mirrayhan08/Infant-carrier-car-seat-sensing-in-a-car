import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

file = "E:\\01 - FRANKFURT LECTURE SLIDE\\Machine Learning\\vehicale_baby_seat_detection\\Data Collection\\dataset.csv"
dataset = pd.read_csv(file)

X = dataset.drop('label', axis=1)
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

params = {
    "loss": ["hinge", "log", "squared_hinge", "modified_huber", "perceptron"],
    "alpha": [0.001, 0.01, 0.1],
    "penalty": ["l2", "l1", "elasticenet", "none"],
    "class_weight": [{0: 0.1, 1: 1.0}, {0: 0.5, 1: 1.0}, {0: 1.0, 1: 1.0}],
}

model = SGDClassifier()

model_cv = GridSearchCV(model, param_grid=params, cv=10, scoring='accuracy')

model_cv.fit(X_train, y_train)

best_params = model_cv.best_params_
best_estimator = model_cv.best_estimator_

# Make predictions
y_pred = best_estimator.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)

print("Prediction on test dataset: {}".format(acc_score))


