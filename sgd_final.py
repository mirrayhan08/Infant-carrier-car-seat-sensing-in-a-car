import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
#from google.colab import drive
#drive.mount('/content/drive')

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#with tf.device('/GPU:0'):
file = "E:\\01 - FRANKFURT LECTURE SLIDE\\Machine Learning\\vehicale_baby_seat_detection\\features.csv"
dataset = pd.read_csv(file)

X = dataset.drop('Label', axis=1)
y = dataset['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

params = {
    "loss": ["hinge", "log_loss", "modified_huber", "perceptron"],
    "alpha": [0.001, 0.01, 0.1],
    "penalty": ["l2", "elasticnet", None],
    "class_weight": ['balanced'],
    "max_iter": [1000, 1500, 2000, 2500]
}

model = SGDClassifier()

model_cv = GridSearchCV(model, param_grid=params, cv=10, scoring='accuracy')

model_cv.fit(X_train, y_train)

best_params = model_cv.best_params_
best_estimator = model_cv.best_estimator_

saved_model = joblib.dump(best_estimator, "E:\\01 - FRANKFURT LECTURE SLIDE\\Machine Learning\\vehicale_baby_seat_detection\\SGD_with_feature.joblib")
# Make predictions
y_pred = best_estimator.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)

print("Prediction on test dataset: {}".format(acc_score))

print(best_params)
print(best_estimator)