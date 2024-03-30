# Baby seat detection in vahicles.

## Introduction
This project is to determine weither a vahicle has baby seat installed and baby is on it or not.
We have gathered 32,000 ultra-sound sensor signal to predict the outcome. Dataset has 16384 timesteps in each signal. Also added
distance of the baby and seat as `measured_distance` column, one meta-data value which differ in each case in column `metadata_value`
and a `label` column where values are 1 when installed and baby on seat properly, 0 otherwise.

### Setting up environment.
- Install python 3.9.18
- Create virtual environment. `python -m venv PATH-AND-NAME-OF-ENVIRONMENT`
- Install packages from requirements.txt. `pip install -r PATH-AND-NAME-OF-REQUIREMENTS.TXT`

### Train model
- Activate virtual environment. `PATH-AND-NAME-OF-ENVIRONMENT\scripts\activate`
- Edit preferred PREFERRED_model.py file for dataset filepath. `file ="dataset.csv file path here"`
- Run preferred model.py file. In terminal when virtual environment is active, `python PREFERRED_model.py`
- We can see the output predictions in the terminal.

### Signal processing

We collected our signals as analogue to digital convert in a text file. From that file first 16 input were meta-data. Among those Data
number 10 was different in each case. So we wanted to keep that value. To work with ML program we wanted to convert those data 
into .csv file. We did not took .xlsx because it has only 16384 column available. And our signal was already 16384. We had to add
distance, metadata_value, labels. So first we converted each adc_.txt file into .csv and added those three values to end. Also signal
timesteps were in string data type. So we converted that to float.
```
for file_dir in file_path:  
    row = []
    single_sample_signal =[]
    
    file = file_dir+"\\adc_.txt"                                 
    
    with open(file, 'r') as openedfile:                         
        for line in openedfile:
            csline = line.replace('\t', ',')
            csline = csline.rstrip('\n')
            row_data = [float(num) for num in csline.split(',')]
            single_sample_signal.append(row_data[10:])

    
    create_csv_file(file_dir, single_sample_signal)
```
Then we applied FFT on the signals. 
```
for file_dir in file_path:  
    fft_signal_real =[]
    
    raw_signal = pd.read_csv(file_dir+"\\signal.csv")              
    
    for index, single_signal in raw_signal.iterrows():
        fft_signal = np.fft.fft(single_signal[6:])
        fft_signal_real.append(np.real(fft_signal).tolist())
```
After converting all the files and adding values we added all 
the signal together.
```
def create_csv_file(file_path, row_data):
    
    #signal = file_path + "\\signal.csv"
    signal = file_path + "\\fft_signal.csv"
    
    with open(signal, 'w', newline='') as signalfile:
        writer = csv.writer(signalfile)
        writer.writerows(row_data)
```

### Feature extraction
When we tried training full signal without feature extraction, it was too heavy. So we tried feature extraction. We extracted 
time-domain and frequency-domain feature extraction. We used `scipy` to calculate mean, median, standard-deviation, skewness,
kurtosis, percentails, entropy. It showed remarkable update in energy consumption and hardware consumption.

### ML algorithms
We used CNN and SGD algorithms to train and predict. Details about CNN and SGD here.
###### CNN

###### SGD


### Model training
##### CNN
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=[11, 1]),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
Model is trained with both raw_signal and features. For CNN we used 2 layers of Convulation layers. Also used `maxpooling1d`.
We had two output nodes for two classes. for both models data was scalled using `StandardScaling` from `Sickit-learn`. For each conv
layers we used 64 filters. We used 300 epochs and batch-size 3000. We got accuracy of 0.9979671835899353 and the loss was 
0.010772799141705036. We got this accuracy with feature dataset.
`###########PIC_1 GOES HERE###############`
####### Confusion-Matrix
As we can see in confusion matrix The output prediction and real value have mostly accurate.
`################PIC_2 GOES HERE###################`
##### SGD
```
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
saved_model = joblib.dump(best_estimator, "FILE_LOCATION")
# Make predictions
y_pred = best_estimator.predict(X_test)
```
For stochastic gradient discent we also used feature dataset. We used `StandardScaling` from `Sickit-learn.preprocessing` for scalling
For class balancing we class weikght value `balanced`. We used grid search cross-validation for getting the optimal hyper-parametre.
We used 10 fold cross-validation. After running the model we get best_estimator as - "alpha=0.01, class_weight='balanced',
 max_iter=2500, penalty=None". With this parametre setting we get accuracy of `0.8104769351055512`. best_params were- 
 ```{'alpha': 0.01, 'class_weight': 'balanced', 'loss': 'hinge', 'max_iter': 2500, 'penalty': None}```
 
```###############PIC_3 GOES HERE#################```

