import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import os

# Reproducible random seed
seed = 1
np.random.seed(seed)

# Create the output directories, if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("Visual", exist_ok=True)

# Import and normalize the data
data = pd.read_csv('/content/creditcard.csv')

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
data.iloc[:, 1:29] = scaler.fit_transform(data.iloc[:, 1:29])

# Convert the data frame to its Numpy-array representation
data_matrix = data.to_numpy()
X = data_matrix[:, 1:29]
Y = data_matrix[:, -1]

# Estimate class weights since the dataset is unbalanced
class_weights = dict(zip([0, 1], compute_class_weight('balanced', classes=[0, 1], y=Y)))

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed, stratify=Y)

# Use SMOTE to balance the training set
smote = SMOTE(random_state=seed)
X_train_sm, Y_train_sm = smote.fit_resample(X_train, Y_train)

# Define a model generator for neural network
def generate_nn_model():
    model = Sequential()
    model.add(Dense(22, input_dim=28))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Hyperparameter tuning for Logistic Regression and Random Forest
logreg = LogisticRegression(max_iter=1000)
logreg_params = {'C': [0.01, 0.1, 1, 10, 100]}
logreg_grid = GridSearchCV(logreg, logreg_params, scoring='accuracy', cv=5)
logreg_grid.fit(X_train_sm, Y_train_sm)

rf = RandomForestClassifier(random_state=seed)
rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(rf, rf_params, scoring='accuracy', cv=5)
rf_grid.fit(X_train_sm, Y_train_sm)

# Evaluate the best models
print("Best Logistic Regression parameters: ", logreg_grid.best_params_)
print("Best Random Forest parameters: ", rf_grid.best_params_)

# Define callbacks for neural network
earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=0, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Train neural network with best parameters
nn_model = generate_nn_model()
history = nn_model.fit(X_train_sm, Y_train_sm,
                        batch_size=1200,
                        epochs=100,
                        verbose=1,
                        validation_data=(X_test, Y_test),
                        class_weight=class_weights,
                        callbacks=[earlystop, reduce_lr, tensor_board])

# Evaluate all models on test set
logreg_yhat = logreg_grid.predict(X_test)
rf_yhat = rf_grid.predict(X_test)
nn_yhat = nn_model.predict(X_test).round().astype(int)

# Performance for Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(Y_test, logreg_yhat))
print(pd.crosstab(Y_test, logreg_yhat, rownames=['Truth'], colnames=['Predictions']))

# Performance for Random Forest
print("Random Forest Classification Report:")
print(classification_report(Y_test, rf_yhat))
print(pd.crosstab(Y_test, rf_yhat, rownames=['Truth'], colnames=['Predictions']))

# Performance for Neural Network
print("Neural Network Classification Report:")
print(classification_report(Y_test, nn_yhat))
print(pd.crosstab(Y_test, nn_yhat.flatten(), rownames=['Truth'], colnames=['Predictions']))

# Plot ROC and Precision-Recall curves for NN model
fpr, tpr, _ = roc_curve(Y_test, nn_yhat)
precision, recall, _ = precision_recall_curve(Y_test, nn_yhat)
roc_auc = auc(fpr, tpr)

sns.set_style("whitegrid")
plt.figure(figsize=(8,5))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.,1.])
plt.ylim([0.,1.])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("Visual/ROC.png")

plt.clf()
plt.title('Precision Recall Curve')
plt.plot(recall, precision, 'b')
plt.xlim([0.,1.])
plt.ylim([0.,1.])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.savefig("Visual/precision-recall.png")
