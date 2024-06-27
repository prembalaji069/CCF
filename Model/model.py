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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
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
data.iloc[:, 1:29] = StandardScaler().fit_transform(data.iloc[:, 1:29])

# Convert the data frame to its Numpy-array representation
data_matrix = data.to_numpy()
X = data_matrix[:, 1:29]
Y = data_matrix[:, -1]

# Estimate class weights since the dataset is unbalanced
class_weights = dict(zip([0, 1], compute_class_weight('balanced', classes=[0, 1], y=Y)))

# Create train/test indices to split data in train/test sets
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# Define a model generator
def generate_model():
    model = Sequential()
    model.add(Dense(22, input_dim=28))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Define callbacks
earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=0, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Storage
k = 0
predictions = np.empty([len(Y), kfold.n_splits])
for train, test in kfold.split(X, Y):
    # Define model
    model = generate_model()
    # Fit the model
    history = model.fit(X[train], Y[train],
                        batch_size=1200,
                        epochs=100,
                        verbose=0,
                        shuffle=True,
                        validation_data=(X[test], Y[test]),
                        class_weight=class_weights,
                        callbacks=[earlystop, reduce_lr, tensor_board])
    # Store the predicted probabilities and iterate k
    predictions[test, k] = model.predict(X[test]).flatten()
    k += 1

# Average the model predictions
yhat = np.nanmean(predictions, axis=1).round().astype(int)

# Performance
print(classification_report(Y, yhat))
print(pd.crosstab(Y, yhat.flatten(), rownames=['Truth'], colnames=['Predictions']))

fpr, tpr, thresholds = roc_curve(Y, yhat)
precision, recall, thresholds = precision_recall_curve(Y, yhat)
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
