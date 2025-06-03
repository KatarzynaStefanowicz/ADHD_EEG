import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten

# Load dataset
df = pd.read_csv(data/adhdata.csv)
print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Encode class labels
df['Class'] = LabelEncoder().fit_transform(df['Class'])  # ADHD -> 1, Control -> 0
X = df.drop(['Class', 'ID'], axis=1).values
y = df['Class'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Reshape for CNN input (samples, timesteps, features) -> (n, 1, features)
X_train_cnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_cnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# One-hot encode y
y_train_cat = tf.keras.utils.to_categorical(y_train)
y_test_cat = tf.keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=(1, X_train.shape[1])))
model.add(MaxPooling1D(pool_size=1))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train_cnn, y_train_cat, epochs=15, batch_size=32, validation_split=0.2, verbose=1)

loss, acc = model.evaluate(X_test_cnn, y_test_cat)
print(f"CNN-LSTM Test Accuracy: {acc:.4f}")

# Confusion Matrix
y_pred_dl = model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred_dl, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Control", "ADHD"], yticklabels=["Control", "ADHD"])
plt.title("CNN-LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print(classification_report(y_test, y_pred_classes))

model.save("adhd_eeg_model.h5")