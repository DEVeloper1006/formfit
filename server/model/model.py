import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing function
def preprocess(data_path, num_classes=4, save_scaler_path=None):
    data = pd.read_csv(data_path)
    Y = data['exercise']
    X = data.drop('exercise', axis=1)
    
    # Standardize features
    scaler = StandardScaler().fit(X)
    if save_scaler_path:
        joblib.dump(scaler, save_scaler_path)  # Save the scaler
    X = scaler.transform(X)
    
    # One-hot encode labels
    Y = tf.keras.utils.to_categorical(Y, num_classes=num_classes)
    return X, Y

# Preprocess and save scalers
X_train, Y_train = preprocess("training_data.csv", save_scaler_path="scaler_train.joblib")
X_val, Y_val = preprocess("validation_data.csv", save_scaler_path="scaler_val.joblib")
X_test, Y_test = preprocess("testing_data.csv", save_scaler_path="scaler_test.joblib")

# Reshape inputs for LSTM: (samples, timesteps, features)
X_train = X_train.reshape((-1, 50, X_train.shape[1] // 50))  # Assuming 50 frames per sample
X_val = X_val.reshape((-1, 50, X_val.shape[1] // 50))
X_test = X_test.reshape((-1, 50, X_test.shape[1] // 50))

# Print data shapes
print("Data Shapes:")
print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"X_val: {X_val.shape}, Y_val: {Y_val.shape}")
print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")

# Define the model
model = Sequential()
model.add(Bidirectional(LSTM(units=91, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(units=91, return_sequences=True)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(units=91, return_sequences=False)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(units=4))  # Assuming 4 exercise classes
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Define callbacks
checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Save training plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.savefig("accuracy_plot.png")  # Save the accuracy plot

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.savefig("loss_plot.png")  # Save the loss plot
plt.close()

# Load the best model
model.load_weights("best_model.keras")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=["Exercise 0", "Exercise 1", "Exercise 2", "Exercise 3"]))
