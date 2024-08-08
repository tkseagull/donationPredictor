import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('donations.csv', thousands=',')

# Convert the date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values('Date', inplace=True)

# Extract features and target
donations = data['Donations'].values

# Normalize the donations data
scaler = MinMaxScaler(feature_range=(0, 1))
donations = scaler.fit_transform(donations.reshape(-1, 1))

# Prepare the data for time series forecasting
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 12  # e.g., using the last 12 months to predict the next one
X, y = create_dataset(donations, time_step)

# Reshape the input to be [samples, time steps, features] which is required for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Build the LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(tf.keras.layers.LSTM(50, return_sequences=False))
model.add(tf.keras.layers.Dense(25))
model.add(tf.keras.layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=20)



# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual values
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Get the last 'time_step' data points to predict the next month's donation
last_month_data = donations[-time_step:]
last_month_data = last_month_data.reshape(1, -1, 1)

# Predict the next month's donation
next_month_prediction = model.predict(last_month_data)
next_month_prediction = scaler.inverse_transform(next_month_prediction)

print(f"Predicted donation for next month: ${next_month_prediction[0][0]:.2f}")




# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f'Train Loss: {train_loss:.4f}')
print(f'Test Loss: {test_loss:.4f}')


import matplotlib.pyplot as plt

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(donations), label='Actual Donations')
plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict, label='Train Predictions')
plt.plot(np.arange(len(train_predict) + 2 * time_step, len(train_predict) + 2 * time_step + len(test_predict)), test_predict, label='Test Predictions')
plt.legend()
plt.show()
