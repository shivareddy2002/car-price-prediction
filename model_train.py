import pandas as pd
import numpy as np
import joblib
import pickle
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Data
print("Reading car_dataset.csv...")
df = pd.read_csv("car_dataset.csv")

# 2. Clean Data (Rename columns to match App labels)
df = df.rename(columns={
    'max_power (in bph)': 'max_power',
    'Engine (CC)': 'Engine'
})

# Drop unused columns
df.drop(columns=['Unnamed: 0', 'Mileage Unit'], inplace=True, errors='ignore')

# Fill missing values
df['max_power'] = df['max_power'].fillna(df['max_power'].median())
df['Mileage'] = df['Mileage'].fillna(df['Mileage'].median())
df['Engine'] = df['Engine'].fillna(df['Engine'].median())
df['seats'] = df['seats'].fillna(df['seats'].mode()[0])

# 3. Define Features
cat_cols = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
num_cols = ['year', 'km_driven', 'seats', 'max_power', 'Mileage', 'Engine']

# 4. Create Preprocessor (Handles One-Hot Encoding and Scaling)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X = df[num_cols + cat_cols]
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and Save Preprocessor
X_train_transformed = preprocessor.fit_transform(X_train)

# Scale target for Deep Learning
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

# 5. Train Models
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_transformed, y_train)

print("Training Neural Network...")
dl = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_transformed.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
from tensorflow.keras.losses import MeanSquaredError

dl.compile(optimizer='adam', loss=MeanSquaredError())
dl.fit(X_train_transformed, y_train_scaled, epochs=15, verbose=0)

# 6. SAVE EVERYTHING (Exact names for app.py)
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')
joblib.dump(rf, 'rf_model.pkl')
dl.save('dl_model.h5')

# Create and save dropdown options
options = {col: sorted(list(df[col].unique())) for col in cat_cols}
with open('options.pkl', 'wb') as f:
    pickle.dump(options, f)

print(f"\n✅ SUCCESS! All files saved in: {os.getcwd()}")