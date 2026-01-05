import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="FAO Crop Production Monitoring", layout="wide")
st.title("🌾 FAO Crop Production Monitoring Dashboard")
st.markdown("Detecting anomalous crop production values using Machine Learning")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fao_data.csv")
    df.columns = df.columns.str.strip()
    df = df[df['ELEMENT'].str.contains('Production', case=False, na=False)]
    df = df[['AREA', 'YEAR', 'VALUE']]
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    df = df.dropna()
    return df

df = load_data()

# -----------------------------
# Encode AREA
# -----------------------------
le = LabelEncoder()
df['AREA_ENCODED'] = le.fit_transform(df['AREA'])

X = df[['AREA_ENCODED', 'YEAR']]
y = df['VALUE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙ Model Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Gradient Boosting", "Linear Regression"]
)

threshold = st.sidebar.slider(
    "Suspicious Threshold (%)",
    min_value=10,
    max_value=50,
    value=30
)

# -----------------------------
# Model Selection
# -----------------------------
if model_name == "Random Forest":
    model = RandomForestRegressor(n_estimators=200, random_state=42)
elif model_name == "Gradient Boosting":
    model = GradientBoostingRegressor(n_estimators=150, random_state=42)
else:
    model = LinearRegression()

# -----------------------------
# Train & Predict
# -----------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

st.metric(label="📉 Mean Squared Error", value=f"{mse:.2f}")

df['PREDICTED'] = model.predict(X)
df['ERROR'] = abs(df['PREDICTED'] - df['VALUE'])
df['SUSPICIOUS'] = df['ERROR'] > (threshold / 100) * df['VALUE']

# -----------------------------
# Display Suspicious Records
# -----------------------------
st.subheader("⚠ Suspicious Production Records")
st.dataframe(
    df[df['SUSPICIOUS']][['AREA', 'YEAR', 'VALUE', 'PREDICTED', 'ERROR']],
    use_container_width=True
)

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📊 Production Trend & Anomaly Visualization")

fig, ax = plt.subplots(figsize=(10, 5))

normal = df[~df['SUSPICIOUS']]
suspicious = df[df['SUSPICIOUS']]

ax.scatter(normal['YEAR'], normal['VALUE'], alpha=0.4, label='Normal')
ax.scatter(suspicious['YEAR'], suspicious['VALUE'], alpha=0.7, label='Suspicious')
ax.plot(df['YEAR'], df['PREDICTED'], linestyle='--', label='Predicted')

ax.set_xlabel("Year")
ax.set_ylabel("Production Value")
ax.set_title(f"{model_name} – Crop Production Monitoring")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🎓 Academic Project | FAO Data | Machine Learning for Agriculture Monitoring")
