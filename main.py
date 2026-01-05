import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv("fao_data.csv")
df.columns = df.columns.str.strip()

print("Detected columns:")
print(df.columns.tolist())


df = df[df['ELEMENT'].str.contains('Production', case=False, na=False)]


df = df[['AREA', 'YEAR', 'VALUE']]

df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
df = df.dropna(subset=['AREA', 'YEAR', 'VALUE'])

print("Usable rows:", len(df))
if df.empty:
    raise ValueError("No usable rows after preprocessing.")


le_area = LabelEncoder()
df['AREA_ENCODED'] = le_area.fit_transform(df['AREA'])


X = df[['AREA_ENCODED', 'YEAR']]
y = df['VALUE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.1, random_state=42
    ),
    "Linear Regression": LinearRegression()
}


for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"📊 {name} MSE: {mse:.2f}")

    # Create model-specific copy
    result_df = df.copy()
    result_df['PREDICTED'] = model.predict(X)
    result_df['ERROR'] = abs(result_df['PREDICTED'] - result_df['VALUE'])
    result_df['SUSPICIOUS'] = result_df['ERROR'] > 0.3 * result_df['VALUE']

    print(f"⚠ Suspicious records ({name}):")
    print(result_df[result_df['SUSPICIOUS']].head())

    # Save output
    filename = f"fao_output_{name.replace(' ', '_').lower()}.csv"
    result_df.to_csv(filename, index=False)
    print(f"✅ Saved → {filename}")

 
    plt.figure(figsize=(10, 6))

    normal = result_df[~result_df['SUSPICIOUS']]
    suspicious = result_df[result_df['SUSPICIOUS']]

    plt.scatter(normal['YEAR'], normal['VALUE'], alpha=0.4, label='Normal')
    plt.scatter(suspicious['YEAR'], suspicious['VALUE'], alpha=0.7, label='Suspicious')
    plt.plot(result_df['YEAR'], result_df['PREDICTED'],
             linestyle='--', alpha=0.6, label='Predicted')

    plt.title(f"{name}: FAO Crop Production Monitoring")
    plt.xlabel("Year")
    plt.ylabel("Production Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
