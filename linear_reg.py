import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv("fao_data.csv")
df.columns = df.columns.str.strip()

print("Columns detected:")
print(df.columns.tolist())


df = df[df['ELEMENT'].str.contains('Production', case=False, na=False)]

df = df[['AREA', 'CROP', 'YEAR', 'VALUE']]


df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

df = df.dropna(subset=['AREA', 'CROP', 'YEAR', 'VALUE'])

print("Usable rows after cleaning:", len(df))

if df.empty:
    raise ValueError("No usable rows after preprocessing.")

le_area = LabelEncoder()
le_crop = LabelEncoder()

df['AREA_ENCODED'] = le_area.fit_transform(df['AREA'])
df['CROP_ENCODED'] = le_crop.fit_transform(df['CROP'])


X = df[['AREA_ENCODED', 'CROP_ENCODED', 'YEAR']]
y = df['VALUE']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression MSE: {mse:.2f}")

df['PREDICTED'] = lr_model.predict(X)
df['ERROR'] = abs(df['PREDICTED'] - df['VALUE'])
df['SUSPICIOUS'] = df['ERROR'] > 0.3 * df['VALUE']

print("\nSuspicious records:")
print(df[df['SUSPICIOUS']].head())


df.to_csv("fao_output_lr_with_flags.csv", index=False)
print("Saved → fao_output_lr_with_flags.csv")


plt.figure(figsize=(10, 6))

normal = df[~df['SUSPICIOUS']]
plt.scatter(normal['YEAR'], normal['VALUE'], color='green', alpha=0.4, label='Normal')

suspicious = df[df['SUSPICIOUS']]
plt.scatter(suspicious['YEAR'], suspicious['VALUE'], color='red', alpha=0.6, label='Suspicious')

plt.plot(df['YEAR'], df['PREDICTED'], color='blue', linestyle='--', alpha=0.5, label='Predicted')

plt.xlabel("Year")
plt.ylabel("Production Value")
plt.title("FAO Crop Production Monitoring using Linear Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
