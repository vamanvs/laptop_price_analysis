# train_model.py
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 1) load data
df = pd.read_csv("laptop_data.csv")

# 2) target and simple cleanup (adapt if your notebook does additional cleaning)
# assume Price column is named 'Price' (change if the notebook uses different)
if 'Price' not in df.columns:
    # if the notebook uses 'price' lowercase, try that
    if 'price' in df.columns:
        df.rename(columns={'price':'Price'}, inplace=True)

# Drop any columns that are not features used by the notebook/app (if needed)
# For simplicity, use object dtype columns as categorical and numeric dtype as numeric
X = df.drop(columns=['Price'])
y = df['Price']

# Simple feature split
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

# Build preprocessing and pipeline (simple, matches typical campusx pipelines)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ],
    remainder='drop'
)

pipe = Pipeline([
    ('preproc', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

print("Training model on", X.shape[0], "rows and", X.shape[1], "features (num:", len(num_cols), "cat:", len(cat_cols), ")")
pipe.fit(X, y)

# Save
with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)
print("Saved pipe.pkl (trained with scikit-learn version):", pipe)
