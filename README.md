# Driver-Analysis
# 1) Load the Excel file, detect header row (header=1 in this file), and preview data:
import pandas as pd
df = pd.read_excel("C:/Users/ogwur/Downloads/Driver Model Data Set (1).xlsx", header=1)

# 2) Detect target (heuristic: column with name containing 'likeli' / 'buy' etc.)
target = "Likelihood_to_Buy"  # automatically detected in the run

# 3) Separate Likert (1-5 numeric) from demographics (non-likert)
likert_cols = [c for c in df.select_dtypes(include='number').columns
               if df[c].dropna().min() >= 1 and df[c].dropna().max() <= 5 and df[c].nunique() <= 7]
demographic_cols = [c for c in df.columns if c not in likert_cols + [target]]

# 4) Build features and target, simple cleaning:
X = df[likert_cols + demographic_cols].copy()
y = pd.to_numeric(df[target], errors='coerce')
mask = y.notna()
X = X[mask]
y = y[mask]

# 5) Preprocessing: scale numeric and one-hot encode categoricals
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
categorical_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features),
                                  ('cat', categorical_transformer, categorical_features)])
X_processed = preprocessor.fit_transform(X)

# Get feature names:
num_names = numeric_features
cat_names = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)) if categorical_features else []
feature_names = num_names + cat_names

# 6) Train-test split and models:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X_processed, columns=feature_names), y, test_size=0.25, random_state=42)

lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

# 7) Evaluate:
from sklearn.metrics import r2_score, mean_squared_error
y_pred_lasso = lasso.predict(X_test)
y_pred_rf = rf.predict(X_test)
metrics = {
  'Lasso_R2': r2_score(y_test, y_pred_lasso),
  'Lasso_RMSE': mean_squared_error(y_test, y_pred_lasso, squared=False),
  'RF_R2': r2_score(y_test, y_pred_rf),
  'RF_RMSE': mean_squared_error(y_test, y_pred_rf, squared=False)
}

# 8) Extract top drivers:
coef_df = pd.DataFrame({'feature': feature_names, 'lasso_coef': lasso.coef_})
from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)
perm_df = pd.DataFrame({'feature': feature_names, 'rf_perm_importance': perm.importances_mean})
importance_df = coef_df.merge(perm_df, on='feature')
importance_df['abs_lasso'] = importance_df['lasso_coef'].abs()
importance_df = importance_df.sort_values(['abs_lasso','rf_perm_importance'], ascending=False)
top_drivers = importance_df.head(12)



 


