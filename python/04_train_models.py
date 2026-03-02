# PHASE 4 — Model Training
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_PATH = '../data/segments.csv'
OUTPUT_MODEL = '../outputs/energy_model.pkl'
OUTPUT_METRICS = '../outputs/metrics_summary.csv'

def train():
    df = pd.read_csv(INPUT_PATH)
    X = df[['Avg_Speed', 'Avg_Gradient', 'Spray_Ratio', 'Area_acres']]
    y = df['Total_Energy_Wh']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression':  Ridge(alpha=1.0),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    results = []
    best_model, best_r2 = None, -999

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2   = r2_score(y_test, preds)
        print(f"\n{name}\n  MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.4f}")
        results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    joblib.dump(best_model, OUTPUT_MODEL)
    pd.DataFrame(results).to_csv(OUTPUT_METRICS, index=False)
    print(f"\nBest model saved → {OUTPUT_MODEL}")

if __name__ == '__main__':
    train()
