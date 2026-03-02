# PHASE 5 — Validation & Plots
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_PATH = '../data/segments.csv'
MODEL_PATH = '../outputs/energy_model.pkl'
PLOT_PATH  = '../outputs/validation_plot.png'

def validate():
    df = pd.read_csv(INPUT_PATH)
    X = df[['Avg_Speed', 'Avg_Gradient', 'Spray_Ratio', 'Area_acres']]
    y = df['Total_Energy_Wh']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    print(f"MAE:  {mean_absolute_error(y_test, preds):.3f} Wh")
    print(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.3f} Wh")
    print(f"R2:   {r2_score(y_test, preds):.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(preds, label='Predicted', linewidth=2, linestyle='--')
    plt.title('Energy Prediction vs Actual (Mission 10)')
    plt.xlabel('Segment')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Plot saved → {PLOT_PATH}")

if __name__ == '__main__':
    validate()
