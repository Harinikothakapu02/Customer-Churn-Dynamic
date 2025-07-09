import pandas as pd
import pickle
from pathlib import Path

def make_predictions():
    """Load model and make predictions with detailed debug"""
    try:
        # 1. Define paths (adjust if running script from different location)
        data_path = Path(__file__).parent.parent / 'data' / 'fetched_data.xlsx'
        model_path = Path(__file__).parent.parent / 'models' / 'churn_model_smoteenn.pkl'
        output_path = Path(__file__).parent.parent / 'data' / 'predictions.xlsx'

        print(f"Loading data from: {data_path.resolve()}")
        print(f"Loading model from: {model_path.resolve()}")

        # 2. Check files exist
        if not data_path.exists():
            raise FileNotFoundError(f"Data file missing at {data_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing at {model_path}")

        # 3. Load data
        df = pd.read_excel(data_path)
        print(f"Data loaded, rows: {len(df)}, columns: {df.columns.tolist()}")

        # 4. Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")

        # 5. Define features - MUST match training features exactly
        features = [
            'Tenure_in_Months',
            'Monthly_Charge',
            'Total_Revenue',
        ]

        # 6. Verify features are present in data
        missing_features = [feat for feat in features if feat not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        print("All features found in data")

        # 7. Handle missing values in features
        missing_before = df[features].isnull().sum()
        print("Missing values before filling:")
        print(missing_before)

        df[features] = df[features].fillna(df[features].median())

        missing_after = df[features].isnull().sum()
        print("Missing values after filling:")
        print(missing_after)

        # 8. Make predictions
        print("Making predictions...")
        churn_prob = model.predict_proba(df[features])[:, 1]
        churn_pred = model.predict(df[features])

        # 9. Add prediction columns
        df['Churn_Probability'] = churn_prob
        df['Churn_Prediction'] = churn_pred
        df['churn_status_predicted'] = df['Churn_Prediction'].map({0: 'Stayed', 1: 'Churned'})

        # 10. Save to Excel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
        print("Prediction counts:")
        print(df['churn_status_predicted'].value_counts())
        print(f"Rows saved: {len(df)}")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    make_predictions()
