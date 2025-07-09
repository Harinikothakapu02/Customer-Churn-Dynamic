import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
import pickle
from pathlib import Path
from collections import Counter
from sklearn.metrics import classification_report

def train_model():
    """Train and save churn prediction model with SMOTEENN"""
    try:
        # 1. Load data
        data_path = Path(__file__).parent.parent / 'data' / 'fetched_data.xlsx'
        df = pd.read_excel(data_path)
        
        # 2. Validate data
        required_statuses = {'Stayed', 'Churned'}
        available_statuses = set(df['Customer_Status'].unique())
        
        if not required_statuses.issubset(available_statuses):
            missing = required_statuses - available_statuses
            raise ValueError(f"Missing required statuses: {missing}")
        
        # 3. Prepare data
        df = df[df['Customer_Status'].isin(required_statuses)]
        df['Churn'] = df['Customer_Status'].map({'Stayed': 0, 'Churned': 1})
        
        if len(df) < 50:
            raise ValueError(f"Only {len(df)} records - need at least 50 for training")
        
        # 4. Select features
        features = [
            'Tenure_in_Months',
            'Monthly_Charge', 
            'Total_Revenue',
            # Add other features as needed
        ]
        
        X = df[features]
        y = df['Churn']
        
        # 5. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 6. Apply SMOTEENN only if multiple classes exist
        print("\nClass distribution before resampling:")
        print(Counter(y_train))
        
        if len(Counter(y_train)) > 1:
            smote_enn = SMOTEENN(random_state=42)
            X_res, y_res = smote_enn.fit_resample(X_train, y_train)
            print("\nClass distribution after resampling:")
            print(Counter(y_res))
        else:
            X_res, y_res = X_train, y_train
            print("\nOnly one class present - skipping SMOTEENN")
        
        # 7. Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_res, y_res)
        
        # 8. Evaluate model
        y_pred = model.predict(X_test)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        
        # 9. Save model
        model_dir = Path(__file__).parent.parent / 'models'
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / 'churn_model_smoteenn.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"\n✅ Model trained on {len(X_res)} samples")
        print(f"Test set churn rate: {y_test.mean():.1%}")
        print(f"Model saved to {model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    train_model()