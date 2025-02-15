import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict

def load_data(filepath: str) -> pd.DataFrame:
    """Loads dataset from a given file path."""
    return pd.read_csv(filepath)

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plots a heatmap of the correlation matrix."""
    plt.figure(figsize=(18, 15))
    sns.heatmap(df.corr(), annot=True, vmin=-1.0, cmap='mako')
    plt.title("Correlation Heatmap")
    plt.show()

def onehot_encode(df: pd.DataFrame, column_dict: Dict[str, str]) -> pd.DataFrame:
    """Applies one-hot encoding to specified columns."""
    df = df.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[column], inplace=True)
    return df

def preprocess_inputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocesses the input data by dropping unnecessary columns, encoding categorical data, and scaling features."""
    df = df.copy()
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
    
    df = onehot_encode(df, {'EDUCATION': 'EDU', 'MARRIAGE': 'MAR'})
    
    y = df['default.payment.next.month'].copy()
    X = df.drop(columns=['default.payment.next.month']).copy()
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for future use
    
    return X_scaled, y

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    """Trains multiple models and saves them as .pkl files."""
    models = {
        "LogisticRegression": LogisticRegression(),
        "SVC": SVC(),
        "MLPClassifier": MLPClassifier(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'{name}.pkl')
    
    return models

def evaluate_models(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluates trained models on the test set."""
    for name, model in models.items():
        accuracy = model.score(X_test, y_test) * 100
        print(f"{name}: {accuracy:.2f}%")

def main() -> None:
    """Main function to execute the machine learning pipeline."""
    df = load_data('UCI_Credit_Card.csv')
    #plot_correlation_heatmap(df)
    X, y = preprocess_inputs(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()