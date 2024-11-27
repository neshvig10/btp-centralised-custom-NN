from scripts.preprocess import load_and_preprocess_data
from scripts.model import build_and_train_model
from scripts.evaluate import evaluate_model
from sklearn.model_selection import train_test_split  # Add this import


def main():
    # File path to your dataset
    file_path = './data/UNSW_2018_IoT_Botnet_Dataset_74.csv'

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)

    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Build and train the model
    model, history = build_and_train_model(X_train, y_train, X_val, y_val)

    # Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    main()
