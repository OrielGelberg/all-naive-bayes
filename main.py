# main.py
"""
Universal Naive Bayes Classifier Application
"""
import os
from data_handler import DataHandler
from naive_bayes import NaiveBayesClassifier
from model_tester import ModelTester
from server_client import ServerClient


class UniversalNaiveBayesApp:
    def __init__(self):
        self.data_handler = DataHandler()
        self.classifier = None
        self.server_client = ServerClient()

    def load_and_train(self, file_path, target_column=None):
        """Load data and train model"""
        success = self.data_handler.load_csv(file_path, target_column)

        if not success:
            return False

        # Train model
        data = self.data_handler.get_data()
        target_col = self.data_handler.get_target_column()
        feature_cols = self.data_handler.get_feature_columns()

        self.classifier = NaiveBayesClassifier(data, target_col, feature_cols)

        # Test accuracy
        tester = ModelTester(self.data_handler)
        accuracy = tester.test_accuracy()
        print(f"Model accuracy: {accuracy:.1%}")

        return True

    def predict_from_input(self, input_dict):
        """Make prediction from input dictionary"""
        if self.classifier is None:
            return None, "No trained model available"

        # First check for exact match in data
        exact_match = self.data_handler.check_exact_match(input_dict)
        if exact_match is not None:
            return exact_match, "exact_match"

        # If no exact match, use classifier
        try:
            prediction, scores = self.classifier.predict(input_dict)
            return prediction, "predicted"
        except Exception as e:
            return None, f"Prediction error: {e}"

    def get_user_input(self):
        """Get input from user interactively with numbered options"""
        input_dict = {}
        feature_columns = self.data_handler.get_feature_columns()

        print("\nEnter values for prediction:")
        for feature in feature_columns:
            unique_values = self.data_handler.get_unique_values(feature)

            print(f"\n{feature} options:")
            for i, value in enumerate(unique_values, 1):
                print(f"{i}. {value}")

            while True:
                try:
                    choice = input(f"Enter number for {feature} (1-{len(unique_values)}): ").strip()
                    choice_num = int(choice)

                    if 1 <= choice_num <= len(unique_values):
                        selected_value = unique_values[choice_num - 1]
                        input_dict[feature] = selected_value
                        print(f"Selected: {selected_value}")
                        break
                    else:
                        print(f"Error: Please enter a number between 1 and {len(unique_values)}")

                except ValueError:
                    print("Error: Please enter a valid number")

        return input_dict

    def run_server_mode(self):
        """Run in server mode - get data from server"""
        print("Running in server mode...")

        if not self.server_client.is_server_available():
            print("Server not available. Running in local mode instead.")
            return self.run_local_mode()

        # Get data from server
        server_data = self.server_client.get_prediction_data()

        if server_data:
            print(f"Received data from server: {server_data}")

            # Make prediction
            prediction, method = self.predict_from_input(server_data)

            if prediction:
                result = {
                    "prediction": prediction,
                    "method": method,
                    "input": server_data
                }
                print(f"Prediction: {prediction} ({method})")

                # Send result back to server
                self.server_client.send_prediction_result(data=result)
                return result

        return None

    def run_local_mode(self):
        """Run in local mode - get input from user"""
        print("Running in local mode...")

        while True:
            input_dict = self.get_user_input()
            prediction, method = self.predict_from_input(input_dict)

            if prediction:
                if method == "exact_match":
                    print(f"\nExact match found: {prediction}")
                else:
                    print(f"\nPrediction: {prediction}")
            else:
                print(f"\nError: {method}")

            while True:
                continue_choice = input("\nMake another prediction? (1=Yes, 2=No): ").strip()
                if continue_choice == "1":
                    break
                elif continue_choice == "2":
                    return
                else:
                    print("Error: Please enter 1 for Yes or 2 for No")

    def run(self, file_path=None, target_column=None, mode="local"):
        """Main application runner"""
        print("Universal Naive Bayes Classifier")
        print("=" * 40)

        # Load file
        if not file_path:
            file_path = input("Enter CSV file path: ").strip()

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        # Load and train
        success = self.load_and_train(file_path, target_column)
        if not success:
            print("Failed to load and train model")
            return

        # Run based on mode
        if mode == "server":
            return self.run_server_mode()
        else:
            self.run_local_mode()


def main():
    """Main function with different usage examples"""
    app = UniversalNaiveBayesApp()

    # Example 1: Computer purchase data
    print("Example 1: Computer Purchase Data")
    app.run("Data.csv", target_column="Buy_Computer", mode="local")

    # Example 2: Titanic survival data
    # app.run("CSV_titanic.csv", target_column="Survived", mode="local")

    # Example 3: Server mode
    # app.run("Data.csv", target_column="Buy_Computer", mode="server")


if __name__ == "__main__":
    # Simple usage
    app = UniversalNaiveBayesApp()

    # Load any CSV file
    csv_file = input("Enter CSV file path (or press Enter for 'Data.csv'): ").strip()
    if not csv_file:
        csv_file = "Data.csv"

    target_col = input("Enter target column name (or press Enter for auto-detect): ").strip()
    if not target_col:
        target_col = None

    print("\nMode options:")
    print("1. Local mode")
    print("2. Server mode")

    while True:
        mode_choice = input("Enter mode choice (1 or 2): ").strip()
        if mode_choice == "1":
            mode = "local"
            break
        elif mode_choice == "2":
            mode = "server"
            break
        else:
            print("Error: Please enter 1 for Local or 2 for Server")

    app.run(csv_file, target_col, mode)
