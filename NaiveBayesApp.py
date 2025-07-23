import os

from sklearn.model_selection import train_test_split

from Classifier import Classifier
from Loader import Loader
from Validator import Validator
from Client_request import ServerClient


class NaiveBayesApp:
    def __init__(self):
        self.data_handler = Loader()
        self.classifier = None
        self.server_client = ServerClient()
        self.model = {}
        self.class_probabilities = {}

    def load_and_train(self, file_path, target_column=None):
        """Load data and train model"""
        success = self.data_handler.load_csv(file_path, target_column)
        if not success:
            return False

        # Train model
    def train_model(self):
        data = self.data_handler.get_data()
        target_col = self.data_handler.get_target_column()
        feature_cols = self.data_handler.get_feature_columns()

        self.classifier = Classifier(data, target_col, feature_cols)
        self.model, self.class_probabilities = self.classifier.calculate_model()
        return self.model, self.class_probabilities


        # Test accuracy
    def tester(self):
        tester = Validator(self.data_handler)
        accuracy = tester.test_accuracy()
        print(f"Model accuracy: {accuracy:.1%}")
        return accuracy


        # """Make prediction from input dictionary"""
        # if self.classifier is None:
        #     return None, "No trained model available"

        # First check for exact match in data
        # exact_match = self.classifier.check_exact_match(input_dict)
        # if exact_match is not None:
        #     return exact_match, "exact_match"

        # If no exact match, use classifier





    # def run_server_mode(self):
    #
    #         # Get input from user instead of server
    #         input_dict = self.get_user_input()
    #
    #         # Send to server and get prediction
    #         result = self.server_client.send_prediction_request(input_dict)
    #
    #         if result and 'target' in result:
    #             print(f"\nServer prediction: {result['target']} ({result.get('result', 'unknown')})")
    #             return result
    #         else:
    #             print("Server prediction failed")
    #             return None


    # def run(self, file_path=None, target_column=None, mode="local"):
    #     """Main application runner"""
    #     print("Universal Naive Bayes Classifier")
    #     print("=" * 40)
    #
    #     # Load file
    #     if not file_path:
    #         file_path = input("Enter CSV file path: ").strip()
    #
    #     if not os.path.exists(file_path):
    #         print(f"File not found: {file_path}")
    #         return
    #
    #     # Load and train
    #     success = self.load_and_train(file_path, target_column)
    #
    #     if not success:
    #         print("Failed to load and train model")
    #         return
    #
    #
    #     return self.run_server_mode()

