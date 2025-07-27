# main.py
"""
Universal Naive Bayes Classifier Application
"""
from Model import Classifier
from Loader import Loader
from Validator import Validator


# def main():
#     """Main function with different usage examples"""

# if __name__ == "__main__":

class NaiveBayesApp:
    def __init__(self):
        self.data_handler = Loader()
        self.classifier = None
        self.model = {}
        self.class_probabilities = {}

    def load_csv(self, file_path, target_column=None):
        """Load data"""
        success = self.data_handler.clean_csv(file_path, target_column)
        if not success:
            return False

        # Train model
    def train_model(self):
        data = self.data_handler.get_data()
        target_col = self.data_handler.get_target_column()
        feature_cols = self.data_handler.get_feature_columns()

        self.classifier = Classifier(data, target_col, feature_cols)
        self.model, self.class_probabilities = self.classifier.calculate_model()
        return self.model, self.class_probabilities , target_col


        # Test accuracy
    def tester(self):
        tester = Validator(self.data_handler)
        accuracy = tester.test_accuracy()
        print(f"Model accuracy: {accuracy:.1%}")
        return accuracy
