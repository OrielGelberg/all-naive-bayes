from Model import Classifier
from Client_request import ServerClient
from Loader import Loader
from Data_cleaner import DataCleaner
from Validator import Validator


class NaiveBayesApp:
    def __init__(self):
        self.data_handler = Loader()
        self.data_cleaner = DataCleaner()
        self.classifier = None
        self.server_client = ServerClient()
        self.model = {}
        self.class_probabilities = {}

    def load_csv(self, file_path, target_column=None):
        """Load and clean data using separate components"""

        # Step 1: Load raw CSV data
        print("Step 1: Loading CSV file...")
        load_success = self.data_handler.load_csv(file_path)
        if not load_success:
            print("Failed to load CSV file")
            return False

        # Step 2: Get raw data and clean it
        print("\nStep 2: Cleaning data...")
        raw_data = self.data_handler.get_data()
        cleaned_data = self.data_cleaner.full_clean(raw_data, verbose=True)

        # Step 3: Set cleaned data back to loader and configure columns
        print("\nStep 3: Configuring data structure...")
        self.data_handler.set_data(cleaned_data)
        config_success = self.data_handler.configure_columns(target_column)

        if not config_success:
            print("Failed to configure data columns")
            return False

        print("Data loading and cleaning completed successfully!")
        return True

    def train_model(self):
        """Train the Naive Bayes model"""
        print("\nTraining model...")
        data = self.data_handler.get_data()
        target_col = self.data_handler.get_target_column()
        feature_cols = self.data_handler.get_feature_columns()

        self.classifier = Classifier(data, target_col, feature_cols)
        self.model, self.class_probabilities = self.classifier.calculate_model()
        print("Model training completed!")
        return self.model, self.class_probabilities, target_col

    def tester(self):
        """Test model accuracy"""
        print("\nTesting model accuracy...")
        tester = Validator(self.data_handler)
        accuracy = tester.test_accuracy()
        print(f"Model accuracy: {accuracy:.1%}")
        return accuracy

    def get_data_info(self):
        """Get information about loaded data"""
        if self.data_handler.get_data() is not None:
            data = self.data_handler.get_data()
            return {
                'rows': len(data),
                'columns': len(data.columns),
                'target': self.data_handler.get_target_column(),
                'features': self.data_handler.get_feature_columns()
            }
        return None