# Validator.py
"""
Test the accuracy of the Naive Bayes model
"""
from sklearn.model_selection import train_test_split
from Classifier import Classifier


class Validator:
    def __init__(self, data_handler):
        """Initialize with data handler"""
        self.data_handler = data_handler

    def test_accuracy(self, test_size=0.3, random_state=42):
        """Test model accuracy using train-test split"""
        data = self.data_handler.get_data()
        target_column = self.data_handler.get_target_column()
        feature_columns = self.data_handler.get_feature_columns()

        if data is None or len(data) < 5:
            return 0.0

        # Split data
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )

        # Train model on training data
        classifier = Classifier(train_data, target_column, feature_columns)

        # Test on test data
        correct_predictions = 0
        total_predictions = len(test_data)

        for _, row in test_data.iterrows():
            # Prepare input
            input_values = {}
            for col in feature_columns:
                input_values[col] = row[col]

            # Get prediction
            predicted_class, _ = classifier.predict(input_values)
            actual_class = row[target_column]

            if predicted_class == actual_class:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return accuracy