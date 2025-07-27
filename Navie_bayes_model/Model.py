# Model.py
"""
Universal Naive Bayes Classifier for any CSV data
"""
from collections import defaultdict


class Classifier:
    def __init__(self, data, target_column, feature_columns):
        """Initialize classifier with data"""
        self.data = data.copy()
        self.target_column = target_column
        self.feature_columns = feature_columns
        
        # Probability storage
        self.class_probabilities = {}
        self.feature_probabilities = {}
        
        # self._calculate_model()

    def calculate_model(self):
        """Calculate all required probabilities"""
        # Get unique classes
        unique_classes = self.data[self.target_column].unique()
        total_rows = len(self.data)

        # Calculate P(class)
        for class_value in unique_classes:
            class_count = len(self.data[self.data[self.target_column] == class_value])
            self.class_probabilities[class_value] = class_count / total_rows

        # Calculate P(feature|class)
        self.feature_probabilities = {}
        
        for class_value in unique_classes:
            self.feature_probabilities[class_value] = {}
            class_data = self.data[self.data[self.target_column] == class_value]
            class_size = len(class_data)

            for feature in self.feature_columns:
                self.feature_probabilities[class_value][feature] = {}
                all_feature_values = self.data[feature].unique()

                for feature_value in all_feature_values:
                    count = len(class_data[class_data[feature] == feature_value])
                    # Laplace smoothing
                    probability = (count + 1) / (class_size + len(all_feature_values))
                    self.feature_probabilities[class_value][feature][feature_value] = probability

        return self.feature_probabilities, self.class_probabilities


    def get_feature_columns(self):
        """Return feature columns"""
        return self.feature_columns



    # def check_exact_match(self, input_dict):
    #     """Check if input exactly matches any row in the data"""
    #     if self.data is None:
    #         return None
    #
    #     # Create a mask for all conditions
    #     mask = pd.Series([True] * len(self.data))
    #
    #     for feature, value in input_dict.items():
    #         if feature in self.feature_columns:
    #             mask = mask & (self.data[feature] == value)
    #
    #     # Find matching rows
    #     matches = self.data[mask]
    #
    #     if len(matches) > 0:
    #         # Return the target value from the first match
    #         return matches.iloc[0][self.target_column]
    #
    #     return None