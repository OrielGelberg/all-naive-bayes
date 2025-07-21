# naive_bayes.py
"""
Universal Naive Bayes Classifier for any CSV data
"""
import pandas as pd
import numpy as np
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self, data, target_column, feature_columns):
        """Initialize classifier with data"""
        self.data = data.copy()
        self.target_column = target_column
        self.feature_columns = feature_columns
        
        # Probability storage
        self.class_probabilities = {}
        self.feature_probabilities = {}
        
        self._calculate_probabilities()

    def _calculate_probabilities(self):
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

    def predict(self, input_dict):
        """Predict class for given input"""
        class_scores = {}

        for class_value in self.class_probabilities:
            score = self.class_probabilities[class_value]

            for feature, value in input_dict.items():
                if feature in self.feature_columns:
                    if (feature in self.feature_probabilities[class_value] and 
                        value in self.feature_probabilities[class_value][feature]):
                        feature_prob = self.feature_probabilities[class_value][feature][value]
                    else:
                        # Handle unseen values with smoothing
                        class_data = self.data[self.data[self.target_column] == class_value]
                        class_size = len(class_data)
                        unique_count = len(self.data[feature].unique()) if feature in self.data.columns else 1
                        feature_prob = 1 / (class_size + unique_count)
                    
                    score *= feature_prob

            class_scores[class_value] = score

        # Return class with highest probability
        predicted_class = max(class_scores, key=class_scores.get)
        return predicted_class, class_scores

    def get_feature_columns(self):
        """Return feature columns"""
        return self.feature_columns