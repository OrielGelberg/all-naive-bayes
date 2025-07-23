

class PredictNaiveBayes:


    @staticmethod
    def predict(input_dict, model, class_probabilities):
        """Predict class for given input"""
        class_scores = {}
        feature_prob = 0
        for class_value in class_probabilities:
            score = class_probabilities[class_value]

            for feature, value in input_dict.items():
                # if feature in self.feature_columns:
                if (feature in model[class_value] and
                        value in model[class_value][feature]):
                    feature_prob = model[class_value][feature][value]
                # else:
                #     # Handle unseen values with smoothing
                #     class_data = self.data[self.data[self.target_column] == class_value]
                #     class_size = len(class_data)
                #     unique_count = len(self.data[feature].unique()) if feature in self.data.columns else 1
                #     feature_prob = 1 / (class_size + unique_count)

                score *= feature_prob

            class_scores[class_value] = score

        # Return class with highest probability
        predicted_class = max(class_scores, key=class_scores.get)
        return "buy computer", predicted_class