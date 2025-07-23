# main.py
"""
Universal Naive Bayes Classifier Application
"""
from NaiveBayesApp import NaiveBayesApp

def main():
    """Main function with different usage examples"""

if __name__ == "__main__":

    app = NaiveBayesApp()

    csv_file = "Data.csv"

    target_col = None


    app.run(csv_file, target_col, "server")
