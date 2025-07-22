# main.py
"""
Universal Naive Bayes Classifier Application
"""
from NaiveBayesApp import NaiveBayesApp

def main():
    """Main function with different usage examples"""

if __name__ == "__main__":
    # Simple usage
    app = NaiveBayesApp()

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
