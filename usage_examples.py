# usage_examples.py
"""
Examples of how to use the Universal Naive Bayes Classifier
"""
from main import UniversalNaiveBayesApp


def example_computer_data():
    """Example using computer purchase data"""
    print("=== Computer Purchase Prediction ===")
    
    app = UniversalNaiveBayesApp()
    
    # Load and train on computer data
    success = app.load_and_train("Data.csv", target_column="Buy_Computer")
    
    if success:
        # Test prediction with exact match (should find in data)
        exact_input = {
            "age": "youth",
            "income": "high", 
            "student": "no",
            "credit_rating": "excellent"
        }
        
        result, method = app.predict_from_input(exact_input)
        print(f"Input: {exact_input}")
        print(f"Result: {result} ({method})")
        
        # Test prediction that needs classification
        predict_input = {
            "age": "middle_age",
            "income": "low",
            "student": "no", 
            "credit_rating": "excellent"
        }
        
        result, method = app.predict_from_input(predict_input)
        print(f"\nInput: {predict_input}")
        print(f"Result: {result} ({method})")


def example_titanic_data():
    """Example using Titanic survival data"""
    print("\n=== Titanic Survival Prediction ===")
    
    app = UniversalNaiveBayesApp()
    
    # Load and train on Titanic data
    success = app.load_and_train("CSV_titanic.csv", target_column="Survived")
    
    if success:
        # Test prediction
        test_input = {
            "Pclass": "1",
            "Sex": "female",
            "Age": "20-39",
            "SibSp": "0",
            "Parch": "0", 
            "Fare": "Average",
            "Embarked": "Southampton"
        }
        
        result, method = app.predict_from_input(test_input)
        print(f"Input: {test_input}")
        print(f"Survival prediction: {result} ({method})")


def example_server_integration():
    """Example of server integration"""
    print("\n=== Server Integration Example ===")
    
    app = UniversalNaiveBayesApp()
    
    # Load model
    success = app.load_and_train("Data.csv", target_column="Buy_Computer")
    
    if success:
        # Try server mode
        result = app.run_server_mode()
        
        if result:
            print(f"Server prediction completed: {result}")
        else:
            print("Server mode failed, falling back to local")


def example_any_csv():
    """Example showing it works with any CSV"""
    print("\n=== Universal CSV Example ===")
    
    # This function demonstrates that the classifier can work with any CSV
    # Just change the file path and target column
    
    csv_files = [
        ("Data.csv", "Buy_Computer"),
        ("CSV_titanic.csv", "Survived")
    ]
    
    for csv_file, target in csv_files:
        print(f"\nProcessing {csv_file}...")
        
        app = UniversalNaiveBayesApp()
        success = app.load_and_train(csv_file, target)
        
        if success:
            print(f"Successfully trained model for {csv_file}")
            print(f"Features: {app.data_handler.get_feature_columns()}")
            print(f"Target: {app.data_handler.get_target_column()}")


if __name__ == "__main__":
    # Run examples
    example_computer_data()
    example_titanic_data()
    example_any_csv()
    
    # Uncomment to test server integration
    # example_server_integration()