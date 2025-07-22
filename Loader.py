# Loader.py
"""
Universal data handler for any CSV file
"""
import pandas as pd
import numpy as np


class Loader:
    def __init__(self):
        self.data = None
        self.target_column = None
        self.feature_columns = []

    def load_csv(self, file_path, target_column=None, index_col=None):
        """Load CSV file and automatically detect structure"""
        try:
            # Always reset index to avoid using index as a feature
            self.data = pd.read_csv(file_path, index_col=None)

            # Clean data
            self.data = self.data.dropna().drop_duplicates().reset_index(drop=True)

            # Remove any columns that have all unique values (ID/index columns)
            index_like_columns = []
            for col in self.data.columns:
                # If all values in the column are unique, it's likely an ID column
                if len(self.data[col].unique()) == len(self.data):
                    index_like_columns.append(col)

            # Drop index-like columns
            if index_like_columns:
                print(f"Removing index-like columns: {index_like_columns}")
                self.data = self.data.drop(columns=index_like_columns)

            # Determine target column
            if target_column and target_column in self.data.columns:
                self.target_column = target_column
            else:
                # Use last column as target by default
                self.target_column = self.data.columns[-1]

            # Set feature columns
            self.feature_columns = [col for col in self.data.columns if col != self.target_column]

            print(f"Loaded {len(self.data)} rows with {len(self.data.columns)} columns")
            print(f"Target: {self.target_column}")
            print(f"Features: {self.feature_columns}")

            return True

        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def get_data(self):
        """Return the loaded data"""
        return self.data

    def get_target_column(self):
        """Return target column name"""
        return self.target_column

    def get_feature_columns(self):
        """Return feature column names"""
        return self.feature_columns

    def get_unique_values(self, column):
        """Get unique values for a specific column"""
        if self.data is not None and column in self.data.columns:
            return list(self.data[column].unique())
        return []

