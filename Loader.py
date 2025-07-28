import pandas as pd


class Loader:
    def __init__(self):
        self.data = None
        self.target_column = None
        self.feature_columns = []

    def load_csv(self, file_path, target_column=None, index_col=None):
        """Load CSV file only - no cleaning"""
        try:
            # Load the CSV file - no cleaning here
            self.data = pd.read_csv(file_path, index_col=None)

            print(f"Raw data loaded: {len(self.data)} rows with {len(self.data.columns)} columns")
            return True

        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def set_data(self, cleaned_data):
        """Set cleaned data from external source"""
        self.data = cleaned_data

    def configure_columns(self, target_column=None):
        """Configure target and feature columns after data is set"""
        if self.data is None:
            return False

        # Determine target column
        if target_column and target_column in self.data.columns:
            self.target_column = target_column
        else:
            # Use last column as target by default
            self.target_column = self.data.columns[-1]

        # Set feature columns
        self.feature_columns = [col for col in self.data.columns if col != self.target_column]

        print(f"Configured {len(self.data)} rows with {len(self.data.columns)} columns")
        print(f"Target: {self.target_column}")
        print(f"Features: {self.feature_columns}")

        return True

    def get_data(self):
        """Return the loaded data"""
        return self.data

    def get_target_column(self):
        """Return target column name"""
        return self.target_column

    def get_feature_columns(self):
        """Return feature column names"""
        return self.feature_columns
