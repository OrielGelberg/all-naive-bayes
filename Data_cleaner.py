# DataCleaner.py
"""
Data cleaning utilities following Single Responsibility Principle
"""
import pandas as pd


class DataCleaner:


    @staticmethod
    def clean_dataframe(df):

        if df is None or df.empty:
            return df

        # Remove null values and duplicates, reset index
        cleaned_df = df.dropna().drop_duplicates().reset_index(drop=True)

        return cleaned_df

    @staticmethod
    def remove_index_like_columns(df, verbose=True):

        if df is None or df.empty:
            return df

        index_like_columns = []

        for col in df.columns:
            # If all values in the column are unique, it's likely an ID column
            if len(df[col].unique()) == len(df):
                index_like_columns.append(col)

        # Drop index-like columns
        if index_like_columns:
            if verbose:
                print(f"Removing index-like columns: {index_like_columns}")
            df = df.drop(columns=index_like_columns)

        return df

    @staticmethod
    def full_clean(df, verbose=True):

        if df is None or df.empty:
            return df

        # Step 1: Basic cleaning
        cleaned_df = DataCleaner.clean_dataframe(df)

        # Step 2: Remove index-like columns
        cleaned_df = DataCleaner.remove_index_like_columns(cleaned_df, verbose)

        if verbose:
            print(f"Data cleaning complete. Shape: {cleaned_df.shape}")

        return cleaned_df