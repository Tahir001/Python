from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

class DataModeler:
    """
    A class for modeling transaction data, including data preparation, imputation, 
    model fitting, and prediction.

    Attributes:
        data (pd.DataFrame): The original input data.
        model (RandomForestClassifier): The trained model.
        train_df (pd.DataFrame): The prepared training data.
        pipeline (Pipeline): The scikit-learn pipeline for data processing and modeling.
        outcome (pd.Series): The target variable for the model.
    """

    def __init__(self, sample_df: pd.DataFrame):
        """
        Initialize the DataModeler with a sample DataFrame.
        """
        self.data = sample_df
        self.model = None
        self.train_df = None
        self.pipeline = None
        self.outcome = None

    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare the data for modeling by converting dates and selecting relevant columns.
        Returns:
            pd.DataFrame: The prepared DataFrame with 'amount' and 'transaction_date' columns.
        """
        df = oos_df if oos_df is not None else self.data.copy()
        
        if 'outcome' in df.columns:
            self.outcome = df['outcome']
            df = df.drop(columns=['outcome'])

        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['transaction_date'] = df['transaction_date'].view(np.int64).astype(np.float64)

        df = df[['amount', 'transaction_date']]
        
        if oos_df is None:
            self.train_df = df
        return df

    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Impute missing values in the dataset using column average strategy
        Returns:
            pd.DataFrame: The DataFrame with imputed values.
        """
        df = oos_df if oos_df is not None else self.train_df.copy()

        imputer = SimpleImputer(strategy='mean')
        df[['amount', 'transaction_date']] = imputer.fit_transform(df)

        if 'customer_id' in df.columns:
            df.index = self.data['customer_id']

        return df

    def fit(self) -> None:
        """
        Fit the model using the prepared and imputed training data.
        This method creates and fits a pipeline with StandardScaler and RandomForestClassifier.
        """
        X = self.train_df
        y = self.outcome

        self.pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42))
        ])

        self.pipeline.fit(X, y)

    def model_summary(self) -> str:
        """
        Generate a summary of the fitted model, including the model name and feature importances.
        Returns:
            str: A string containing the model summary.
        """
        if self.pipeline is None:
            return "Model has not been fit yet."

        model = self.pipeline.named_steps['model']
        model_name = type(model).__name__
        feature_importances = model.feature_importances_
        feature_names = ['amount', 'transaction_date']
        feature_summary = "\n".join([f"{name}: {importance}" for name, importance in zip(feature_names, feature_importances)])

        return f"Model Name: {model_name}\nFeature importances:\n{feature_summary}"

    def predict(self, oos_df: pd.DataFrame = None) -> np.ndarray:
        """
        Make predictions using the fitted model.
        Returns:
            np.ndarray: An array of predictions.
        """
        df = oos_df if oos_df is not None else self.train_df
        predictions = self.pipeline.predict(df)
        return predictions

    def save(self, path: str) -> None:
        """
        Save the fitted pipeline to a file.
        Args:
            path (str): The file path to save the model.
        """
        joblib.dump(self.pipeline, path)

    @staticmethod
    def load(path: str) -> DataModeler:
        """
        Load a saved DataModeler instance from a file
        Args:
            path (str): The file path to load the model from
        Returns:
            DataModeler: A DataModeler instance with the loaded pipeline.
        """
        modeler = DataModeler(pd.DataFrame())
        modeler.pipeline = joblib.load(path)
        return modeler

# Execution
if __name__ == "__main__":
#################################################################################
# You should not have to modify the code below this point

    transact_train_sample = pd.DataFrame(
        {
            "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
            "transaction_date": [
                '2022-01-01',
                '2022-08-01',
                None,
                '2022-12-01',
                '2022-02-01',
                None,
                '2022-02-01',
                '2022-01-01',
                '2022-11-01',
                '2022-01-01'
            ],
            "outcome" : [False, True, True, True, False, False, True, True, True, False]
        }
    )


    print(f"Training sample:\n{transact_train_sample}\n")

    # <Expected Output>
    # Training sample:
    #    customer_id  amount transaction_date  outcome
    # 0           11     1.0       2022-01-01    False
    # 1           12     3.0       2022-08-01     True
    # 2           13    12.0             None     True
    # 3           14     6.0       2022-12-01     True
    # 4           15     0.5       2022-02-01    False
    # 5           16     0.2             None    False
    # 6           17     NaN       2022-02-01     True
    # 7           18     5.0       2022-01-01     True
    # 8           19     NaN       2022-11-01     True
    # 9           20     3.0       2022-01-01    False


    print(f"Current dtypes:\n{transact_train_sample.dtypes}\n")

    # <Expected Output>
    # Current dtypes:
    # customer_id           int64
    # amount              float64
    # transaction_date     object
    # outcome                bool
    # dtype: object

    transactions_modeler = DataModeler(transact_train_sample)

    transactions_modeler.prepare_data()

    print(f"Changed columns to:\n{transactions_modeler.train_df.dtypes}\n")

    # <Expected Output>
    # Changed columns to:
    # amount              float64
    # transaction_date    float64
    # dtype: object

    transactions_modeler.impute_missing()

    print(f"Imputed missing as mean:\n{transactions_modeler.train_df}\n")

    # <Expected Output>
    # Imputed missing as mean:
    #               amount  transaction_date
    # customer_id
    # 11            1.0000      1.640995e+18
    # 12            3.0000      1.659312e+18
    # 13           12.0000      1.650845e+18
    # 14            6.0000      1.669853e+18
    # 15            0.5000      1.643674e+18
    # 16            0.2000      1.650845e+18
    # 17            3.8375      1.643674e+18
    # 18            5.0000      1.640995e+18
    # 19            3.8375      1.667261e+18
    # 20            3.0000      1.640995e+18


    print("Fitting  model")
    transactions_modeler.fit()

    print(f"Fit model:\n{transactions_modeler.model_summary()}\n")

    # <Expected Output>
    # Fitting  model
    # Fit model:
    # <<< ANY SHORT SUMMARY OF THE MODEL YOU CHOSE >>>

    in_sample_predictions = transactions_modeler.predict()
    print(f"Predicted on training sample: {in_sample_predictions}\n")
    print(f'Accuracy = {sum(in_sample_predictions ==  [False, True, True, True, False, False, True, True, True, False])/.1}%')

    # <Expected Output>
    # Predicting on training sample [False  True  True  True False False True  True  True False]
    # Accuracy = 100.0%

    transactions_modeler.save("transact_modeler")
    loaded_modeler = DataModeler.load("transact_modeler")

    print(f"Loaded DataModeler sample df:\n{loaded_modeler.model_summary()}\n")

    # <Expected Output>
    # Loaded DataModeler sample df:
    # <<< THE SUMMARY OF THE MODEL YOU CHOSE >>>

    transact_test_sample = pd.DataFrame(
        {
            "customer_id": [21, 22, 23, 24, 25],
            "amount": [0.5, np.nan, 8, 3, 2],
            "transaction_date": [
                '2022-02-01',
                '2022-11-01',
                '2022-06-01',
                None,
                '2022-02-01'
            ]
        }
    )

    adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)

    print(f"Changed columns to:\n{adjusted_test_sample.dtypes}\n")

    # <Expected Output>
    # Changed columns to:
    # amount              float64
    # transaction_date    float64
    # dtype: object

    filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)

    print(f"Imputed missing as mean:\n{filled_test_sample}\n")

    # <Expected Output>
    # Imputed missing as mean:
    #              amount  transaction_date
    # customer_id
    # 21           0.5000      1.643674e+18
    # 22           3.8375      1.667261e+18
    # 23           8.0000      1.654042e+18
    # 24           3.0000      1.650845e+18
    # 25           2.0000      1.643674e+18

    oos_predictions = transactions_modeler.predict(filled_test_sample)
    print(f"Predicted on out of sample data: {oos_predictions}\n")
    print(f'Accuracy = {sum(oos_predictions == [False, True, True, False, False])/.05}%')

    # <Expected Output>
    # Predicted on out of sample data: [False True True False False] ([0 1 1 0 0])
    # Accuracy = 100.0%

