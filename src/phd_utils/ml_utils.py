import inspect
from typing import Literal, Type
from dataclasses import dataclass, field

import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin, clone


class SimplePandasDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 target_col: str,
                 feature_cols: list[str]
                 ):
        """PyTorch Dataset wrapping a DataFrame. Strict column matching to catch 
        data mismatches early — feature_cols and target_col must cover all columns exactly."""
        # Basic checks
        self._validate_inputs(df, target_col, feature_cols)

        # Convert DataFrame to tensors
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def _validate_inputs(self,
                         df: pd.DataFrame,
                         target_col: str,
                         feature_cols: list[str]
                         ):
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        if len(feature_cols) == 0:
            raise ValueError("feature_cols must not be empty")
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found")

        # Check that the columns are all numeric types.
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"Target column '{target_col}' must be numeric, got {df[target_col].dtype}")

        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Feature column '{col}' must be numeric, got {df[col].dtype}")

        # Check that the target and feature columns match the dataset exactly. 
        # This is strict to ensure that the user is aware of any discrepancies in the dataset and
        #  to prevent silent errors where columns are missing or extra columns are present.
        expected = sorted([target_col] + feature_cols)
        actual = sorted(df.columns.tolist())
        if expected != actual:
            raise ValueError(
                f"Columns mismatch. The feature and target columns passed must match the dataset exactly.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )

        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        if df[feature_cols].isnull().sum().sum() != 0:
            raise ValueError("Feature columns contain NaN values")


class SimpleNNModel(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 layer_sizes: list[int],
                 activation: Type[torch.nn.Module],
                 dropout: float|None
                 ):
        """Feedforward regression network. Pass layer_sizes=[] for a direct 
        linear input-to-output model with no hidden layers."""
        super(SimpleNNModel, self).__init__()
        self._validate_input(input_size, output_size, layer_sizes, activation, dropout)

        self.layers = torch.nn.ModuleList()
        
        # Combine input, hidden, and output sizes into one list for easier layer creation
        layer_sizes = [input_size] + layer_sizes + [output_size] 

        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.activation = activation()
        self.squeeze_output = (output_size == 1) # Flag to indicate if output should be squeezed. Squeezing avoids shape mismatch in loss functions
        
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        for layer in self.layers[:-1]: # Iterate and apply activation through the layers
            x = layer(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)

        x = self.layers[-1](x) # No activation on the output layer for regression

        if self.squeeze_output: # If output is a single value, flatten the output
            x = x.squeeze(-1)

        return x
    
    def _validate_input(self, input_size, output_size, layer_sizes, activation, dropout):
        if input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        if output_size <= 0:
            raise ValueError("output_size must be a positive integer")
        if not isinstance(layer_sizes, list):
            raise TypeError("layer_sizes must be a list of integers")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("All layer sizes must be positive integers")
        if not inspect.isclass(activation) or not issubclass(activation, torch.nn.Module):
            raise TypeError("activation must be a subclass of torch.nn.Module")
        if dropout is not None:
            if not (0 <= dropout < 1):
                raise ValueError("dropout must be between 0 and 1, or None for no dropout")



class PandasScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler: BaseEstimator):
        """Wraps any sklearn scaler to work directly on DataFrames, preserving 
        column names. Strict column matching on transform to catch train/test mismatches."""
        self.scaler = scaler
        if not isinstance(scaler, TransformerMixin):
            raise ValueError("Scaler must be an instance of sklearn's TransformerMixin")

    def fit(self,
            df: pd.DataFrame,
            target_col: str,
            feature_cols: list[str],
            scale_target: bool
            ) -> "PandasScaler":
        """Fits the scaler to the feature columns (and optionally the target column) of the DataFrame.
           Stores the column names for later validation during transform."""
        
        # Store feature and target column names
        self._validate_inputs(df, target_col, feature_cols)
        self.target_col_ = target_col
        self.feature_cols_ = feature_cols
        self.scale_target_ = scale_target

        # Fit scaler
        self.scaler_ = clone(self.scaler).set_output(transform="pandas") # Clone the scaler to avoid modifying the original instance
        self.scaler_.fit(df[self.feature_cols_])

        # If scaling the target, fit a separate scaler for the target column
        if self.scale_target_:
            self.target_scaler_ = clone(self.scaler).set_output(transform="pandas") # Clone the scaler to avoid modifying the original instance
            self.target_scaler_.fit(df[[self.target_col_]])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the feature columns (and optionally the target column) of the DataFrame using the fitted scaler."""
        if not hasattr(self, "scaler_"):
            raise ValueError("The scaler has not been fitted yet. Please call fit() before transform().")
        self._validate_inputs(df, self.target_col_, self.feature_cols_)
        
        df = df.copy() # Avoid modifying the original DataFrame
        df[self.feature_cols_] = self.scaler_.transform(df[self.feature_cols_])

        if self.scale_target_:
            df[[self.target_col_]] = self.target_scaler_.transform(df[[self.target_col_]])

        return df

    def fit_transform(self, # type: ignore[override]  Override fit_transform to include column validation and matching
                      df: pd.DataFrame,
                      target_col: str,
                      feature_cols: list[str],
                      scale_target: bool
                      ) -> pd.DataFrame:
        """Fits the scaler and transforms the DataFrame in one step. This is a convenience method that calls fit() followed by transform()."""
        return self.fit(df, target_col=target_col, feature_cols=feature_cols, scale_target=scale_target).transform(df)

    def inverse_transform_target(self, values: pd.Series) -> pd.Series:
        if not self.scale_target_:
            raise ValueError("Target was not scaled, nothing to inverse transform.")
        return self.target_scaler_.inverse_transform(values.to_frame()).squeeze()

    def _validate_inputs(self, df: pd.DataFrame, target_col: str, feature_cols: list[str]):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        if len(feature_cols) == 0:
            raise ValueError("feature_cols must not be empty")
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Feature column '{col}' not found")

        # Check that the target and feature columns match the dataset exactly, this is strict to ensure that the user is aware of any discrepancies in the dataset and to prevent silent errors where columns are missing or extra columns are present.
        expected = sorted([target_col] + feature_cols)
        actual = sorted(df.columns.tolist())
        if expected != actual:
            raise ValueError(
                f"Columns mismatch. The feature and target columns passed must match the dataset exactly.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )

        if df[feature_cols].isnull().sum().sum() != 0:
            raise ValueError("Feature columns contain NaN values")
        if len(df) == 0:
            raise ValueError("DataFrame is empty")

        # Check that the feature columns are all numeric types (int or float). This is to give the user a clear warning and not a confusing error when fitting.
        invalid_cols = {
            col: df[col].dtype
            for col in feature_cols
            if not (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]))
        }
        if invalid_cols:
            formatted = "\n".join(f"  '{col}': {dtype}" for col, dtype in invalid_cols.items())
            raise TypeError(
                f"Feature columns contain unsupported dtypes. Only numeric (float/int) types are supported.\n"
                f"Offending columns:\n{formatted}"
            )


def convert_timedelta_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all timedelta columns in the DataFrame to float (by default, total seconds).
    Returns a copy of the DataFrame with the converted columns."""
    df = df.copy()
    timedelta_cols = [col for col in df.columns if pd.api.types.is_timedelta64_dtype(df[col])]
    if len(timedelta_cols) == 0:
        raise ValueError("No timedelta columns found in the DataFrame")
    for col in timedelta_cols:
        df[col] = df[col].dt.total_seconds()
    print(f"Converted timedelta columns to float: {timedelta_cols}")
    return df



@dataclass
class DataSplits:
    """Simple dataclass to hold train/val/test splits along with metadata about the splitting method and parameters used."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    method: Literal["date_splitter", "random_splitter"]
    params: dict = field(default_factory=dict)
    is_scaled: bool = False

    def info(self):
        """Prints basic info about the splits and the splitting method used."""
        print(f"Splitting method: {self.method}")
        print(f"Parameters: {self.params}")
        print(f"Train set: {len(self.train)} samples")
        print(f"Validation set: {len(self.val)} samples")
        print(f"Test set: {len(self.test)} samples")
        if self.train.index.is_monotonic_increasing:
            print(f"Train set date range: {self.train.index.min()} to {self.train.index.max()}")
            print(f"Validation set date range: {self.val.index.min()} to {self.val.index.max()}")
            print(f"Test set date range: {self.test.index.min()} to {self.test.index.max()}")

    def fit_and_scale(self, scaler: PandasScaler, target_col: str, feature_cols: list[str], scale_target: bool) -> "DataSplits":
        """Applies a PandasScaler to the train/val/test splits, returning a new DataSplits object with the scaled data."""
        if not isinstance(scaler, PandasScaler):
            raise TypeError("Scaler must be an instance of PandasScaler")
        if self.is_scaled:
            raise ValueError("Data is already scaled. Scaling again may lead to data leakage. Please check that you are creating a new DataSplits object after scaling:D(It's more safe).")

        scaler.fit(self.train, target_col=target_col, feature_cols=feature_cols, scale_target=scale_target)

        scaled_train = scaler.transform(self.train)
        scaled_val = scaler.transform(self.val)
        scaled_test = scaler.transform(self.test)

        return DataSplits(train=scaled_train,
                          val=scaled_val,
                          test=scaled_test,
                          method=self.method,
                          params=self.params,
                          is_scaled=True)


def date_splitter(df: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp) -> DataSplits:
    """Splits a DataFrame into train/val/test sets based on date. Assumes the DataFrame has a datetime index."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted in ascending order — call df.sort_index() first")
    if not (isinstance(train_end, pd.Timestamp) and isinstance(val_end, pd.Timestamp)):
        raise TypeError("train_end and val_end must be pandas Timestamps")
    if not (train_end < val_end):
        raise ValueError("train_end must be before val_end")

    # TODO: Add checks to ensure that time snapping is not a issue, with 15 minute data train_end = '2023-01-01' would include data up to '2023-01-01 00:00:00' but not '2023-01-01 00:15:00'.
    # The dataset is so large that we ignore this for now, but in smaller datasets this could lead to significant data leakage. 

    train = df[df.index <= train_end]
    val = df[(df.index > train_end) & (df.index <= val_end)]
    test = df[df.index > val_end]

    if len(train) == 0:
        raise ValueError("Train split is empty — check your train_end date")
    if len(val) == 0:
        raise ValueError("Validation split is empty — check your train_end and val_end dates")
    if len(test) == 0:
        raise ValueError("Test split is empty — check your val_end date")

    return DataSplits(train=train,
                      val=val,
                      test=test,
                      method="date_splitter",
                      params={"train_end": train_end, "val_end": val_end})


def random_splitter(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, random_state: int|None = None) -> DataSplits:
    """Splits a DataFrame into train/val/test sets randomly. The fractions must sum to 1."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and 0 < test_frac < 1):
        raise ValueError("Fractions must be between 0 and 1")
    if not abs(train_frac + val_frac + test_frac - 1) < 1e-6:
        raise ValueError("Fractions must sum to 1")

    df_shuffled = df.sample(frac=1, random_state=random_state)
    n = len(df)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    test = df_shuffled.iloc[val_end:]
    val = df_shuffled.iloc[train_end:val_end]
    train = df_shuffled.iloc[:train_end]

    return DataSplits(train=train,
                      val=val, test=test,
                      method="random_splitter",
                      params={"train_frac": train_frac, "val_frac": val_frac, "test_frac": test_frac, "random_state": random_state})


@dataclass
class SplitPandasDatasets:
    """Simple dataclass to hold train/val/test PyTorch Datasets along with metadata about the splitting method and parameters used."""
    train: SimplePandasDataset
    val: SimplePandasDataset
    test: SimplePandasDataset

    @classmethod
    def from_DataSplits(cls,
                        splits: DataSplits,
                        target_col: str,
                        feature_cols: list[str]
                        ) -> "SplitPandasDatasets":
        if not isinstance(splits, DataSplits):
            raise TypeError("splits must be an instance of DataSplits")
        return cls(
                   train=SimplePandasDataset(splits.train, target_col=target_col, feature_cols=feature_cols),
                   val=SimplePandasDataset(splits.val, target_col=target_col, feature_cols=feature_cols),
                   test=SimplePandasDataset(splits.test, target_col=target_col, feature_cols=feature_cols),
                   )

@dataclass
class DataLoaders:
    """Simple dataclass to hold train/val/test PyTorch DataLoaders along with metadata about the splitting method and parameters used."""
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader

    @classmethod
    def from_SplitPandasDatasets(cls,
                                 datasets: SplitPandasDatasets,
                                 batch_size: int,
                                 shuffle_train: bool = True
                                 ) -> "DataLoaders":
        """Factory method to create DataLoaders from a SplitPandasDatasets object."""
        if not isinstance(datasets, SplitPandasDatasets):
            raise TypeError("datasets must be an instance of SplitPandasDatasets")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        train_loader = torch.utils.data.DataLoader(datasets.train, batch_size=batch_size, shuffle=shuffle_train)
        val_loader = torch.utils.data.DataLoader(datasets.val, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(datasets.test, batch_size=batch_size, shuffle=False)

        return cls(train=train_loader, val=val_loader, test=test_loader)



