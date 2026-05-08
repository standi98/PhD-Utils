import pytest
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler

import phd_utils.ml_utils as ml_utils


class TestSimplePandasDataset:
    @pytest.fixture
    def basic_df(self):
        return pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "target": [0.0, 1.0]})

    def test_dataset_length(self, basic_df):
        dataset = ml_utils.SimplePandasDataset(basic_df, target_col="target", feature_cols=["a", "b"])
        assert len(dataset) == 2

    def test_dataset_feature_shape(self, basic_df):
        dataset = ml_utils.SimplePandasDataset(basic_df, target_col="target", feature_cols=["a", "b"])
        X, y = dataset[0]
        assert X.shape == (2,)

    def test_dataset_target_shape(self, basic_df):
        dataset = ml_utils.SimplePandasDataset(basic_df, target_col="target", feature_cols=["a", "b"])
        X, y = dataset[0]
        assert y.shape == ()

    def test_dataset_rejects_nonexistent_target_col(self, basic_df):
        with pytest.raises(ValueError):
            ml_utils.SimplePandasDataset(basic_df, target_col="nonexistent", feature_cols=["a", "b"])

    @pytest.mark.parametrize("df", [
                                    pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "target": [0.0, 1.0]}),
                                    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "target": ["x", "y"]}),
                                    ], ids=["non_numeric_feature", "non_numeric_target"])
    def test_dataset_rejects_non_numeric(self, df):
        with pytest.raises(ValueError):
            ml_utils.SimplePandasDataset(df, target_col="target", feature_cols=["a", "b"])



class TestSimpleNNModel:
    @pytest.mark.parametrize("output_size, expected_shape", [
                                                                (1, (32,)),
                                                                (3, (32, 3)),
                                                                (10, (32, 10)),
                                                            ], ids=["single_regression", "multi_output_regression", "classification"])
    def test_model_output_shape(self, output_size, expected_shape):
        model = ml_utils.SimpleNNModel(input_size=8, output_size=output_size, layer_sizes=[], activation=torch.nn.ReLU, dropout=None)
        x = torch.randn(32, 8)
        assert model(x).shape == expected_shape

    @pytest.mark.parametrize("output_size, expected_shape", [
                                                                (1, (1,)),   # Critical case: must not squeeze to scalar ()
                                                                (3, (1, 3)),
                                                                (10, (1, 10)),
                                                            ], ids=["single_regression", "multi_output", "classification"])
    def test_model_output_with_batch_size_one(self, output_size, expected_shape):
        model = ml_utils.SimpleNNModel(input_size=8, output_size=output_size, layer_sizes=[], activation=torch.nn.ReLU, dropout=None)
        x = torch.randn(1, 8)
        output = model(x)
        assert output.shape == torch.Size(expected_shape), (f"Expected shape {expected_shape}, got {output.shape}.")

    @pytest.mark.parametrize("layer_sizes, expected_num_layers", [
                                                                    ([64, 32], 3),
                                                                    ([64], 2),
                                                                    ([64, 32, 16], 4),
                                                                ], ids=["two_hidden", "one_hidden", "three_hidden"])
    def test_model_layer_count(self, layer_sizes, expected_num_layers):
        model = ml_utils.SimpleNNModel(input_size=8, output_size=1, layer_sizes=layer_sizes, activation=torch.nn.ReLU, dropout=None)
        assert len(model.layers) == expected_num_layers

    def test_model_loss_decreases(self):
        model = ml_utils.SimpleNNModel(input_size=4, output_size=1, layer_sizes=[8, 8], activation=torch.nn.ReLU, dropout=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        x = torch.randn(32, 4)
        y = torch.randn(32)

        initial_loss = criterion(model(x), y).item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        final_loss = criterion(model(x), y).item()
        assert final_loss < initial_loss

    @pytest.mark.parametrize("activation", [torch.nn.ReLU, torch.nn.Sigmoid, str, 'invalid'], ids=["relu", "sigmoid", "invalid_type", "invalid_string"])
    def test_model_rejects_invalid_activation(self, activation):
        if (activation == 'invalid') or (activation == str):
            with pytest.raises(TypeError):
                ml_utils.SimpleNNModel(input_size=8, output_size=1, layer_sizes=[], activation=activation, dropout=None) # type: ignore
        else:
            model = ml_utils.SimpleNNModel(input_size=8, output_size=1, layer_sizes=[], activation=activation, dropout=None)
            assert isinstance(model.activation, activation)




class TestPandasScaler:
    @pytest.fixture
    def basic_df(self):
        return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "target": [0.0, 1.0, 2.0]})

    @pytest.fixture
    def fitted_scaler(self, basic_df):
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        scaler.fit(basic_df, target_col="target", feature_cols=["a", "b"], scale_target=False)
        return scaler

    @pytest.fixture
    def fitted_scaler_with_target(self, basic_df):
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        scaler.fit(basic_df, target_col="target", feature_cols=["a", "b"], scale_target=True)
        return scaler

    # --- Init ---
    def test_rejects_non_transformer(self):
        with pytest.raises(ValueError):
            ml_utils.PandasScaler(scaler="standard") # type: ignore[arg-type]

    # --- Fit ---
    def test_fit_returns_self(self, basic_df): # Fit should return self to allow method chaining
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        result = scaler.fit(basic_df, target_col="target", feature_cols=["a", "b"], scale_target=False)
        assert result is scaler

    def test_fit_stores_fitted_attributes(self, basic_df):
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        scaler.fit(basic_df, target_col="target", feature_cols=["a", "b"], scale_target=False)
        assert hasattr(scaler, "scaler_")
        assert hasattr(scaler, "target_col_")
        assert hasattr(scaler, "feature_cols_")

    def test_fit_does_not_modify_original_scaler(self, basic_df):
        original_scaler = StandardScaler()
        scaler = ml_utils.PandasScaler(scaler=original_scaler)
        scaler.fit(basic_df, target_col="target", feature_cols=["a", "b"], scale_target=False)
        assert not hasattr(original_scaler, "mean_")  # original scaler should not be fitted

    # --- Transform ---
    def test_transform_returns_dataframe(self, basic_df, fitted_scaler):
        result = fitted_scaler.transform(basic_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_preserves_columns(self, basic_df, fitted_scaler):
        result = fitted_scaler.transform(basic_df)
        assert sorted(result.columns.tolist()) == sorted(basic_df.columns.tolist())

    def test_transform_does_not_modify_input(self, basic_df, fitted_scaler):
        original = basic_df.copy()
        fitted_scaler.transform(basic_df)
        pd.testing.assert_frame_equal(basic_df, original)

    def test_transform_does_not_scale_target(self, basic_df, fitted_scaler):
        result = fitted_scaler.transform(basic_df)
        pd.testing.assert_series_equal(result["target"], basic_df["target"])

    def test_transform_scales_features(self, basic_df, fitted_scaler):
        result = fitted_scaler.transform(basic_df)
        assert not result["a"].equals(basic_df["a"])
        assert not result["b"].equals(basic_df["b"])

    def test_transform_before_fit_raises(self, basic_df):
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        with pytest.raises(ValueError):
            scaler.transform(basic_df)

    def test_transform_with_target_scaling(self, basic_df, fitted_scaler_with_target):
        result = fitted_scaler_with_target.transform(basic_df)
        assert not result["target"].equals(basic_df["target"])

    # --- Validation ---
    def test_rejects_non_dataframe(self):
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        with pytest.raises(TypeError):
            scaler.fit("not_a_df", target_col="target", feature_cols=["a"]) # type: ignore[arg-type]
 
    def test_rejects_missing_target_col(self, basic_df):
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        with pytest.raises(ValueError):
            scaler.fit(basic_df, target_col="nonexistent", feature_cols=["a", "b"], scale_target=False)

    def test_rejects_missing_feature_col(self, basic_df):
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        with pytest.raises(ValueError):
            scaler.fit(basic_df, target_col="target", feature_cols=["a", "nonexistent"], scale_target=False)

    def test_rejects_empty_feature_cols(self, basic_df):
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        with pytest.raises(ValueError):
            scaler.fit(basic_df, target_col="target", feature_cols=[], scale_target=False)

    def test_rejects_column_mismatch(self, basic_df):
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        with pytest.raises(ValueError):
            scaler.fit(basic_df, target_col="target", feature_cols=["a"], scale_target=False)  # missing "b"

    def test_rejects_nan_values(self):
        df = pd.DataFrame({"a": [1.0, None], "b": [3.0, 4.0], "target": [0.0, 1.0]})
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        with pytest.raises(ValueError):
            scaler.fit(df, target_col="target", feature_cols=["a", "b"], scale_target=False)

    def test_rejects_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": [], "target": []})
        scaler = ml_utils.PandasScaler(scaler=StandardScaler())
        with pytest.raises(ValueError):
            scaler.fit(df, target_col="target", feature_cols=["a", "b"], scale_target=False)

