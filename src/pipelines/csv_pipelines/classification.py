import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple

from pipelines.base_pipeline import BasePipeline, convert_to_tensor


class ClassificationPipeline_V1(BasePipeline):
    def __init__(
        self, df: pd.DataFrame, target_col: str, null_threshold: float = 0.7
    ) -> None:
        self.df = df
        self.target_col = target_col
        self.null_threshold = null_threshold

        # Pipeline State
        self.drop_columns = []
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()

        self.label_categories = None
        self.feature_columns_count = None

    def _validate_input(self):
        print("Step 1: Running Input Validation")
        if self.df is None or self.df.empty:
            raise ValueError("Pipeline Halted: Dataset is empty.")

        if len(self.df.columns) < 2:
            raise ValueError("Pipeline Halted: Need at least 1 feature and 1 label.")

        if self.target_col not in self.df.columns:
            raise KeyError(f"Pipeline Halted: '{self.target_col}' not found.")

        label_null_ratio = self.df[self.target_col].isnull().mean()
        if label_null_ratio > self.null_threshold:
            raise ValueError(
                f"Pipeline Halted: Target is {label_null_ratio:.1%} empty."
            )

        self.df = self.df.dropna(subset=[self.target_col])

    def _train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify: pd.Series,
        test_split: float = 0.2,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return tuple(
            train_test_split(
                X,
                y,
                stratify=stratify,
                test_size=test_split,
                random_state=random_state,
                shuffle=shuffle,
            )
        )

    def _handle_imputation(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        print("Step 2: Imputing missing values")
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

        if not num_cols.empty:
            X_train[num_cols] = np.asarray(
                self.num_imputer.fit_transform(X_train[num_cols])
            )
            X_test[num_cols] = np.asarray(self.num_imputer.transform(X_test[num_cols]))

        if not cat_cols.empty:
            X_train[cat_cols] = np.asarray(
                self.cat_imputer.fit_transform(X_train[cat_cols])
            )
            X_test[cat_cols] = np.asarray(self.cat_imputer.transform(X_test[cat_cols]))

        return X_train, X_test

    def _handle_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        print("Step 3: Encoding and Scaling")
        # Identify types
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

        # Encode categorical safely
        if not cat_cols.empty:
            X_train_cat = np.asarray(self.encoder.fit_transform(X_train[cat_cols]))
            X_test_cat = np.asarray(self.encoder.transform(X_test[cat_cols]))
        else:
            X_train_cat = np.empty((X_train.shape[0], 0))
            X_test_cat = np.empty((X_test.shape[0], 0))

        # Extract numeric safely
        if not num_cols.empty:
            X_train_num = X_train[num_cols].values
            X_test_num = X_test[num_cols].values
        else:
            X_train_num = np.empty((X_train.shape[0], 0))
            X_test_num = np.empty((X_test.shape[0], 0))

        # Combine back
        X_train_final = np.hstack([X_train_num, X_train_cat])
        X_test_final = np.hstack([X_test_num, X_test_cat])

        # Scale everything
        if X_train_final.shape[1] > 0:
            X_train_scaled = np.asarray(self.scaler.fit_transform(X_train_final))
            X_test_scaled = np.asarray(self.scaler.transform(X_test_final))
        else:
            X_train_scaled = X_train_final
            X_test_scaled = X_test_final

        return X_train_scaled, X_test_scaled

    def pipeline(self):
        self._validate_input()

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # 1. Split
        X_train, X_test, y_train, y_test = self._train_test_split(X, y, stratify=y)

        # 2. Drop high-null columns
        self.drop_columns = [
            c
            for c in X_train.columns
            if X_train[c].isnull().mean() > self.null_threshold
        ]
        X_train = X_train.drop(columns=self.drop_columns)
        X_test = X_test.drop(columns=self.drop_columns)

        # 3. Impute
        X_train, X_test = self._handle_imputation(X_train, X_test)

        # 4. Encode & Scale
        X_train_proc, X_test_proc = self._handle_encoding(X_train, X_test)

        # 5. Process Labels
        self.label_categories = pd.Categorical(y_train).categories
        y_train_enc = pd.Categorical(y_train, categories=self.label_categories).codes
        y_test_enc = pd.Categorical(y_test, categories=self.label_categories).codes

        # 6. Final Tensors
        self.feature_columns_count = X_train_proc.shape[1]

        print("Pipeline Completed Successfully")
        return (
            convert_to_tensor(X_train_proc, torch.float32),
            convert_to_tensor(X_test_proc, torch.float32),
            convert_to_tensor(y_train_enc, torch.long),
            convert_to_tensor(y_test_enc, torch.long),
        )
