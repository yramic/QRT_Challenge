import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from typing import Union
from sklearn.base import BaseEstimator
from torch.nn import Module as TorchModule

class ModelTrainer:
    def __init__(
            self, 
            model: Union[BaseEstimator, TorchModule], 
            use_model: str, 
            n_splits: int = 4, 
            random_state: int = 0
        ) -> None:
        """Initialize Model Trainer"""
        self.model = model
        self.use_model = use_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = []
        self.scores = []

    def cross_validate(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        """
        Perform standard K-fold cross-validation without using date-based splits.
        """
        Y = Y.loc[X.index]  # Align Y with X

        kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)

        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            # Subset the data using indices
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

            # Train the model
            self.model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred_prob = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_prob > np.median(y_pred_prob)).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            self.scores.append(accuracy)
            self.models.append(self.model)

            print(f"Fold {i+1} - Accuracy: {accuracy * 100:.2f}%")

        mean_score = np.mean(self.scores) * 100
        std_score = np.std(self.scores) * 100
        print(f"Cross-validation Accuracy: {mean_score:.2f}% ± {std_score:.2f}%")

    
    def validate_model(
            self, 
            X: pd.DataFrame, 
            Y: pd.DataFrame,
            time_limit: int = 3600, # 1 hour time limit for training
            num_gpus: int = 1,
            num_cpus: int = 1
        ) -> None:
        """For some NN models or something like AutoGluon Cross Validation can't be used
        thus the model's performance can be evaluated by a simple train /test split."""

        assert self.use_model in ['AutoGluon', 'TabNet'], (
            "This validation is only valid for AutoGluon and TabNet; else use Cross Validation"
        )

        if self.use_model == 'AutoGluon':
            X = pd.concat((X, Y), axis=1)

            X_train, X_eval = train_test_split(
                X,  
                test_size=0.2,  
                random_state=42, 
                stratify=X["RET"]
            )

            self.model.fit(
                train_data=X_train,
                time_limit=time_limit, 
                verbosity=3,
                num_gpus=num_gpus,
                num_cpus=num_cpus
            )

            prediction = self.model.predict_proba(X_eval)[True]
            Y_pred = prediction.transform(
                lambda x: x > x.median()).values
            
            Y_eval = X_eval['RET']
            
        else:
            X_train, X_eval, Y_train, Y_eval = train_test_split(
                X, Y,
                test_size=0.2,
                random_state=42,
                stratify=Y
            )

            self.model.fit(
                X_train.values.astype(np.float32), 
                Y_train.values.astype(np.int32), 
                eval_set=[
                    (X_eval.values.astype(np.float32), 
                     Y_eval.values.astype(np.int32))
                ], 
                max_epochs=100
            )

            Y_proba = self.model.predict_proba(X_eval.values.astype(np.float32))[:, 1]
            Y_pred = (Y_proba > np.median(Y_proba)).astype(int)
       
        print('MEDIAN', accuracy_score(Y_pred, Y_eval))


    def predict(self, X, group_col=None):
        """
        Predict on new data.
        """
        if group_col:
            X_copy = X.copy()
            if self.use_model != 'AutoGluon':
                X_copy['pred'] = self.model.predict_proba(X.drop(columns=[group_col]))[:, 1]
            elif self.use_model == 'AutoGluon':
                X_copy['pred'] = self.model.predict_proba(X.drop(columns=[group_col]))[True]
            return X_copy.groupby(group_col)['pred'].transform(lambda x: x > x.median()).astype(int)
        return self.model.predict(X)


    def save_submission(self, pred, test_set, file_path='y_test.csv', target_name='RET'):
        """
        Save predictions to a CSV file.
        """
        submission = pd.Series(pred)
        submission.index = test_set['ID']
        submission.name = target_name
        submission.to_csv(file_path, index=True, header=True)
        print(f"Submission saved to {file_path}.")