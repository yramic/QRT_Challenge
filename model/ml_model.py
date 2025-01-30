import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class ModelTrainer:
    def __init__(self, model, n_splits=4, random_state=0):
        self.model = model
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = []
        self.scores = []

    def cross_validate(self, X, Y):
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



    def predict(self, X, group_col=None):
        """
        Predict on new data.
        """
        if group_col:
            X_copy = X.copy()
            X_copy['pred'] = self.model.predict_proba(X.drop(columns=[group_col]))[:, 1]
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