from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

from model_weird_behavior.process_data import (
    split_dataframe_by_user,
    extract_features_from_groups,
    format_features_to_df,
)


def rf_cross_validation(X, y, n_estimators=100, cv=5, random_state=42):
    """
    Perform k-fold cross-validation for a Random Forest classifier.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    n_estimators : int, optional (default=100)
        The number of trees in the forest.
    cv : int, optional (default=5)
        Number of folds for cross-validation.
    random_state : int, optional (default=42)
        Controls the randomness of the classifier.

    Returns:
    float
        The mean accuracy score across all CV folds.
    """
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )

    scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring="f1_micro")

    mean_f1 = np.mean(scores)
    return mean_f1


def run_rf_w_cv(reduced_data, rows_per_segment, user_id_column):
    n_users = len(reduced_data[user_id_column].unique())
    segments = split_dataframe_by_user(
        reduced_data, "user_id", rows_per_segment=rows_per_segment
    )

    features = extract_features_from_groups(segments)
    features = format_features_to_df(features)
    # Get X and y
    X = features.drop(["user_id", "interval"], axis=1)
    y = features["user_id"]

    try:
        if n_users < 3:
            cv_split = n_users
        else:
            cv_split = 3
        cv_f1 = rf_cross_validation(X, y, n_estimators=250, cv=cv_split)
        return cv_f1
    except ValueError:
        return np.nan
