from model_weird_behavior.process_data import reduce_data
from model_weird_behavior.test_model import run_rf_w_cv


def process_combo(user_combo, df, rows_per_user, window_sizes, user_id_column):
    reduced_data = reduce_data(df, "user_id", rows_per_user, user_combo)
    return [
        (user_combo, len(user_combo), w, run_rf_w_cv(reduced_data, w, user_id_column))
        for w in window_sizes
    ]
