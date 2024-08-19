import pandas as pd
import numpy as np
from typing import List, Dict
from itertools import combinations

from model_weird_behavior.calculate_features import KeystrokeFeatureExtractor


def split_df_by_user(df, n_groups, group_length):
    # Get unique users
    users = df["user_id"].unique()

    # Initialize an empty list to store the resulting DataFrames
    result_dfs = []

    for user in users:
        # Get data for the current user
        user_data = df[df["user_id"] == user].reset_index(drop=True)
        if len(user_data) == 0:
            print(f"User {user} has no data.")

        # Calculate the total number of rows needed
        total_rows = n_groups * group_length

        # If user_data has fewer rows than needed, pad it with NaN values
        if len(user_data) < total_rows:
            pad_length = total_rows - len(user_data)
            pad_df = pd.DataFrame(
                np.nan, index=range(pad_length), columns=user_data.columns
            )
            user_data = pd.concat([user_data, pad_df], ignore_index=True)

        # If user_data has more rows than needed, truncate it
        elif len(user_data) > total_rows:
            user_data = user_data.iloc[:total_rows]

        # Reshape the data into n_groups of length group_length
        reshaped_data = user_data.values.reshape(n_groups, group_length, -1)

        # Convert the reshaped data back to DataFrames, add group number, and add to the result list
        for group_num, group in enumerate(reshaped_data, 1):
            group_df = pd.DataFrame(group, columns=user_data.columns)
            group_df["interval"] = group_num
            result_dfs.append(group_df)

    return result_dfs


# Import your feature extractor


def extract_features_from_groups(result_dfs):
    # Initialize the feature extractor
    # extractor = KeystrokeFeatureExtractor()

    # List to store feature DataFrames
    feature_dfs = []

    for idx, df in enumerate(result_dfs):
        try:
            extractor = KeystrokeFeatureExtractor(df)
            # Extract features
            features = extractor.get_features()

            # Add group number and user to the features DataFrame
            features["interval"] = df["interval"].iloc[
                0
            ]  # Assuming group_number is the same for all rows in df
            features["user_id"] = df["user_id"].iloc[
                0
            ]  # Assuming user is the same for all rows in df

            # Append to the list of feature DataFrames
            feature_dfs.append(features)

            # Optional: Print progress
            # if (idx + 1) % 100 == 0:
            #     print(f"Processed {idx + 1} groups")
        except IndexError:
            continue

    # # Combine all feature DataFrames into a single DataFrame
    # all_features = pd.concat(feature_dfs, ignore_index=True)

    return feature_dfs


def split_dataframe_by_user(df, user_id_column, rows_per_segment=1000):
    segments = []

    for user_id, user_df in df.groupby(user_id_column):
        num_segments = len(user_df) // rows_per_segment

        for i in range(num_segments):
            start_idx = i * rows_per_segment
            end_idx = (i + 1) * rows_per_segment
            segment = user_df.iloc[start_idx:end_idx].reset_index(drop=True)
            segment["interval"] = i
            segments.append(segment)

    return segments


def filter_by_row_count(df, user_id_column, rows_per_user):
    # Group the DataFrame by user_id and filter users with enough rows
    user_groups = df.groupby(user_id_column)
    filtered_groups = [group for _, group in user_groups if len(group) >= rows_per_user]

    # Concatenate the filtered groups and reset the index
    filtered_df = pd.concat(filtered_groups)
    filtered_df = filtered_df.reset_index(drop=True)

    # Truncate each user's data to the specified number of rows
    result = []
    for _, group in filtered_df.groupby(user_id_column):
        result.append(group.iloc[:rows_per_user])

    return pd.concat(result).reset_index(drop=True)


def format_features_to_df(results: List[Dict[str, float]]) -> pd.DataFrame:
    features_list = []
    for result in results:

        # pd.DataFrame.from_dict(results[0], orient='index', columns=cols)
        features = pd.DataFrame.from_dict([result])
        features_list.append(features)

    data = pd.concat(features_list, ignore_index=True)
    return data


def get_all_combos(input_list, list_length) -> List[List[str]]:
    return list(combinations(input_list, list_length))


def reduce_data(df, user_id_column, rows_per_user, users_to_test):
    filtered_data = filter_by_row_count(df, user_id_column, rows_per_user)
    filtered_data = filtered_data.loc[filtered_data[user_id_column].isin(users_to_test)]
    return filtered_data
