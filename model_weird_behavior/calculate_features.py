import pandas as pd


class KeystrokeFeatureExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self._parse_input_data()
        self.features = self._extract_features()

    def _parse_input_data(self):
        data = []
        with open(self.file_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    key, action, timestamp = parts
                    data.append(
                        {"key": key, "action": action, "timestamp": int(timestamp)}
                    )
        return pd.DataFrame(data)

    def _extract_features(self):
        features = {
            "WPM": self._calculate_wpm(),
            "NegUD": self._calculate_neg_ud(),
            "NegUU": self._calculate_neg_uu(),
            "ErrorRate": self._calculate_error_rate(),
            "CapsLockUsage": self._calculate_capslock_usage(),
        }
        features.update(self._calculate_shift_key_usage())
        return features

    def _calculate_wpm(self):
        total_time = (
            (self.df["timestamp"].max() - self.df["timestamp"].min()) / 1000 / 60
        )  # in minutes
        word_count = (
            len(self.df[self.df["key"] == "Space"]) + 1
        )  # Assuming each space separates words
        return word_count / total_time if total_time > 0 else 0

    def _calculate_neg_ud(self):
        key_presses = self.df[self.df["action"] == "KeyDown"].sort_values("timestamp")
        key_releases = self.df[self.df["action"] == "KeyUp"].sort_values("timestamp")

        neg_ud_count = 0
        total_keypairs = 0

        for i in range(len(key_presses) - 1):
            ud_time = (
                key_presses.iloc[i + 1]["timestamp"] - key_releases.iloc[i]["timestamp"]
            )
            if ud_time < 0:
                neg_ud_count += 1
            total_keypairs += 1

        return neg_ud_count / total_keypairs if total_keypairs > 0 else 0

    def _calculate_neg_uu(self):
        key_releases = self.df[self.df["action"] == "KeyUp"].sort_values("timestamp")

        neg_uu_count = 0
        total_keypairs = 0

        for i in range(len(key_releases) - 1):
            uu_time = (
                key_releases.iloc[i + 1]["timestamp"]
                - key_releases.iloc[i]["timestamp"]
            )
            if uu_time < 0:
                neg_uu_count += 1
            total_keypairs += 1

        return neg_uu_count / total_keypairs if total_keypairs > 0 else 0

    def _calculate_error_rate(self):
        backspace_count = len(self.df[self.df["key"] == "Back"])
        total_keys = len(self.df[self.df["action"] == "KeyDown"])
        return backspace_count / total_keys if total_keys > 0 else 0

    # I don't think this works because
    def _calculate_capslock_usage(self):
        capslock_count = len(self.df[self.df["key"] == "CapsLock"])
        total_capital_letters = len(
            self.df[(self.df["key"].str.len() == 1) & (self.df["key"].str.isupper())]
        )
        return (
            capslock_count / total_capital_letters if total_capital_letters > 0 else 0
        )

    def _calculate_shift_key_usage(self):
        shift_keys = self.df[self.df["key"].isin(["LShiftKey", "RShiftKey"])]
        capital_letters = self.df[
            (self.df["key"].str.len() == 1) & (self.df["key"].str.isupper())
        ]

        rsa, rsb, lsa, lsb = 0, 0, 0, 0
        total_shifts = 0

        for _, shift in shift_keys.iterrows():
            next_letter = capital_letters[
                capital_letters["timestamp"] > shift["timestamp"]
            ].iloc[0]
            if shift["key"] == "RShiftKey":
                if shift["timestamp"] < next_letter["timestamp"]:
                    rsa += 1
                else:
                    rsb += 1
            else:  # LShiftKey
                if shift["timestamp"] < next_letter["timestamp"]:
                    lsa += 1
                else:
                    lsb += 1
            total_shifts += 1

        return {
            "RSA": rsa / total_shifts if total_shifts > 0 else 0,
            "RSB": rsb / total_shifts if total_shifts > 0 else 0,
            "LSA": lsa / total_shifts if total_shifts > 0 else 0,
            "LSB": lsb / total_shifts if total_shifts > 0 else 0,
        }

    def get_features(self):
        return self.features
