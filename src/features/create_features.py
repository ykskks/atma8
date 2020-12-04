import sys
from pathlib import Path

import pandas as pd
import numpy as np
import feather
from sklearn.preprocessing import LabelEncoder

sys.path.append('.')
from base import get_arguments, FeatureGenerator, Feature


Feature.base_dir = './data/features'


class Base(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        def careful_encode(series, encode_mode):
            series = series.copy()
            train_series = series[:len(train)]
            test_series = series[len(train):]
            only_test_element = set(test_series.dropna()) - set(train_series.dropna())
            only_test_element_idx = series.isin(only_test_element)

            if encode_mode == "le":
                # testのみ水準はtrainのmodeで置換
                mode = series.value_counts().keys()[0]
                series[only_test_element_idx] = mode
                nan_idx = series.isnull()
                series.fillna("NaN", inplace=True)
                series = LabelEncoder().fit_transform(series.astype(str))
                series = series.astype(str)
                series[nan_idx] = np.nan

            elif encode_mode == "ce":
                nan_idx = series.isnull()
                series.fillna("NaN", inplace=True)
                series = series.astype(str)
                freq = series.value_counts()
                series = series.map(freq)
                series = series.astype(str)
                series[nan_idx] = np.nan

            return series.astype(float)

        def encode_category(df):
            df_copy = df.copy()
            le_cols = ["Platform", "Genre", "Rating"]
            ce_cols = ["Name", "Publisher", "Developer"]

            for col in le_cols:
                series = df_copy[col]
                df_copy[col] = careful_encode(series, "le")

            for col in ce_cols:
                series = df_copy[col]
                df_copy[col] = careful_encode(series, "ce")

            return df_copy

        whole_df = encode_category(whole_df)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        train["User_Score"] = train["User_Score"].replace("tbd", "999")
        test["User_Score"] = test["User_Score"].replace("tbd", "999")
        train["User_Score"] = train["User_Score"].astype(float)
        test["User_Score"] = test["User_Score"].astype(float)

        gen_cols = [
            "Platform", "Genre", "Rating", "Name", "Publisher", "Developer", "Year_of_Release",
            "Critic_Score", "Critic_Count", "User_Score", "User_Count"
        ]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


if __name__ == '__main__':
    TRAIN_PATH = Path("./data/raw/train.csv")
    TEST_PATH = Path("./data/raw/test.csv")

    args = get_arguments()

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    FeatureGenerator(globals(), args.force, args.which).run()
