import sys
from pathlib import Path
import gc
import pickle

from tqdm import tqdm
import pandas as pd
import numpy as np
import feather
from sklearn.preprocessing import LabelEncoder
import transformers
from transformers import BertTokenizer
import torch
tqdm.pandas()

sys.path.append('.')
from base import get_arguments, FeatureGenerator, Feature


Feature.base_dir = './data/features'


def careful_encode(series, encode_mode):
    series = series.copy()
    train_series = series[:len(train)]
    test_series = series[len(train):]
    target = train["Global_Sales"]
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

    elif encode_mode == "te":
        nan_idx = series.isnull()
        series.fillna("NaN", inplace=True)
        from sklearn.model_selection import GroupKFold
        folds = GroupKFold(n_splits=5)
        groups = train["Publisher"].to_frame()
        tmp = pd.concat([train_series, target], axis=1)
        tmp.columns = ["col", "target"]
        agg_df = tmp.groupby('col').agg({'target': ['sum', 'count']})
        ts = pd.Series(np.empty(tmp.shape[0]), index=tmp.index)
        for train_idx, val_idx in folds.split(tmp, groups=groups):
            _, _val = tmp.iloc[train_idx], tmp.iloc[val_idx]
            holdout_agg_df = _val.groupby('col').agg({'target': ['sum', 'count']})
            train_agg_df = agg_df - holdout_agg_df
            oof_ts = _val.apply(lambda row: train_agg_df.loc[row.col][('target', 'sum')] / (train_agg_df.loc[row.col][('target', 'count')] + 1), axis=1)
            ts[oof_ts.index] = oof_ts

        train_agg_df = tmp.groupby('col').agg({'target': ['mean']})
        train_agg_df.columns = ["mean"]
        test_te = test_series.apply(lambda row: train_agg_df.loc[row]["mean"] if (row in train_agg_df.index) and (row != "NaN") else np.nan)
        series = pd.concat([ts, test_te], axis=0)
        series[nan_idx] = np.nan

    return series.astype(float)


class Base(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

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


class Base_2(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

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

        train["User_Score"] = train["User_Score"].replace("tbd", None)
        test["User_Score"] = test["User_Score"].replace("tbd", None)
        train["User_Score"] = train["User_Score"].astype(float)
        test["User_Score"] = test["User_Score"].astype(float)

        gen_cols = [
            "Platform", "Genre", "Rating", "Name", "Publisher", "Developer", "Year_of_Release",
            "Critic_Score", "Critic_Count", "User_Score", "User_Count"
        ]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Base_3(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

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

        train["User_Score"] = train["User_Score"].replace("tbd", None)
        test["User_Score"] = test["User_Score"].replace("tbd", None)
        train["User_Score"] = train["User_Score"].astype(float)
        test["User_Score"] = test["User_Score"].astype(float)

        # add
        platform_year_mode = whole_df.groupby("Platform")["Year_of_Release"].agg(lambda x:x.value_counts().index[0])
        whole_df.fillna(whole_df["Platform"].map(platform_year_mode), inplace=True)
        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].astype(float)

        gen_cols = [
            "Platform", "Genre", "Rating", "Name", "Publisher", "Developer", "Year_of_Release",
            "Critic_Score", "Critic_Count", "User_Score", "User_Count"
        ]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Base_4(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        def encode_category(df):
            df_copy = df.copy()
            te_cols = ["Platform", "Genre", "Rating"]
            ce_cols = ["Name", "Publisher", "Developer"]

            for col in te_cols:
                series = df_copy[col]
                df_copy[col] = careful_encode(series, "te")

            for col in ce_cols:
                series = df_copy[col]
                df_copy[col] = careful_encode(series, "ce")

            return df_copy

        whole_df = encode_category(whole_df)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        train["User_Score"] = train["User_Score"].replace("tbd", None)
        test["User_Score"] = test["User_Score"].replace("tbd", None)
        train["User_Score"] = train["User_Score"].astype(float)
        test["User_Score"] = test["User_Score"].astype(float)

        # add
        platform_year_mode = whole_df.groupby("Platform")["Year_of_Release"].agg(lambda x:x.value_counts().index[0])
        whole_df.fillna(whole_df["Platform"].map(platform_year_mode), inplace=True)
        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].astype(float)

        gen_cols = [
            "Platform", "Genre", "Rating", "Name", "Publisher", "Developer", "Year_of_Release",
            "Critic_Score", "Critic_Count", "User_Score", "User_Count"
        ]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class TargetEncode2way(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Rating"] = whole_df["Platform"] + "_" + whole_df["Rating"]
        whole_df["Platform_Genre"] = whole_df["Platform"] + "_" + whole_df["Genre"]
        whole_df["Platform_Year"] = whole_df["Platform"] + "_" + whole_df["Year_of_Release"]

        def encode_category(df):
            df_copy = df.copy()
            te_cols = ["Platform_Rating", "Platform_Genre", "Platform_Year"]

            for col in te_cols:
                series = df_copy[col]
                df_copy[col + "_" + "te"] = careful_encode(series, "te")

            return df_copy

        whole_df = encode_category(whole_df)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        gen_cols = ["Platform_Rating_te", "Platform_Genre_te", "Platform_Year_te"]
        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Base_5(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        ohe_col_names = []

        def encode_category(df):
            df_copy = df.copy()
            ohe_cols = ["Platform", "Genre", "Rating"]
            ce_cols = ["Name", "Publisher", "Developer"]

            for col in ohe_cols:
                series = df_copy[col]
                tmp = pd.get_dummies(series, prefix=col)
                df_copy[tmp.columns.tolist()] = tmp
                ohe_col_names.extend(tmp.columns.tolist())

            for col in ce_cols:
                series = df_copy[col]
                df_copy[col] = careful_encode(series, "ce")

            return df_copy

        whole_df = encode_category(whole_df)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        train["User_Score"] = train["User_Score"].replace("tbd", None)
        test["User_Score"] = test["User_Score"].replace("tbd", None)
        train["User_Score"] = train["User_Score"].astype(float)
        test["User_Score"] = test["User_Score"].astype(float)

        # add
        platform_year_mode = whole_df.groupby("Platform")["Year_of_Release"].agg(lambda x:x.value_counts().index[0])
        whole_df.fillna(whole_df["Platform"].map(platform_year_mode), inplace=True)
        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].astype(float)

        gen_cols = [
            "Name", "Publisher", "Developer", "Year_of_Release",
            "Critic_Score", "Critic_Count", "User_Score", "User_Count"
        ]
        gen_cols.extend(ohe_col_names)

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Base_6(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        ohe_col_names = []

        def encode_category(df):
            df_copy = df.copy()
            ohe_cols = ["Platform", "Genre", "Rating"]
            ce_cols = ["Name", "Developer"]

            for col in ohe_cols:
                series = df_copy[col]
                tmp = pd.get_dummies(series, prefix=col)
                df_copy[tmp.columns.tolist()] = tmp
                ohe_col_names.extend(tmp.columns.tolist())

            for col in ce_cols:
                series = df_copy[col]
                df_copy[col] = careful_encode(series, "ce")

            return df_copy

        whole_df = encode_category(whole_df)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        train["User_Score"] = train["User_Score"].replace("tbd", None)
        test["User_Score"] = test["User_Score"].replace("tbd", None)
        train["User_Score"] = train["User_Score"].astype(float)
        test["User_Score"] = test["User_Score"].astype(float)

        # add
        platform_year_mode = whole_df.groupby("Platform")["Year_of_Release"].agg(lambda x:x.value_counts().index[0])
        whole_df.fillna(whole_df["Platform"].map(platform_year_mode), inplace=True)
        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].astype(float)

        gen_cols = [
            "Name", "Developer", "Year_of_Release",
            "Critic_Score", "Critic_Count", "User_Score", "User_Count"
        ]
        gen_cols.extend(ohe_col_names)

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Bert(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        class BertSequenceVectorizer:
            def __init__(self):
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model_name = 'bert-base-uncased'
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
                self.bert_model = self.bert_model.to(self.device)
                # self.max_len = 128

            def vectorize(self, sentence : str) -> np.array:
                inp = self.tokenizer(sentence, return_tensors="pt")
                out = self.bert_model(**inp)
                return out.last_hidden_state[0][0].detach().numpy()

        BSV = BertSequenceVectorizer()

        # whole_df['Name_bert'] = whole_df['Name'].progress_apply(lambda x: BSV.vectorize(x) if not pd.isnull(x) else np.nan)
        # whole_df['Publisher_bert'] = whole_df['Publisher'].progress_apply(lambda x: BSV.vectorize(x) if not pd.isnull(x) else np.nan)
        # whole_df['Developer_bert'] = whole_df['Developer'].progress_apply(lambda x: BSV.vectorize(x) if not pd.isnull(x) else np.nan)

        # with open("./data/features/bert.pkl", mode="wb") as f:
        #     pickle.dump(whole_df[["Name_bert", "Publisher_bert", "Developer_bert"]], f)

        whole_df[["Name_bert", "Publisher_bert", "Developer_bert"]] = pickle.load(open("./data/features/bert.pkl", mode="rb"))

        name_bert, pub_bert, dev_bert = [], [], []

        for i, b in enumerate(whole_df["Name_bert"].isnull().values):
            if b:
                name_bert.append([np.nan] * 768)
            else:
                name_bert.append(whole_df.loc[i, "Name_bert"].tolist())

        for i, b in enumerate(whole_df["Publisher_bert"].isnull().values):
            if b:
                pub_bert.append([np.nan] * 768)
            else:
                pub_bert.append(whole_df.loc[i, "Publisher_bert"].tolist())

        for i, b in enumerate(whole_df["Developer_bert"].isnull().values):
            if b:
                dev_bert.append([np.nan] * 768)
            else:
                dev_bert.append(whole_df.loc[i, "Developer_bert"].tolist())

        name_bert = pd.DataFrame(name_bert)
        name_bert.columns = [f"name_bert_{i}" for i in range(768)]
        pub_bert = pd.DataFrame(pub_bert)
        pub_bert.columns = [f"pub_bert_{i}" for i in range(768)]
        dev_bert = pd.DataFrame(dev_bert)
        dev_bert.columns = [f"dev_bert_{i}" for i in range(768)]

        whole_df = pd.concat([whole_df, name_bert, pub_bert, dev_bert], axis=1)
        del name_bert, pub_bert, dev_bert
        gc.collect()

        train, test = whole_df[:len(train)], whole_df[len(train):]
        del whole_df
        gc.collect()

        gen_cols = [f"name_bert_{i}" for i in range(768)] + [f"pub_bert_{i}" for i in range(768)] + [f"dev_bert_{i}" for i in range(768)]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class SumBert(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df[["Name_bert", "Publisher_bert", "Developer_bert"]] = pickle.load(open("./data/features/bert.pkl", mode="rb"))

        name_bert, pub_bert, dev_bert = [], [], []

        for i, b in enumerate(whole_df["Name_bert"].isnull().values):
            if b:
                name_bert.append([np.nan] * 768)
            else:
                name_bert.append(whole_df.loc[i, "Name_bert"].tolist())

        for i, b in enumerate(whole_df["Publisher_bert"].isnull().values):
            if b:
                pub_bert.append([np.nan] * 768)
            else:
                pub_bert.append(whole_df.loc[i, "Publisher_bert"].tolist())

        for i, b in enumerate(whole_df["Developer_bert"].isnull().values):
            if b:
                dev_bert.append([np.nan] * 768)
            else:
                dev_bert.append(whole_df.loc[i, "Developer_bert"].tolist())

        name_bert = pd.DataFrame(name_bert)
        name_bert.columns = [f"name_bert_{i}" for i in range(768)]
        pub_bert = pd.DataFrame(pub_bert)
        pub_bert.columns = [f"pub_bert_{i}" for i in range(768)]
        dev_bert = pd.DataFrame(dev_bert)
        dev_bert.columns = [f"dev_bert_{i}" for i in range(768)]

        sum_bert = pd.DataFrame()
        for i in range(768):
            sum_bert[f"sum_bert_{i}"] = pd.concat([name_bert[f"name_bert_{i}"], pub_bert[f"pub_bert_{i}"], dev_bert[f"dev_bert_{i}"]], axis=1).mean(axis=1, skipna=True)

        whole_df = pd.concat([whole_df, sum_bert], axis=1)
        del sum_bert, name_bert, pub_bert, dev_bert
        gc.collect()

        train, test = whole_df[:len(train)], whole_df[len(train):]
        del whole_df
        gc.collect()

        gen_cols = [f"sum_bert_{i}" for i in range(768)]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class NameBert(Feature):
    def create_features(self):
        global train, test
        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df[["Name_bert", "Publisher_bert", "Developer_bert"]] = pickle.load(open("./data/features/bert.pkl", mode="rb"))

        name_bert = []

        for i, b in enumerate(whole_df["Name_bert"].isnull().values):
            if b:
                name_bert.append([np.nan] * 768)
            else:
                name_bert.append(whole_df.loc[i, "Name_bert"].tolist())

        name_bert = pd.DataFrame(name_bert)
        name_bert.columns = [f"name_bert_{i}" for i in range(768)]

        whole_df = pd.concat([whole_df, name_bert], axis=1)
        del name_bert
        gc.collect()

        train, test = whole_df[:len(train)], whole_df[len(train):]
        del whole_df
        gc.collect()

        gen_cols = [f"name_bert_{i}" for i in range(768)]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Interaction(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Genre"] = whole_df["Platform"] + "_" + whole_df["Genre"]
        whole_df["Platform_Rating"] = whole_df["Platform"] + "_" + whole_df["Rating"]
        whole_df["Genre_Rating"] = whole_df["Genre"] + "_" + whole_df["Rating"]
        whole_df["Platform_Year_of_Release"] = whole_df["Platform"] + "_" + whole_df["Year_of_Release"]

        def encode_category(df):
            df_copy = df.copy()
            ce_cols = ["Platform_Genre", "Platform_Rating", "Genre_Rating", "Platform_Year_of_Release"]
            # ce_cols = ["Name", "Publisher", "Developer"]

            # for col in le_cols:
            #     series = df_copy[col]
            #     df_copy[col] = careful_encode(series, "le")

            for col in ce_cols:
                series = df_copy[col]
                df_copy[col] = careful_encode(series, "ce")

            return df_copy

        whole_df = encode_category(whole_df)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        gen_cols = ["Platform_Genre", "Platform_Rating", "Genre_Rating", "Platform_Year_of_Release"]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Interaction_le(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Genre"] = whole_df["Platform"] + "_" + whole_df["Genre"]
        whole_df["Platform_Rating"] = whole_df["Platform"] + "_" + whole_df["Rating"]
        whole_df["Genre_Rating"] = whole_df["Genre"] + "_" + whole_df["Rating"]
        whole_df["Platform_Year_of_Release"] = whole_df["Platform"] + "_" + whole_df["Year_of_Release"]

        def encode_category(df):
            df_copy = df.copy()
            le_cols = ["Platform_Genre", "Platform_Rating", "Genre_Rating", "Platform_Year_of_Release"]
            # ce_cols = ["Name", "Publisher", "Developer"]

            for col in le_cols:
                series = df_copy[col]
                df_copy[col] = careful_encode(series, "le")

            # for col in ce_cols:
            #     series = df_copy[col]
            #     df_copy[col] = careful_encode(series, "ce")

            return df_copy

        whole_df = encode_category(whole_df)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        gen_cols = ["Platform_Genre", "Platform_Rating", "Genre_Rating", "Platform_Year_of_Release"]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Momentum(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        train, test = whole_df[:len(train)], whole_df[len(train):]
        gen_cols = ["Critic_Momentum", "User_Momentum"]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class CriticUser(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["is_tbd"] = (whole_df["User_Score"] == "tbd").astype(int)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float) * 10

        whole_df["Score_sub"] = whole_df["Critic_Score"] - whole_df["User_Score"]
        whole_df["Score_div"] = whole_df["Critic_Score"] / whole_df["User_Score"]
        whole_df['total_count'] = whole_df["Critic_Count"] + whole_df["User_Count"]
        whole_df["Count_sub"] = whole_df["Critic_Count"] - whole_df["User_Count"]
        whole_df["Count_div"] = whole_df["Critic_Count"] / whole_df["User_Count"]

        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]
        whole_df["Total_Momentum"] = whole_df["Critic_Momentum"] + whole_df["User_Momentum"]

        train, test = whole_df[:len(train)], whole_df[len(train):]
        gen_cols = ["is_tbd", "Score_sub", "Score_div", "total_count", "Count_sub", "Count_div", "Total_Momentum"]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class CriticUser_2(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        # whole_df["is_tbd"] = (whole_df["User_Score"] == "tbd").astype(int)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float) * 10

        # whole_df["Score_sub"] = whole_df["Critic_Score"] - whole_df["User_Score"]
        # whole_df["Score_div"] = whole_df["Critic_Score"] / whole_df["User_Score"]
        whole_df['total_count'] = whole_df["Critic_Count"] + whole_df["User_Count"]
        # whole_df["Count_sub"] = whole_df["Critic_Count"] - whole_df["User_Count"]
        # whole_df["Count_div"] = whole_df["Critic_Count"] / whole_df["User_Count"]

        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]
        whole_df["Total_Momentum"] = whole_df["Critic_Momentum"] + whole_df["User_Momentum"]

        train, test = whole_df[:len(train)], whole_df[len(train):]
        gen_cols = ["total_count", "Total_Momentum"]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class SimultaneousPlatformCount(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Name_Year_of_Release"] = whole_df["Name"] + "_" + whole_df["Year_of_Release"]

        simul_title_map = whole_df["Name_Year_of_Release"].value_counts()
        whole_df["simul_count"] = whole_df["Name_Year_of_Release"].map(simul_title_map)

        train, test = whole_df[:len(train)], whole_df[len(train):]
        gen_cols = ["simul_count"]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Remake(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Name_Year_of_Release"] = whole_df["Name"] + "_" + whole_df["Year_of_Release"]
        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].astype(float)

        def is_remake(name_year):
            if pd.isnull(name_year):
                return np.nan
            name, year = name_year.split("_")
            year = float(year)
            tmp = whole_df[whole_df["Name"] == name]
            return tmp["Year_of_Release"].min() < year

        whole_df["is_remake"] = whole_df["Name_Year_of_Release"].progress_apply(is_remake)
        whole_df["is_remake"] = whole_df["is_remake"].astype(float)

        train, test = whole_df[:len(train)], whole_df[len(train):]
        gen_cols = ["is_remake"]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class PublisherPCA(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        gen_cols = []

        # unknown -> nan
        whole_df["Publisher"] = whole_df["Publisher"].replace("unknown", np.nan)

        def pca(col):
            from sklearn.decomposition import PCA
            pivot = whole_df.pivot_table(index='Publisher', columns=col, values='Name', aggfunc='count').fillna(0)
            pca = PCA(n_components=7)
            transformed = pca.fit_transform(pivot)
            transformed_df = pd.DataFrame(transformed, index=pivot.index)
            transformed_df.columns = [f"publisher_{col}_pca_{i}" for i in range(7)]
            return transformed_df.reset_index()

        pca_cols = ["Platform", "Year_of_Release", "Genre", "Rating"]
        for col in pca_cols:
            ret = pca(col)
            whole_df = whole_df.merge(ret, on="Publisher", how="left")
            gen_cols.extend([col for col in ret if "pca" in col])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class DeveloperPCA(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        gen_cols = []

        def pca(col):
            from sklearn.decomposition import PCA
            pivot = whole_df.pivot_table(index='Developer', columns=col, values='Name', aggfunc='count').fillna(0)
            pca = PCA(n_components=7)
            transformed = pca.fit_transform(pivot)
            transformed_df = pd.DataFrame(transformed, index=pivot.index)
            transformed_df.columns = [f"developer_{col}_pca_{i}" for i in range(7)]
            return transformed_df.reset_index()

        pca_cols = ["Platform", "Year_of_Release", "Genre", "Rating"]
        for col in pca_cols:
            ret = pca(col)
            whole_df = whole_df.merge(ret, on="Developer", how="left")
            gen_cols.extend([col for col in ret if "pca" in col])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class PublisherLDA(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        gen_cols = []

        # unknown -> nan
        whole_df["Publisher"] = whole_df["Publisher"].replace("unknown", np.nan)

        def lda(col):
            from sklearn.decomposition import LatentDirichletAllocation
            pivot = whole_df.pivot_table(index='Publisher', columns=col, values='Name', aggfunc='count').fillna(0)
            lda = LatentDirichletAllocation(n_components=7)
            transformed = lda.fit_transform(pivot)
            transformed_df = pd.DataFrame(transformed, index=pivot.index)
            transformed_df.columns = [f"publisher_{col}_lda_{i}" for i in range(7)]
            return transformed_df.reset_index()

        lda_cols = ["Platform", "Year_of_Release", "Genre", "Rating"]
        for col in lda_cols:
            ret = lda(col)
            whole_df = whole_df.merge(ret, on="Publisher", how="left")
            gen_cols.extend([col for col in ret if "lda" in col])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class PublisherDeveloperLDA(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        gen_cols = []

        # unknown -> nan
        whole_df["Publisher"] = whole_df["Publisher"].replace("unknown", np.nan)

        def lda(col, n_components):
            from sklearn.decomposition import LatentDirichletAllocation
            pivot = whole_df.pivot_table(index='Publisher', columns=col, values='Name', aggfunc='count').fillna(0)
            lda = LatentDirichletAllocation(n_components=n_components)
            transformed = lda.fit_transform(pivot)
            transformed_df = pd.DataFrame(transformed, index=pivot.index)
            transformed_df.columns = [f"publisher_{col}_lda_{n_components}_{i}" for i in range(n_components)]
            return transformed_df.reset_index()

        lda_cols = ["Developer"]
        n_cs = [5, 20]
        for col in lda_cols:
            for n_c in n_cs:
                ret = lda(col, n_c)
                whole_df = whole_df.merge(ret, on="Publisher", how="left")
                gen_cols.extend([col for col in ret if "lda" in col])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class DeveloperPublisherLDA(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        gen_cols = []

        # unknown -> nan
        whole_df["Publisher"] = whole_df["Publisher"].replace("unknown", np.nan)

        def lda(col, n_components):
            from sklearn.decomposition import LatentDirichletAllocation
            pivot = whole_df.pivot_table(index='Developer', columns=col, values='Name', aggfunc='count').fillna(0)
            lda = LatentDirichletAllocation(n_components=n_components)
            transformed = lda.fit_transform(pivot)
            transformed_df = pd.DataFrame(transformed, index=pivot.index)
            transformed_df.columns = [f"developer_{col}_lda_{n_components}_{i}" for i in range(n_components)]
            return transformed_df.reset_index()

        lda_cols = ["Publisher"]
        n_cs = [5, 20]
        for col in lda_cols:
            for n_c in n_cs:
                ret = lda(col, n_c)
                whole_df = whole_df.merge(ret, on="Developer", how="left")
                gen_cols.extend([col for col in ret if "lda" in col])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class DeveloperLDA(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        gen_cols = []

        def lda(col):
            from sklearn.decomposition import LatentDirichletAllocation
            pivot = whole_df.pivot_table(index='Developer', columns=col, values='Name', aggfunc='count').fillna(0)
            lda = LatentDirichletAllocation(n_components=7)
            transformed = lda.fit_transform(pivot)
            transformed_df = pd.DataFrame(transformed, index=pivot.index)
            transformed_df.columns = [f"developer_{col}_lda_{i}" for i in range(7)]
            return transformed_df.reset_index()

        lda_cols = ["Platform", "Year_of_Release", "Genre", "Rating"]
        for col in lda_cols:
            ret = lda(col)
            whole_df = whole_df.merge(ret, on="Developer", how="left")
            gen_cols.extend([col for col in ret if "lda" in col])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class BertPCA(Feature):
    def create_features(self):
        global train, test
        gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df[["Name_bert", "Publisher_bert", "Developer_bert"]] = pickle.load(open("./data/features/bert.pkl", mode="rb"))

        name_bert, pub_bert, dev_bert = [], [], []

        for i, b in enumerate(whole_df["Name_bert"].isnull().values):
            if b:
                name_bert.append([np.nan] * 768)
            else:
                name_bert.append(whole_df.loc[i, "Name_bert"].tolist())

        for i, b in enumerate(whole_df["Publisher_bert"].isnull().values):
            if b:
                pub_bert.append([np.nan] * 768)
            else:
                pub_bert.append(whole_df.loc[i, "Publisher_bert"].tolist())

        for i, b in enumerate(whole_df["Developer_bert"].isnull().values):
            if b:
                dev_bert.append([np.nan] * 768)
            else:
                dev_bert.append(whole_df.loc[i, "Developer_bert"].tolist())

        name_bert = pd.DataFrame(name_bert)
        name_bert.columns = [f"name_bert_{i}" for i in range(768)]
        pub_bert = pd.DataFrame(pub_bert)
        pub_bert.columns = [f"pub_bert_{i}" for i in range(768)]
        dev_bert = pd.DataFrame(dev_bert)
        dev_bert.columns = [f"dev_bert_{i}" for i in range(768)]

        def pca(x, col_name):
            from sklearn.decomposition import PCA
            pca = PCA(n_components=30)
            nonnull_idx = (x.isnull().sum(axis=1) == 0)
            transformed = pca.fit_transform(x.loc[nonnull_idx])
            new_cols = [f"{col_name}_bert_pca_{i}" for i in range(30)]
            x[new_cols] = np.nan
            x.loc[nonnull_idx, new_cols] = transformed
            transformed_df = x[new_cols].copy()
            return transformed_df

        name_bert_pca = pca(name_bert, "name")
        pub_bert_pca = pca(pub_bert, "pub")
        dev_bert_pca = pca(dev_bert, "dev")

        del name_bert, pub_bert, dev_bert
        import gc; gc.collect()

        gen_cols.extend(name_bert_pca.columns.tolist())
        gen_cols.extend(pub_bert_pca.columns.tolist())
        gen_cols.extend(dev_bert_pca.columns.tolist())

        whole_df = pd.concat([whole_df, name_bert_pca, pub_bert_pca, dev_bert_pca], axis=1)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class BertPCA50(Feature):
    def create_features(self):
        global train, test
        gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df[["Name_bert", "Publisher_bert", "Developer_bert"]] = pickle.load(open("./data/features/bert.pkl", mode="rb"))

        name_bert, pub_bert, dev_bert = [], [], []

        for i, b in enumerate(whole_df["Name_bert"].isnull().values):
            if b:
                name_bert.append([np.nan] * 768)
            else:
                name_bert.append(whole_df.loc[i, "Name_bert"].tolist())

        for i, b in enumerate(whole_df["Publisher_bert"].isnull().values):
            if b:
                pub_bert.append([np.nan] * 768)
            else:
                pub_bert.append(whole_df.loc[i, "Publisher_bert"].tolist())

        for i, b in enumerate(whole_df["Developer_bert"].isnull().values):
            if b:
                dev_bert.append([np.nan] * 768)
            else:
                dev_bert.append(whole_df.loc[i, "Developer_bert"].tolist())

        name_bert = pd.DataFrame(name_bert)
        name_bert.columns = [f"name_bert_{i}" for i in range(768)]
        pub_bert = pd.DataFrame(pub_bert)
        pub_bert.columns = [f"pub_bert_{i}" for i in range(768)]
        dev_bert = pd.DataFrame(dev_bert)
        dev_bert.columns = [f"dev_bert_{i}" for i in range(768)]

        # unknown -> nan
        pub_unknown_idx = (whole_df["Publisher"] == "unknown")
        pub_bert.loc[pub_unknown_idx] = np.nan

        def pca(x, col_name):
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            nonnull_idx = (x.isnull().sum(axis=1) == 0)
            transformed = pca.fit_transform(x.loc[nonnull_idx])
            new_cols = [f"{col_name}_bert_pca_{i}" for i in range(50)]
            x[new_cols] = np.nan
            x.loc[nonnull_idx, new_cols] = transformed
            transformed_df = x[new_cols].copy()
            return transformed_df

        name_bert_pca = pca(name_bert, "name")
        pub_bert_pca = pca(pub_bert, "pub")
        dev_bert_pca = pca(dev_bert, "dev")

        del name_bert, pub_bert, dev_bert
        import gc; gc.collect()

        gen_cols.extend(name_bert_pca.columns.tolist())
        gen_cols.extend(pub_bert_pca.columns.tolist())
        gen_cols.extend(dev_bert_pca.columns.tolist())

        whole_df = pd.concat([whole_df, name_bert_pca, pub_bert_pca, dev_bert_pca], axis=1)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class BertTSNE(Feature):
    def create_features(self):
        global train, test
        gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df[["Name_bert", "Publisher_bert", "Developer_bert"]] = pickle.load(open("./data/features/bert.pkl", mode="rb"))

        name_bert, pub_bert, dev_bert = [], [], []

        for i, b in enumerate(whole_df["Name_bert"].isnull().values):
            if b:
                name_bert.append([np.nan] * 768)
            else:
                name_bert.append(whole_df.loc[i, "Name_bert"].tolist())

        for i, b in enumerate(whole_df["Publisher_bert"].isnull().values):
            if b:
                pub_bert.append([np.nan] * 768)
            else:
                pub_bert.append(whole_df.loc[i, "Publisher_bert"].tolist())

        for i, b in enumerate(whole_df["Developer_bert"].isnull().values):
            if b:
                dev_bert.append([np.nan] * 768)
            else:
                dev_bert.append(whole_df.loc[i, "Developer_bert"].tolist())

        name_bert = pd.DataFrame(name_bert)
        name_bert.columns = [f"name_bert_{i}" for i in range(768)]
        pub_bert = pd.DataFrame(pub_bert)
        pub_bert.columns = [f"pub_bert_{i}" for i in range(768)]
        dev_bert = pd.DataFrame(dev_bert)
        dev_bert.columns = [f"dev_bert_{i}" for i in range(768)]

        def tsne(x, col_name):
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            pca = PCA(n_components=50)
            tsne = TSNE(n_components=3, perplexity=10, random_state=42)
            nonnull_idx = (x.isnull().sum(axis=1) == 0)
            X = pca.fit_transform(x.loc[nonnull_idx])  # pca before tsne reccomened in sklearn doc
            transformed = tsne.fit_transform(X)
            new_cols = [f"{col_name}_bert_tsne_{i}" for i in range(3)]
            x[new_cols] = np.nan
            x.loc[nonnull_idx, new_cols] = transformed
            transformed_df = x[new_cols].copy()
            return transformed_df

        name_bert_tsne = tsne(name_bert, "name")
        pub_bert_tsne = tsne(pub_bert, "pub")
        dev_bert_tsne = tsne(dev_bert, "dev")

        del name_bert, pub_bert, dev_bert
        import gc; gc.collect()

        gen_cols.extend(name_bert_tsne.columns.tolist())
        gen_cols.extend(pub_bert_tsne.columns.tolist())
        gen_cols.extend(dev_bert_tsne.columns.tolist())

        whole_df = pd.concat([whole_df, name_bert_tsne, pub_bert_tsne, dev_bert_tsne], axis=1)
        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class Series(Feature):
    def create_features(self):
        global train, test

        import texthero as hero
        from texthero import preprocessing
        from nltk.util import ngrams

        gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        custom_pipeline = [preprocessing.fillna
                        , preprocessing.lowercase
                        , preprocessing.remove_digits
                        , preprocessing.remove_punctuation
                        , preprocessing.remove_diacritics
                        , preprocessing.remove_whitespace
                        , preprocessing.remove_stopwords
                        ]

        def line_ngram(line, n=2):
            words = [w for w in line.split(' ') if len(w) != 0] # 空文字は取り除く
            return list(ngrams(words, n))

        names = hero.clean(whole_df['Name'], custom_pipeline)
        name_grams = names.map(line_ngram)

        grams = [x for row in name_grams for x in row if len(x) > 0]
        top_grams = pd.Series(grams).value_counts().head(50).index

        for i, ng in enumerate(top_grams):
            col_name = "_".join(ng)
            whole_df[f"name_has_{col_name}"] = name_grams.map(lambda x: ng in x).astype(int)
            gen_cols.append(f"name_has_{col_name}")

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class NumSeries(Feature):
    def create_features(self):
        global train, test

        import texthero as hero
        from texthero import preprocessing
        from nltk.util import ngrams

        gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        custom_pipeline = [preprocessing.fillna
                        , preprocessing.lowercase
                        , preprocessing.remove_digits
                        , preprocessing.remove_punctuation
                        , preprocessing.remove_diacritics
                        , preprocessing.remove_whitespace
                        , preprocessing.remove_stopwords
                        ]

        # 連続2単語と3単語で頻出するやつをseriesとする
        def line_ngram(line):
            words = [w for w in line.split(' ') if len(w) != 0] # 空文字は取り除く
            return list(ngrams(words, 2)) + list(ngrams(words, 3))

        names = hero.clean(whole_df['Name'], custom_pipeline)
        name_grams = names.map(line_ngram)

        grams = [x for row in name_grams for x in row if len(x) > 0]
        top_grams = pd.Series(grams).value_counts().head(10000).sort_values(ascending=True)

        # 昇順なのでそのタイトルに含まれる最も出現頻度の高いシリーズがそのタイトルの値となる
        # n-gramのnの値については要検討
        whole_df["num_series"] = 0
        gen_cols.append("num_series")
        for g, n in tqdm(top_grams.items()):
            idx = name_grams.map(lambda x: g in x)
            whole_df.loc[idx, "num_series"] = n

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class NumSeriesNormalized(Feature):
    def create_features(self):
        global train, test

        import texthero as hero
        from texthero import preprocessing
        from nltk.util import ngrams

        gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        custom_pipeline = [preprocessing.fillna
                        , preprocessing.lowercase
                        , preprocessing.remove_digits
                        , preprocessing.remove_punctuation
                        , preprocessing.remove_diacritics
                        , preprocessing.remove_whitespace
                        , preprocessing.remove_stopwords
                        ]

        # 連続2単語と3単語で頻出するやつをseriesとする
        def line_ngram(line):
            words = [w for w in line.split(' ') if len(w) != 0] # 空文字は取り除く
            return list(ngrams(words, 2)) + list(ngrams(words, 3))

        # https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
        import scipy.stats

        def ent(data):
            """Calculates entropy of the passed `pd.Series`
            """
            p_data = data.value_counts()           # counts occurrence of each value
            entropy = scipy.stats.entropy(p_data)  # get entropy from counts
            return entropy

        names = hero.clean(whole_df['Name'], custom_pipeline)
        name_grams = names.map(line_ngram)

        grams = [x for row in name_grams for x in row if len(x) > 0]
        top_grams = pd.Series(grams).value_counts().head(10000).sort_values(ascending=True)

        # 昇順なのでそのタイトルに含まれる最も出現頻度の高いシリーズがそのタイトルの値となる
        # n-gramのnの値については要検討
        whole_df["num_series_norm"] = 0
        gen_cols.append("num_series_norm")
        for g, n in tqdm(top_grams.items()):
            idx = name_grams.map(lambda x: g in x)
            entropy = ent(whole_df.loc[idx, "Platform"])
            whole_df.loc[idx, "num_series_norm"] = n / entropy

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class SeriesLifespan(Feature):
    def create_features(self):
        global train, test

        import texthero as hero
        from texthero import preprocessing
        from nltk.util import ngrams

        gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        custom_pipeline = [preprocessing.fillna
                        , preprocessing.lowercase
                        , preprocessing.remove_digits
                        , preprocessing.remove_punctuation
                        , preprocessing.remove_diacritics
                        , preprocessing.remove_whitespace
                        , preprocessing.remove_stopwords
                        ]

        # 連続2単語と3単語で頻出するやつをseriesとする
        def line_ngram(line):
            words = [w for w in line.split(' ') if len(w) != 0] # 空文字は取り除く
            return list(ngrams(words, 2)) + list(ngrams(words, 3))

        names = hero.clean(whole_df['Name'], custom_pipeline)
        name_grams = names.map(line_ngram)

        grams = [x for row in name_grams for x in row if len(x) > 0]
        top_grams = pd.Series(grams).value_counts().head(10000).sort_values(ascending=True)

        # 昇順なのでそのタイトルに含まれる最も出現頻度の高いシリーズがそのタイトルの値となる
        # n-gramのnの値については要検討
        whole_df["series_lifespan"] = 0
        gen_cols.append("series_lifespan")
        for g, n in tqdm(top_grams.items()):
            idx = name_grams.map(lambda x: g in x)
            tmp = whole_df.loc[idx]
            lifespan = tmp["Year_of_Release"].max() - tmp["Year_of_Release"].min()
            whole_df.loc[idx, "series_lifespan"] = lifespan

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class SeriesOrder(Feature):
    def create_features(self):
        global train, test

        import texthero as hero
        from texthero import preprocessing
        from nltk.util import ngrams

        gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        custom_pipeline = [preprocessing.fillna
                        , preprocessing.lowercase
                        , preprocessing.remove_digits
                        , preprocessing.remove_punctuation
                        , preprocessing.remove_diacritics
                        , preprocessing.remove_whitespace
                        , preprocessing.remove_stopwords
                        ]

        # 連続2単語と3単語で頻出するやつをseriesとする
        def line_ngram(line):
            words = [w for w in line.split(' ') if len(w) != 0] # 空文字は取り除く
            return list(ngrams(words, 2)) + list(ngrams(words, 3))

        names = hero.clean(whole_df['Name'], custom_pipeline)
        name_grams = names.map(line_ngram)

        grams = [x for row in name_grams for x in row if len(x) > 0]
        top_grams = pd.Series(grams).value_counts().head(10000).sort_values(ascending=True)

        # 昇順なのでそのタイトルに含まれる最も出現頻度の高いシリーズがそのタイトルの値となる
        # n-gramのnの値については要検討
        whole_df["series_order"] = 0
        gen_cols.append("series_order")
        for g, n in tqdm(top_grams.items()):
            idx = name_grams.map(lambda x: g in x)
            tmp = whole_df.loc[idx]
            whole_df.loc[idx, "series_order"] = tmp["Year_of_Release"].rank(method="dense")

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class SeriesRelative(Feature):
    def create_features(self):
        global train, test

        import texthero as hero
        from texthero import preprocessing
        from nltk.util import ngrams

        gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        custom_pipeline = [preprocessing.fillna
                        , preprocessing.lowercase
                        , preprocessing.remove_digits
                        , preprocessing.remove_punctuation
                        , preprocessing.remove_diacritics
                        , preprocessing.remove_whitespace
                        , preprocessing.remove_stopwords
                        ]

        # 連続2単語と3単語で頻出するやつをseriesとする
        def line_ngram(line):
            words = [w for w in line.split(' ') if len(w) != 0] # 空文字は取り除く
            return list(ngrams(words, 2)) + list(ngrams(words, 3))

        names = hero.clean(whole_df['Name'], custom_pipeline)
        name_grams = names.map(line_ngram)

        grams = [x for row in name_grams for x in row if len(x) > 0]
        top_grams = pd.Series(grams).value_counts().head(10000).sort_values(ascending=True)

        # 昇順なのでそのタイトルに含まれる最も出現頻度の高いシリーズがそのタイトルの値となる
        # n-gramのnの値については要検討
        cols = [
            "user_score_relative_to_series_mean",
            "critic_score_relative_to_series_mean",
            "user_count_relative_to_series_mean",
            "critic_count_relative_to_series_mean",
            "user_mometum_relative_to_series_mean",
            "critic_momentum_relative_to_series_mean"
        ]
        whole_df[cols] = np.nan
        gen_cols.extend(cols)

        for g, n in tqdm(top_grams.items()):
            idx = name_grams.map(lambda x: g in x)
            tmp = whole_df.loc[idx]

            if len(tmp["User_Score"].dropna()) > 0:
                mean = tmp["User_Score"].dropna().mean()
                whole_df.loc[idx, "user_score_relative_to_series_mean"] = whole_df.loc[idx, "User_Score"] - mean

            if len(tmp["Critic_Score"].dropna()) > 0:
                mean = tmp["Critic_Score"].dropna().mean()
                whole_df.loc[idx, "critic_score_relative_to_series_mean"] = whole_df.loc[idx, "Critic_Score"] - mean

            if len(tmp["User_Count"].dropna()) > 0:
                mean = tmp["User_Count"].dropna().mean()
                whole_df.loc[idx, "user_count_relative_to_series_mean"] = whole_df.loc[idx, "User_Count"] - mean

            if len(tmp["Critic_Count"].dropna()) > 0:
                mean = tmp["Critic_Count"].dropna().mean()
                whole_df.loc[idx, "critic_count_relative_to_series_mean"] = whole_df.loc[idx, "Critic_Count"] - mean

            if len(tmp["User_Momentum"].dropna()) > 0:
                mean = tmp["User_Momentum"].dropna().mean()
                whole_df.loc[idx, "user_mometum_relative_to_series_mean"] = whole_df.loc[idx, "User_Momentum"] - mean

            if len(tmp["Critic_Momentum"].dropna()) > 0:
                mean = tmp["Critic_Momentum"].dropna().mean()
                whole_df.loc[idx, "critic_momentum_relative_to_series_mean"] = whole_df.loc[idx, "Critic_Momentum"] - mean

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromPlatform(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Platform"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromPlatform(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Platform"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromYear(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Year_of_Release"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromYear(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Year_of_Release"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


def generate_groupby_diff_features(whole_df, key_cols, methods):
    gen_cols = []
    target_cols = ["User_Score", "Critic_Score", "User_Count", "Critic_Count", "User_Momentum", "Critic_Momentum"]
    for key_col in key_cols:
        for method in methods:
            new_cols = [f"{col}_diff_{key_col}_{method}" for col in target_cols]
            gen_cols.extend(new_cols)
            whole_df[new_cols] = np.nan
            for i in whole_df[key_col].unique():
                idx = (whole_df[key_col] == i)
                tmp = whole_df.loc[idx]
                for c in target_cols:
                    if len(tmp[c].dropna()) > 0:
                        if method == "mean":
                            agg = tmp[c].dropna().mean()
                        elif method == "max":
                            agg = tmp[c].dropna().max()
                        elif method == "min":
                            agg = tmp[c].dropna().min()
                        whole_df.loc[idx, f"{c}_diff_{key_col}_{method}"] = whole_df.loc[idx, c] - agg
    return whole_df, gen_cols


def generate_groupby_ratio_features(whole_df, key_cols, methods):
    gen_cols = []
    target_cols = ["User_Score", "Critic_Score", "User_Count", "Critic_Count", "User_Momentum", "Critic_Momentum"]
    for key_col in key_cols:
        for method in methods:
            new_cols = [f"{col}_ratio_{key_col}_{method}" for col in target_cols]
            gen_cols.extend(new_cols)
            whole_df[new_cols] = np.nan
            for i in whole_df[key_col].unique():
                idx = (whole_df[key_col] == i)
                tmp = whole_df.loc[idx]
                for c in target_cols:
                    if len(tmp[c].dropna()) > 0:
                        if method == "mean":
                            agg = tmp[c].dropna().mean()
                        elif method == "max":
                            agg = tmp[c].dropna().max()
                        elif method == "min":
                            agg = tmp[c].dropna().min()
                        whole_df.loc[idx, f"{c}_diff_{key_col}_{method}"] = whole_df.loc[idx, c] / agg
    return whole_df, gen_cols


class ScoreDiffFromGenre(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Genre"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromGenre(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Genre"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromPublisher(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        # unknown -> nan
        whole_df["Publisher"] = whole_df["Publisher"].replace("unknown", np.nan)

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Publisher"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromPublisher(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df["Publisher"] = whole_df["Publisher"].replace("unknown", np.nan)

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Publisher"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromDeveloper(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Developer"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromDeveloper(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Developer"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromRating(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Rating"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromRating(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Rating"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromPlatformYear(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Year"] = whole_df["Platform"] + "_" + whole_df["Year_of_Release"]

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Platform_Year"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromPlatformYear(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Year"] = whole_df["Platform"] + "_" + whole_df["Year_of_Release"]

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Platform_Year"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromPlatformGenre(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        # whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Genre"] = whole_df["Platform"] + "_" + whole_df["Genre"]

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Platform_Genre"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromPlatformGenre(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        # whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Genre"] = whole_df["Platform"] + "_" + whole_df["Genre"]

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Platform_Genre"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromPlatformRating(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        # whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Rating"] = whole_df["Platform"] + "_" + whole_df["Rating"]

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Platform_Rating"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromPlatformRating(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        # whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Rating"] = whole_df["Platform"] + "_" + whole_df["Rating"]

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Platform_Rating"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromGenreRating(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        # whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Genre_Rating"] = whole_df["Genre"] + "_" + whole_df["Rating"]

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Genre_Rating"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromGenreRating(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        # whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Genre_Rating"] = whole_df["Genre"] + "_" + whole_df["Rating"]

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Genre_Rating"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreDiffFromPlatformGenreRating(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        # whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Genre_Rating"] = whole_df["Platform"] + "_" + whole_df["Genre"] + "_" + whole_df["Rating"]

        whole_df, gen_cols = generate_groupby_diff_features(whole_df, ["Platform_Genre_Rating"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class ScoreRatioFromPlatformGenreRating(Feature):
    def create_features(self):
        global train, test

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)
        whole_df["User_Score"] = whole_df["User_Score"].replace("tbd", np.nan)
        whole_df["User_Score"] = whole_df["User_Score"].astype(float)
        whole_df["Critic_Momentum"] = whole_df["Critic_Score"] * whole_df["Critic_Count"]
        whole_df["User_Momentum"] = whole_df["User_Score"] * whole_df["User_Count"]

        # whole_df["Year_of_Release"] = whole_df["Year_of_Release"].apply(lambda x: str(x) if not pd.isnull(x) else np.nan)
        whole_df["Platform_Genre_Rating"] = whole_df["Platform"] + "_" + whole_df["Genre"] + "_" + whole_df["Rating"]

        whole_df, gen_cols = generate_groupby_ratio_features(whole_df, ["Platform_Genre_Rating"], ["mean", "max", "min"])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class PlatformLDA(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        gen_cols = []

        # unknown -> nan
        whole_df["Publisher"] = whole_df["Publisher"].replace("unknown", np.nan)

        def lda(col, n_compo):
            from sklearn.decomposition import LatentDirichletAllocation
            pivot = whole_df.pivot_table(index='Platform', columns=col, values='Name', aggfunc='count').fillna(0)
            lda = LatentDirichletAllocation(n_components=n_compo)
            transformed = lda.fit_transform(pivot)
            transformed_df = pd.DataFrame(transformed, index=pivot.index)
            transformed_df.columns = [f"platform_{col}_lda_{i}" for i in range(n_compo)]
            return transformed_df.reset_index()

        ret = lda("Year_of_Release", 7)
        whole_df = whole_df.merge(ret, on="Platform", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        ret = lda("Genre", 7)
        whole_df = whole_df.merge(ret, on="Platform", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        ret = lda("Publisher", 20)
        whole_df = whole_df.merge(ret, on="Platform", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        ret = lda("Developer", 20)
        whole_df = whole_df.merge(ret, on="Platform", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        ret = lda("Rating", 7)
        whole_df = whole_df.merge(ret, on="Platform", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class RatingLDA(Feature):
    def create_features(self):
        global train, test

        whole_df = pd.concat([train, test], ignore_index=True)
        gen_cols = []

        # unknown -> nan
        whole_df["Publisher"] = whole_df["Publisher"].replace("unknown", np.nan)

        def lda(col, n_compo):
            from sklearn.decomposition import LatentDirichletAllocation
            pivot = whole_df.pivot_table(index='Rating', columns=col, values='Name', aggfunc='count').fillna(0)
            lda = LatentDirichletAllocation(n_components=n_compo)
            transformed = lda.fit_transform(pivot)
            transformed_df = pd.DataFrame(transformed, index=pivot.index)
            transformed_df.columns = [f"rating_{col}_lda_{i}" for i in range(n_compo)]
            return transformed_df.reset_index()

        ret = lda("Year_of_Release", 7)
        whole_df = whole_df.merge(ret, on="Rating", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        ret = lda("Genre", 7)
        whole_df = whole_df.merge(ret, on="Rating", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        ret = lda("Publisher", 20)
        whole_df = whole_df.merge(ret, on="Rating", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        ret = lda("Developer", 20)
        whole_df = whole_df.merge(ret, on="Rating", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        ret = lda("Platform", 7)
        whole_df = whole_df.merge(ret, on="Rating", how="left")
        gen_cols.extend([col for col in ret if "lda" in col])

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class NonEnglishWords(Feature):
    def create_features(self):
        global train, test

        import texthero as hero
        from texthero import preprocessing
        from nltk.corpus import words, brown
        import nltk
        nltk.download('brown')
        nltk.download('words')

        words_vocab = set(words.words())
        brown_vocab = set(brown.words())
        vocab = words_vocab.union(brown_vocab)

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        custom_pipeline = [preprocessing.fillna
                        , preprocessing.lowercase
                        , preprocessing.remove_digits
                        , preprocessing.remove_punctuation
                        , preprocessing.remove_diacritics
                        , preprocessing.remove_whitespace
                        , preprocessing.remove_stopwords
                        ]

        names = hero.clean(whole_df['Name'], custom_pipeline)
        pubs = hero.clean(whole_df['Publisher'], custom_pipeline)
        devs = hero.clean(whole_df['Developer'], custom_pipeline)

        def contains_non_english(name):
            # ratio of english words >= 50% ===> False
            ws = name.split()
            evals = [w in vocab for w in ws]
            if len(evals) == 0:
                return np.nan
            english_greater_or_equal_50 = sum(evals)/len(evals) >= 0.5
            return not english_greater_or_equal_50

        whole_df["name_non_english"] = names.map(contains_non_english).astype(float)
        whole_df["pub_non_english"] = pubs.map(contains_non_english).astype(float)
        whole_df["dev_non_english"] = devs.map(contains_non_english).astype(float)

        gen_cols = ["name_non_english", "pub_non_english", "dev_non_english"]

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class NonEnglishWords_2(Feature):
    def create_features(self):
        global train, test

        import texthero as hero
        from texthero import preprocessing
        from nltk.corpus import words, brown
        import nltk
        nltk.download('brown')
        nltk.download('words')

        words_vocab = set(words.words())
        brown_vocab = set(brown.words())
        vocab = words_vocab.union(brown_vocab)

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        custom_pipeline = [preprocessing.fillna
                        , preprocessing.lowercase
                        , preprocessing.remove_digits
                        , preprocessing.remove_punctuation
                        , preprocessing.remove_diacritics
                        , preprocessing.remove_whitespace
                        , preprocessing.remove_stopwords
                        ]

        names = hero.clean(whole_df['Name'], custom_pipeline)
        pubs = hero.clean(whole_df['Publisher'], custom_pipeline)
        # devs = hero.clean(whole_df['Developer'], custom_pipeline)

        def contains_non_english(name):
            # ratio of english words >= 50% ===> False
            ws = name.split()
            evals = [w in vocab for w in ws]
            if len(evals) == 0:
                return np.nan
            english_greater_or_equal_50 = sum(evals)/len(evals) >= 0.5
            return not english_greater_or_equal_50

        whole_df["name_non_english"] = names.map(contains_non_english)
        whole_df["pub_non_english"] = pubs.map(contains_non_english)
        whole_df["name_and_pub_non_english"] = (whole_df["name_non_english"] & whole_df["pub_non_english"]).astype(float)
        # whole_df["dev_non_english"] = devs.map(contains_non_english).astype(float)

        gen_cols = ["name_and_pub_non_english"]

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


class NonEnglishWords_3(Feature):
    def create_features(self):
        global train, test

        import texthero as hero
        from texthero import preprocessing
        from nltk.corpus import words, brown
        import nltk
        nltk.download('brown')
        nltk.download('words')

        words_vocab = set(words.words())
        brown_vocab = set(brown.words())
        vocab = words_vocab.union(brown_vocab)

        # gen_cols = []

        whole_df = pd.concat([train, test], ignore_index=True)

        custom_pipeline = [preprocessing.fillna
                        , preprocessing.lowercase
                        , preprocessing.remove_digits
                        , preprocessing.remove_punctuation
                        , preprocessing.remove_diacritics
                        , preprocessing.remove_whitespace
                        , preprocessing.remove_stopwords
                        ]

        names = hero.clean(whole_df['Name'], custom_pipeline)
        pubs = hero.clean(whole_df['Publisher'], custom_pipeline)
        # devs = hero.clean(whole_df['Developer'], custom_pipeline)

        def contains_non_english(name):
            # ratio of english words >= 50% ===> False
            ws = name.split()
            evals = [w in vocab for w in ws]
            if len(evals) == 0:
                return np.nan
            return not all(evals)

        whole_df["name_non_english"] = names.map(contains_non_english)
        whole_df["pub_non_english"] = pubs.map(contains_non_english)
        whole_df["name_and_pub_non_english"] = (whole_df["name_non_english"] & whole_df["pub_non_english"]).astype(float)
        # whole_df["dev_non_english"] = devs.map(contains_non_english).astype(float)

        gen_cols = ["name_and_pub_non_english"]

        train, test = whole_df[:len(train)], whole_df[len(train):]

        self.train = train[gen_cols]
        self.test = test[gen_cols]


if __name__ == '__main__':
    # TRAIN_PATH = Path("./data/raw/train_fixed.csv")
    # TEST_PATH = Path("./data/raw/test_fixed.csv")

    # exp039
    TRAIN_PATH = Path("./data/raw/train_fixed.csv")
    TEST_PATH = Path("./data/raw/test_year_predicted.csv")

    # exp040
    # TRAIN_PATH = Path("./data/raw/train_year_predicted.csv")
    # TEST_PATH = Path("./data/raw/test_year_predicted.csv")

    args = get_arguments()

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    FeatureGenerator(globals(), args.force, args.which).run()
