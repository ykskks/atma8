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
        test_te = test_series.apply(lambda row: train_agg_df.loc[row]["mean"] if row != "NaN" else np.nan)
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


if __name__ == '__main__':
    TRAIN_PATH = Path("./data/raw/train_fixed.csv")
    TEST_PATH = Path("./data/raw/test_fixed.csv")

    args = get_arguments()

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    FeatureGenerator(globals(), args.force, args.which).run()
