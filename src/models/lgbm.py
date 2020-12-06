# import json
import sys

from mcs_kfold import MCSKFold
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.append("../../")
from utils import get_logger, log_evaluation, top2accuracy, eval_func, load_datasets, track_experiment


# NUM_CLASS = 1


class Experiment:
    def __init__(self, exp_id, config, model_type):
        for k, v in config.items():
            setattr(self, k, v)
        self.exp_id = exp_id
        self.logger = get_logger(exp_id)
        self.model_type = model_type

    def load_data(self):
        train = pd.read_csv("./data/raw/train_fixed.csv")
        X_train, X_test = load_datasets(self.features)
        train["Global_Sales_log1p"] = np.log1p(train["Global_Sales"])
        y_train = train["Global_Sales_log1p"].to_frame()
        groups = train["Publisher"].to_frame()
        return X_train, X_test, y_train, groups

    def fit_and_predict(self, X_train, X_test, y_train, groups):
        if self.cv == "mcs":
            folds = MCSKFold(n_splits=5, shuffle_mc=True, max_iter=100)
        elif self.cv == "group":
            folds = GroupKFold(n_splits=5)

        oof = np.zeros(len(X_train))
        predictions = np.zeros(len(X_test))
        feature_importance_df = pd.DataFrame()
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, groups=groups)):
            self.logger.debug("-" * 100)
            self.logger.debug(f"Fold {fold+1}")
            train_data = lgb.Dataset(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
            val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train.iloc[val_idx])
            callbacks = [log_evaluation(self.logger, period=100)]
            clf = lgb.train(self.params, train_data, valid_sets=[train_data, val_data], verbose_eval=100, early_stopping_rounds=100, callbacks=callbacks)  #, feval=eval_func)
            oof[val_idx] = clf.predict(X_train.iloc[val_idx].values, num_iteration=clf.best_iteration)
            fold_score = mean_squared_log_error(np.expm1(y_train.iloc[val_idx].values), np.expm1(oof[val_idx])) ** .5
            fold_scores.append(fold_score)

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = X_train.columns.values
            fold_importance_df["importance"] = clf.feature_importance(importance_type="gain")
            fold_importance_df["fold"] = fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            predictions += np.expm1(clf.predict(X_test, num_iteration=clf.best_iteration)) / folds.n_splits

        feature_importance_df = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).head(50)
        self.logger.debug("##### feature importance #####")
        self.logger.debug(feature_importance_df)
        cv_score_fold_mean = sum(fold_scores) / len(fold_scores)
        self.logger.debug(f"cv_score_fold_mean: {cv_score_fold_mean}")
        return predictions, cv_score_fold_mean

    def save(self, predictions):
        spsbm = pd.read_csv("./data/raw/atmaCup8_sample-submission.csv")
        spsbm["Global_Sales"] = predictions
        spsbm.to_csv(f"./submissions/{self.exp_id}_sub.csv", index=False)

    def track(self, cv_score):
        # track_experiment(self.exp_id, "model", self.model_type)
        # track_experiment(self.exp_id, "features", self.features)
        # track_experiment(self.exp_id, "cv", self.cv)
        # track_experiment(self.exp_id, "params", self.params)
        track_experiment(self.exp_id, "cv_score", cv_score)

    def run(self):
        X_train, X_test, y_train, groups = self.load_data()
        predictions, cv_score = self.fit_and_predict(X_train, X_test, y_train, groups)
        self.save(predictions)
        self.track(cv_score)
