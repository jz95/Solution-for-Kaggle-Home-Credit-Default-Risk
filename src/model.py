from src.utils import *
from src.feature_extract import feature_extract
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import numpy as np
import pandas as pd


def run(name, mode, debug, **kwargs):
    config = read_config('./config/model_config.yaml')
    train_df = pd.read_csv(kwargs['train_path']
                           ) if kwargs['train_path'] else None
    test_df = pd.read_csv(kwargs['test_path']) if kwargs['test_path'] else None

    if mode == 'search':
        # param_search(df, )
        pass
    elif mode == 'all':
        if train_df is None or test_df is None:
            train_df, test_df = extract(name, debug, config['input'])
        models = train(train_df, name, config['kfold_setting'])
        predict(test_df, name, pred_model=models)
    elif mode == 'train':
        if train_df is None:
            train_df, _ = extract(name, debug, config['input'])
        train(train_df, name, config)
    elif mode == 'predict':
        if test_df is None:
            raise ValueError(
                'must specify the test_path when using predict mode.')
        predict(test_df, name)
    elif mode == 'extract':
        extract(name, debug, config['input'])
    else:
        raise NotImplementedError('unknown mode %s.' % mode)


def extract(name, debug, input_config):
    workspace = make_workspace(name)
    train_df, test_df = feature_extract(debug, input_config)
    train_df.to_csv(workspace + '/train_df.csv', index=False)
    test_df.to_csv(workspace + '/test_df.csv', index=False)
    return train_df, test_df


def predict(test_df, name, **kwargs):
    workspace = get_workspace(name)
    models = kwargs['pred_model'] if kwargs['pred_model'] else load_model(
        workspace)
    preds = lightgbm_pred(test_df, models)
    test_df.loc['TARGET', :] = preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(
        workspace + '/preds.csv', index=False)


def lightgbm_pred(test_df, models):
    feats = [f for f in test_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    preds = np.zeros(test_df.shape[0])
    num_folds = 1
    for clf in models.values():
        preds += clf.predict_proba(test_df[feats],
                                   num_iteration=clf.best_iteration_)[:, 1]
        num_folds += 1
    preds = preds / num_folds
    return preds


def train(train_df, name, config):
    workspace = make_workspace(name)
    models, feature_importance_df = \
        kfold_lightgbm(train_df, **config)
    save_model(models, workspace)
    display_importances(feature_importance_df,
                        workspace + '/feature_importance.png')
    train_df.to_csv(workspace + '/train_df.csv', index=False)
    feature_importance_df.to_csv(
        workspace + '/feature_importance.csv', index=False)
    return models


def kfold_lightgbm(train_df, num_folds, stratified, seed, model_param, **kwargs):
    print(model_param)
    seed = np.random.randint(2018) if seed is None else seed
    if stratified:
        folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])

    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    models = {}

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization

        clf = LGBMClassifier(
            nthread=4,
            # is_unbalance=True,
            # suggested 10000?
            n_estimators=100,
            learning_rate=0.01,
            num_leaves=32,
            colsample_bytree=0.2497036,
            subsample=0.8715623,
            max_depth=-1,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            # scale_pos_weight=11
        )

        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=200)

        clf_name = 'fold_%d/%d' % (n_fold + 1, num_folds)
        models[clf_name] = clf

        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_)[:, 1]

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

    return models, feature_importance_df


def display_importances(feature_importance_df_, filename):
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(by="importance", ascending=False)[:80].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(
        cols)]
    plt.figure(figsize=(12, 18))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(
        by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(filename)
