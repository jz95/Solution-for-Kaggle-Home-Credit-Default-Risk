from src.utils import *
from src.feature_extract import feature_extract
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
import gc
import numpy as np
import pandas as pd


def run(name, mode, debug, **kwargs):
    basic_config = read_config('./config/basic_config.yaml')
    workspace = WorkSpace(name, basic_config['workspace'])
    train_df = pd.read_csv(kwargs['train_path']
                           ) if kwargs['train_path'] else None
    test_df = pd.read_csv(kwargs['test_path']) if kwargs['test_path'] else None
    if mode == 'search':
        if train_df is None or test_df is None:
                train_df, _ = extract(workspace, debug, basic_config['input'])
        search_config = read_config('./config/param_search.yaml')
        parameter_search(train_df, workspace, debug, search_config)
    elif mode == 'extract':
        extract(workspace, debug, basic_config['input'])
    elif mode == 'predict':
        if test_df is None:
            raise ValueError(
                'must specify the test_path when using predict mode.')
        predict(test_df, workspace)
    # all & train mode
    else:
        model_config = read_config('./config/model_config.yaml')
        if mode == 'all':
            if train_df is None or test_df is None:
                train_df, test_df = extract(workspace, debug, basic_config['input'])
            models = train(train_df, workspace, debug, model_config)
            predict(test_df, workspace, pred_model=models)
        elif mode == 'train':
            if train_df is None:
                train_df, _ = extract(workspace, debug, basic_config['input'])
            train(train_df, workspace, debug, config)


def extract(workspace, debug, input_config):
    train_df, test_df = feature_extract(debug, input_config)
    workspace.save(train_df, 'train_df.csv')
    workspace.save(test_df, 'test_df.csv')
    return train_df, test_df


def predict(test_df, workspace, **kwargs):
    if 'pred_model' in kwargs and kwargs['pred_model'] is not None:
        models = kwargs['pred_model']
    else:
        models = workspace.load('kfold_lgb.pkl')
    preds = lightgbm_pred(test_df, models)
    test_df['TARGET'] = preds
    workspace.save(test_df[['SK_ID_CURR', 'TARGET']], 'preds.csv')


def lightgbm_pred(test_df, models):
    feats = [f for f in test_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    test_preds = np.zeros(test_df.shape[0])
    for clf in models.values():
        test_preds += clf.predict_proba(test_df[feats],
                                   num_iteration=clf.best_iteration_)[:, 1]
    num_folds = len(models)
    test_preds = test_preds / num_folds
    return test_preds


def train(train_df, workspace, debug, model_config):
    models, result = \
        kfold_lightgbm(train_df, debug=debug, **model_config)

    workspace.save(result, 'result.pkl')
    workspace.save(models, 'kfold_lgb.pkl')
    workspace.gen_report('kfold')
    return models


def kfold_lightgbm(train_df, debug, num_folds, stratified, seed, model_param):
    seed = np.random.randint(2018) if seed is None else seed
    if stratified:
        folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # out of fold results
    oof_preds = np.zeros(train_df.shape[0])

    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    models = {}
    training_result = {}
    training_result['feature'] = feats
    training_result['seed'] = seed
    training_result['stratified'] = stratified
    training_result['lgb_param'] = model_param
    training_result['kfold_result'] = {}
    kfold_result = training_result['kfold_result']

    # force reset the model_param in debug mode
    if debug:
        model_param['n_estimators'] = 100
        model_param['learning_rate'] = 0.3

    for n_fold, (train_idx, valid_idx) in enumerate(
            folds.split(train_df[feats], train_df['TARGET']), start=1):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization

        clf = LGBMClassifier(**model_param)
        t0 = time.time()
        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=200)
        t1 = time.time()
        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_)[:, 1]
        # collect fold result for analysis
        fold_result = {}
        fold_result['using_time'] = t1 - t0
        fold_result['best_iteration'] = clf.best_iteration_
        fold_result['feature_importance'] = clf.feature_importances_

        fold_result['best_score'] = {}
        fold_result['best_score']['training'] = clf.best_score_[
            'training']['auc']
        fold_result['best_score']['valid'] = clf.best_score_[
            'valid_1']['auc']

        fold_result['evals_result'] = {}
        fold_result['evals_result']['training'] = clf.evals_result_[
            'training']['auc']
        fold_result['evals_result']['valid'] = clf.evals_result_[
            'valid_1']['auc']

        print('Fold %2d AUC : %.6f' % (n_fold, fold_result['best_score']['valid']))

        kfold_result['fold_%s' % n_fold] = fold_result
        models[n_fold] = clf

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    full_auc = roc_auc_score(train_df['TARGET'], oof_preds)
    print('Full AUC score %.6f' % full_auc)
    training_result['full_auc'] = full_auc

    return models, training_result


def parameter_search(train_df, workspace, debug, search_config):
    searcher = grid_search(train_df, debug=debug, **search_config)
    workspace.save(searcher, 'grid_search.pkl')
    workspace.gen_report('grid_search')


def grid_search(train_df, debug, num_folds, stratified, seed, param_grid):
    seed = np.random.randint(2018) if seed is None else seed
    if stratified:
        folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    def check_param_grid(param_grid):
        fixed_params = {}
        for param in list(param_grid.keys()):
            val = param_grid[param]
            if not isinstance(val, list):
                fixed_params[param] = val
                del param_grid[param]
            if isinstance(val, list) and len(val) == 1:
                fixed_params[param] = val[0]
                del param_grid[param]

        return param_grid, fixed_params

    # param_grid would be passed to Grid Search
    # fixed_params would be passed to the base model
    param_grid, fixed_params = check_param_grid(param_grid)

    # force reset the model_param in debug mode
    if debug:
        fixed_params['n_estimators'] = 100
        fixed_params['learning_rate'] = 0.3

    lgbClf = LGBMClassifier(**fixed_params)
    searcher = GridSearchCV(estimator=lgbClf,
                            param_grid=param_grid,
                            cv=folds,
                            scoring='roc_auc',
                            n_jobs=-1,
                            refit=False,
                            verbose=5,
                            return_train_score=True)
    searcher.fit(train_df[feats], train_df['TARGET'])

    return searcher
