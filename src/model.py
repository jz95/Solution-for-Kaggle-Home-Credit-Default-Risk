from src.utils import *
from src.feature_extract import feature_extract
from src.HomeCreditClassifier import *
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
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
    elif mode == 'stacking':
        stacking_config = read_config('./config/stacking_config.yaml')
        stacking(train_df, workspace, debug, stacking_config)
    # all & train mode
    else:
        model_config = read_config('./config/model_config.yaml')
        if mode == 'all':
            if train_df is None or test_df is None:
                train_df, test_df = extract(
                    workspace, debug, basic_config['input'])
            models = train(train_df, workspace, debug, model_config)
            predict(test_df, workspace, pred_model=models)
        elif mode == 'train':
            if train_df is None:
                train_df, _ = extract(workspace, debug, basic_config['input'])
            train(train_df, workspace, debug, config)


def stacking(train_df, workspace, debug, stacking_config):
    # construct base classifiers
    base_lgbClfs = {}
    for clf_setting in stacking_config['base_classifier']:
        clf_name = clf_setting['name']
        model_param = clf_setting['model_param']
        # force reset the model_param in debug mode
        if debug:
            model_param['n_estimators'] = 100
            model_param['learning_rate'] = 0.3
        base_lgbClfs[clf_name] = LGBMClassifier(**model_param)

    # construct meta classifier
    meta_param = stacking_config['meta_classifier']['model_param']
    meta_clf = LogisticRegressionCV(refit=True, scoring='roc_auc', verbose=10, **meta_param)

    # construct cv for stacking model
    cv = gen_cv(**stacking_config['stacking_setting'])

    # construct stacking model
    stClf = StackingClassifier(base_lgbClfs, meta_clf, cv)
    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    stClf.fit(train_df[feats], train_df['TARGET'])
    workspace.save(stClf, 'stacking.pkl')


def extract(workspace, debug, input_config):
    train_df, test_df = feature_extract(debug, input_config)
    workspace.save(train_df, 'train_df.csv')
    workspace.save(test_df, 'test_df.csv')
    return train_df, test_df


def predict(test_df, workspace, **kwargs):
    if 'pred_model' in kwargs and kwargs['pred_model'] is not None:
        model = kwargs['pred_model']
    else:
        model = workspace.load_model()
    preds = lightgbm_pred(test_df, model)
    test_df['TARGET'] = preds
    workspace.save(test_df[['SK_ID_CURR', 'TARGET']], 'preds.csv')


def lightgbm_pred(test_df, model):
    feats = [f for f in test_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    return model.predict_proba(test_df[feats])


def train(train_df, workspace, debug, model_config):
    is_single = model_config['kfold_setting']['num_folds'] < 1
    if is_single:
        print("INFO: num_folds is less than 1, SINGLE MODEL would be trained.")
        model = single_lightgbm(train_df, debug=debug, **model_config)
        workspace.save(model, 'single_lgb.pkl')
    else:
        model = kfold_lightgbm(train_df, debug=debug, **model_config)
        workspace.save(model, 'kfold_lgb.pkl')
        workspace.gen_report('kfold')
    return model


def kfold_lightgbm(train_df, debug, kfold_setting, model_param):
    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    if debug:
        model_param['n_estimators'] = 100
        model_param['learning_rate'] = 0.3

    lgbClf = LGBMClassifier(**model_param)
    cv = gen_cv(**kfold_setting)
    model = KFoldClassifier(lgbClf, cv)

    with timer('Train KFold lightgbm'):
        model.fit(train_df[feats], train_df['TARGET'])
    return model


def single_lightgbm(train_df, debug, model_param, **kwargs):
    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    train_x, train_y = train_df[feats], train_df['TARGET']
    # force reset the model_param in debug mode
    if debug:
        model_param['n_estimators'] = 100
        model_param['learning_rate'] = 0.3

    model = LGBMClassifier(**model_param)
    with timer('Train single lightgbm'):
        model.fit(train_x, train_y,
                  eval_set=[(train_x, train_y)],
                  eval_metric='auc', verbose=100)

    return model


def parameter_search(train_df, workspace, debug, search_config):
    searcher = grid_search(train_df, debug=debug, **search_config)
    workspace.save(searcher, 'grid_search.pkl')
    workspace.gen_report('grid_search')


def grid_search(train_df, debug, kfold_setting, param_grid):
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

    # log info
    print("fixed_params:")
    for key, val in fixed_params.items():
        print("%-20s = %s" % (key, val))

    print("seach_params:")
    for key, val in param_grid.items():
        print("%-20s = %s" % (key, val))

    lgbClf = LGBMClassifier(**fixed_params)

    # construct cv
    cv = gen_cv(**kfold_setting)

    # construct Grid searcher
    searcher = GridSearchCV(estimator=lgbClf,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1,
                            refit=False,
                            verbose=5,
                            return_train_score=True)
    searcher.fit(train_df[feats], train_df['TARGET'])

    return searcher
