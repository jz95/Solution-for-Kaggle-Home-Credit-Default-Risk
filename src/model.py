from src.utils import *
from src.feature_extract import feature_extract
from src.HomeCreditClassifier import *
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd


def run(name, mode, debug, type_, **kwargs):
    basic_config = read_config('./config/basic_config.yaml')
    workspace = WorkSpace(name, basic_config['workspace'])

    train_df = pd.read_csv(kwargs['train_path']
                           ) if kwargs['train_path'] else None
    test_df = pd.read_csv(kwargs['test_path']) if kwargs['test_path'] else None

    if mode == 'search':
        if train_df is None:
            train_df, _ = runExtract(workspace, debug, basic_config['input'])
        search_config = read_config('./config/param_search.yaml')
        runSearch(train_df, workspace, debug, type_, search_config)
    elif mode == 'extract':
        runExtract(workspace, debug, basic_config['input'])
    elif mode == 'predict':
        if test_df is None:
            raise ValueError(
                'must specify the test_path when using predict mode.')
        runPredict(test_df, workspace)
    elif mode == 'stacking':
        stacking_config = read_config('./config/stacking_config.yaml')
        model = runStacking(train_df, workspace, debug, stacking_config)
        runPredict(test_df, workspace, pred_model=model)
        workspace.save(model, 'stacking.pkl')
        workspace.gen_report('stacking')
    # all & train mode
    else:
        model_config = read_config('./config/model_config.yaml')
        if mode == 'all':
            if train_df is None or test_df is None:
                train_df, test_df = runExtract(
                    workspace, debug, basic_config['input'])
            models = runTrain(train_df, workspace, debug, type_, model_config)
            runPredict(test_df, workspace, pred_model=models)
        elif mode == 'train':
            if train_df is None:
                train_df, _ = runExtract(
                    workspace, debug, basic_config['input'])
            runTrain(train_df, workspace, debug, type_, model_config)


def runExtract(workspace, debug, input_config):
    train_df, test_df = feature_extract(debug, input_config)
    workspace.save(train_df, 'train_df.csv')
    workspace.save(test_df, 'test_df.csv')
    return train_df, test_df


def runStacking(train_df, workspace, debug, stacking_config):
    model = stacking(train_df, debug, stacking_config)
    return model


def stacking(train_df, debug, stacking_config):
    # construct cv for stacking model
    cv = gen_cv(**stacking_config['stacking_setting'])

    # construct base classifiers
    base_lgbClfs = {}
    for clf_setting in stacking_config['base_classifier']:
        clf_name = clf_setting['name']
        model_param = clf_setting['model_param']
        # force reset the model_param in debug mode
        if debug:
            model_param['n_estimators'] = 10
            model_param['learning_rate'] = 0.3
        lgb = LGBMClassifier(**model_param)
        sp_rate, sp_seed = clf_setting['feat_sample'], model_param['random_state']
        base_lgbClfs[clf_name] = KFoldClassifier(lgb, cv, sp_rate, sp_seed)

    # construct meta classifier
    meta_param = stacking_config['meta_classifier']['model_param']
    # all models in stacking would use the same cv obj
    meta_clf = LogisticRegressionCV(
        refit=True, scoring='roc_auc', cv=cv, verbose=10, **meta_param)

    # construct stacking model
    stClf = StackingClassifier(base_lgbClfs, meta_clf)
    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    stClf.fit(train_df[feats], train_df['TARGET'])
    return stClf


def runPredict(test_df, workspace, **kwargs):
    if 'pred_model' in kwargs and kwargs['pred_model'] is not None:
        model = kwargs['pred_model']
    else:
        model = workspace.load_model()
    preds = make_pred(test_df, model)
    test_df['TARGET'] = preds
    workspace.save(test_df[['SK_ID_CURR', 'TARGET']], 'preds.csv')


def make_pred(test_df, model):
    feats = [f for f in test_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    return model.predict_proba(test_df[feats])


def runTrain(train_df, workspace, debug, type_, model_config):
    kfold_setting = model_config['kfold_setting']
    is_single = kfold_setting['num_folds'] < 1

    weights = read_weights(model_config['weights_file'])
    if type_ == 'lgb':
        model_param = model_config['lgb_model_param']
        constructor = LGBMClassifier
    elif type_ == 'lgb-rf':
        model_param = model_config['lgb_rf_model_param']
        model_param['boosting_type'] = 'rf'
        constructor = LGBMClassifier
    elif type_ == 'xgb':
        model_param = model_config['xgb_model_param']
        constructor = XGBClassifier

    if debug:
        model_param['n_estimators'] = 100
        model_param['learning_rate'] = 0.3
        weights = weights[: len(train_df)]

    clf = constructor(**model_param)

    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    X, y = train_df[feats], train_df['TARGET']

    if is_single:
        print("INFO: num_folds is less than 1, SINGLE MODEL would be trained.")
        model = clf
        with timer('Train Single %s Model' % type_):
            clf.fit(X, y, eval_set=[(X, y)], eval_metric='auc', verbose=100, sample_weight=weights)
        workspace.save(model, 'single_model.pkl')
    else:
        cv = gen_cv(**kfold_setting)
        model = KFoldClassifier(clf, cv)
        with timer('Train KFold %s Model' % type_):
            model.fit(X, y, sample_weight=weights)
        workspace.save(model, 'kfold_model.pkl')
        workspace.gen_report('kfold')
    return model


def runSearch(train_df, workspace, debug, type_, search_config):
    if type_ == 'lgb':
        param_grid = search_config['lgb_param_grid']
        constructor = LGBMClassifier
    elif type_ == 'lgb-rf':
        param_grid = search_config['lgb_rf_param_grid']
        param_grid['boosting_type'] = 'rf'
        constructor = LGBMClassifier
    elif type_ == 'xgb':
        param_grid = search_config['xgb_param_grid']
        constructor = XGBClassifier

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

    clf = constructor(**fixed_params)

    # prepare training data
    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    X, y = train_df[feats], train_df['TARGET']

    # construct cv
    kfold_setting = search_config['kfold_setting']
    cv = gen_cv(**kfold_setting)

    # construct Grid searcher
    searcher = GridSearchCV(estimator=clf,
                            param_grid=param_grid,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1,
                            refit=False,
                            verbose=5,
                            return_train_score=True)
    searcher.fit(X, y)

    # save result
    workspace.save(searcher, 'grid_search.pkl')
    workspace.gen_report('grid_search')
