import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def agg_groupby(df, groupby_info):
    for spec in groupby_info:
        # Name of the aggregation we're applying
        agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']

        # Unique list of features to select
        all_features = list(set(spec['groupby'] + [spec['select']]))

        # Name of new feature
        new_feature = '{}_ON_{}_GBY_{}'.format(agg_name.upper(), spec['select'],
                                               '_'.join(spec['groupby']))

        # Perform the groupby
        gp = df[all_features].\
            groupby(spec['groupby'])[spec['select']].\
            agg(spec['agg']).\
            reset_index().\
            rename(index=str, columns={spec['select']: new_feature})
        # Merge back to df
        df = df.merge(gp, on=spec['groupby'], how='left')
    return df


def apply_groupby(df, groupby_info):
    for spec in groupby_info:
        # Name of the aggregation we're applying
        app_name = spec['app_name'] if 'app_name' in spec else spec['apply']

        # Unique list of features to select
        all_features = list(set(spec['groupby'] + [spec['select']]))

        # Name of new feature
        new_feature = '{}_ON_{}_GBY_{}'.format(app_name, spec['select'],
                                               '_'.join(spec['groupby']))

        # Perform the groupby
        gpSeries = df[all_features + ['SK_ID_CURR']].\
            groupby(spec['groupby'])[spec['select']].\
            apply(spec['apply']).rename(new_feature)

        df = df.join(gpSeries)
    return df


def extreme_eliminate(series):
    ret = series.copy()
    up_bnd = ret.quantile(.995)
    low_bnd = ret.quantile(.005)
    ret[series > up_bnd] = up_bnd
    ret[series < low_bnd] = low_bnd
    return ret


def box(df, cut_info):
    for infoDict in cut_info:
        col = infoDict['col']
        new_col_name = infoDict['new_col']
        n_bins = infoDict['bins']
        df[new_col_name] = pd.cut(
            extreme_eliminate(df[col]), bins=n_bins, labels=range(n_bins)).astype('float')
    return df

# One-hot encoding for categorical columns with get_dummies


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns,
                        dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv


def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv('../input/application_train.csv', nrows=num_rows)
    test_df = pd.read_csv('../input/application_test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & (
        'FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby(
        'ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    # AMT RATIO
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / \
        (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']
    df['NEW_GOODS_TO_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']

    # TIME RATIO
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / \
        (df['DAYS_BIRTH'] / 365.25)
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / \
        (df['DAYS_EMPLOYED'] / 365.35)
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_REGISTRATION_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / \
        df['DAYS_REGISTRATION']
    df['NEW_PHONE_TO_EMPLOY_RATIO_'] = df['DAYS_LAST_PHONE_CHANGE'] / \
        df['DAYS_EMPLOYED']
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_EMPLOY_TO_REGISTRATION_RATIO'] = df['DAYS_EMPLOYED'] / \
        df['DAYS_REGISTRATION']
    df['NEW_ID_PUB_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['NEW_REGISTRATION_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
    df['NEW_ID_PUB_TO_REGISTRATION_RATIO'] = df['DAYS_ID_PUBLISH'] / \
        df['DAYS_REGISTRATION']

    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_DOC_IND_SUM'] = df[docs].sum(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_PER_ADULT'] = df['AMT_INCOME_TOTAL'] / \
        (df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)

    df['NEW_EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * \
        2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4
    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian', 'std']:
        df['NEW_EXT_SOURCES_{}'.format(function_name.upper())] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    # OTHER
    df['NEW_SCORE1_TO_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / \
        (df['DAYS_BIRTH'] / 365.25)

    ##################
    # DO DISCRETIZAION
    ##################
    cut_info = [
        {'col': 'NEW_EXT_SOURCES_MEAN', 'new_col': 'EXT_SOURCE_MEAN_LEVEL', 'bins': 4},
        {'col': 'REGION_RATING_CLIENT',
            'new_col': 'REGION_RATING_CLIENT_LEVEL', 'bins': 4},
    ]
    df = box(df, cut_info)

    ################
    # DO AGGREGATION
    ################

    agg_gby_info = [
        {'groupby': ['CODE_GENDER', 'NAME_EDUCATION_TYPE'],
            'select': 'AMT_ANNUITY', 'agg': 'max'},
        {'groupby': ['CODE_GENDER', 'NAME_EDUCATION_TYPE'],
            'select': 'AMT_CREDIT', 'agg': 'mean'},
        {'groupby': ['CODE_GENDER', 'NAME_EDUCATION_TYPE'],
            'select': 'OWN_CAR_AGE', 'agg': 'max'},
        {'groupby': ['CODE_GENDER', 'NAME_EDUCATION_TYPE'],
            'select': 'NEW_EXT_SOURCES_MEAN', 'agg': 'mean'},

        {'groupby': ['CODE_GENDER', 'ORGANIZATION_TYPE'],
            'select': 'AMT_ANNUITY', 'agg': 'max'},
        {'groupby': ['CODE_GENDER', 'ORGANIZATION_TYPE'],
            'select': 'AMT_CREDIT', 'agg': 'mean'},
        {'groupby': ['CODE_GENDER', 'ORGANIZATION_TYPE'],
            'select': 'OWN_CAR_AGE', 'agg': 'max'},
        {'groupby': ['CODE_GENDER', 'ORGANIZATION_TYPE'],
            'select': 'NEW_EXT_SOURCES_MEAN', 'agg': 'mean'},
        {'groupby': ['CODE_GENDER', 'ORGANIZATION_TYPE'],
            'select': 'AMT_INCOME_TOTAL', 'agg': 'mean'},

        {'groupby': ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'],
            'select': 'AMT_ANNUITY', 'agg': 'max'},
        {'groupby': ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'],
            'select': 'CNT_CHILDREN', 'agg': 'mean'},
        {'groupby': ['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'],
            'select': 'DAYS_ID_PUBLISH', 'agg': 'mean'},

        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE',
                     'REG_CITY_NOT_WORK_CITY'], 'select': 'ELEVATORS_AVG', 'agg': 'mean'},
        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE',
                     'REG_CITY_NOT_WORK_CITY'], 'select': 'CNT_CHILDREN', 'agg': 'mean'},
        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE',
                     'REG_CITY_NOT_WORK_CITY'], 'select': 'NEW_EXT_SOURCES_MEAN', 'agg': 'mean'},

        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
            'select': 'AMT_CREDIT', 'agg': 'mean'},
        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
            'select': 'AMT_REQ_CREDIT_BUREAU_YEAR', 'agg': 'mean'},
        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
            'select': 'NEW_EXT_SOURCES_MEAN', 'agg': 'mean'},
        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
            'select': 'BASEMENTAREA_AVG', 'agg': 'mean'},
        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
            'select': 'YEARS_BUILD_AVG', 'agg': 'mean'},
        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
            'select': 'OWN_CAR_AGE', 'agg': 'mean'},
        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
            'select': 'NONLIVINGAREA_AVG', 'agg': 'mean'},
        {'groupby': ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'],
            'select': 'APARTMENTS_AVG', 'agg': 'mean'},
    ]

    # df = agg_groupby(df, agg_gby_info)

    ################
    # DO APPLY
    ################
    def rankRatio(x):
        return x / (x.max() - x.min())

    app_gby_info = [
        # SOME RANK RATIO
        {'groupby': ['ORGANIZATION_TYPE'], 'select': 'DAYS_EMPLOYED',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['ORGANIZATION_TYPE'], 'select': 'AMT_INCOME_TOTAL',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['ORGANIZATION_TYPE'], 'select': 'AMT_CREDIT',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['ORGANIZATION_TYPE'], 'select': 'NEW_CREDIT_TO_INCOME_RATIO',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},

        {'groupby': ['OCCUPATION_TYPE'], 'select': 'DAYS_EMPLOYED',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['OCCUPATION_TYPE'], 'select': 'AMT_INCOME_TOTAL',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['OCCUPATION_TYPE'], 'select': 'AMT_CREDIT',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['OCCUPATION_TYPE'], 'select': 'NEW_CREDIT_TO_INCOME_RATIO',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},

        {'groupby': ['NAME_EDUCATION_TYPE'], 'select': 'DAYS_EMPLOYED',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['NAME_EDUCATION_TYPE'], 'select': 'AMT_INCOME_TOTAL',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['NAME_EDUCATION_TYPE'], 'select': 'AMT_CREDIT',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['NAME_EDUCATION_TYPE'], 'select': 'NEW_CREDIT_TO_INCOME_RATIO',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},

        {'groupby': ['NAME_INCOME_TYPE'], 'select': 'DAYS_EMPLOYED',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['NAME_INCOME_TYPE'], 'select': 'AMT_INCOME_TOTAL',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['NAME_INCOME_TYPE'], 'select': 'AMT_CREDIT',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['NAME_INCOME_TYPE'], 'select': 'NEW_CREDIT_TO_INCOME_RATIO',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},

        {'groupby': ['REGION_RATING_CLIENT_LEVEL'], 'select': 'DAYS_EMPLOYED',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['REGION_RATING_CLIENT_LEVEL'], 'select': 'AMT_INCOME_TOTAL',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['REGION_RATING_CLIENT_LEVEL'], 'select': 'AMT_CREDIT',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['REGION_RATING_CLIENT_LEVEL'], 'select': 'NEW_CREDIT_TO_INCOME_RATIO',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},

        {'groupby': ['EXT_SOURCE_MEAN_LEVEL'], 'select': 'DAYS_EMPLOYED',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['EXT_SOURCE_MEAN_LEVEL'], 'select': 'AMT_INCOME_TOTAL',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['EXT_SOURCE_MEAN_LEVEL'], 'select': 'AMT_CREDIT',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
        {'groupby': ['EXT_SOURCE_MEAN_LEVEL'], 'select': 'NEW_CREDIT_TO_INCOME_RATIO',
            'apply': rankRatio, 'app_name': 'RANK_RATIO'},
    ]

    # df = apply_groupby(df, app_gby_info)

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4',
                 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    df = df.drop(dropcolum, axis=1)

    del test_df
    gc.collect()
    return df
# Preprocess bureau.csv and bureau_balance.csv


def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('../input/bureau.csv', nrows=num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean', 'sum']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper()
                               for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    del bb, bb_agg
    gc.collect()

    # SOME OPERATION ON BUREAU
    bureau['NEW_OVERDUE_TO_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / \
        bureau['AMT_CREDIT_SUM']
    bureau['NEW_MAX_OVERDUE_TO_CREDIT_RATIO'] = bureau['AMT_CREDIT_MAX_OVERDUE'] / \
        bureau['AMT_CREDIT_SUM']
    bureau['NEW_DEBT_TO_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / \
        bureau['AMT_CREDIT_SUM']

    bureau['NEW_ENDDATE_TO_DAYS_CREDIT_RATIO'] = bureau['DAYS_CREDIT_ENDDATE'] / \
        bureau['DAYS_CREDIT']

    ## LOAN TYPE DIVERSITY ----> NEW_TYPE_DIVERSE
    grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].nunique(
    ).reset_index().rename(index=str, columns={'CREDIT_TYPE': 'NEW_LOAN_TYPE_CNT'})
    bureau = bureau.merge(grp, on=['SK_ID_CURR'], how='left')

    grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].size(
    ).reset_index().rename(index=str, columns={'CREDIT_TYPE': 'NEW_TOTAL_LOANS_CNT'})
    bureau = bureau.merge(grp, on=['SK_ID_CURR'], how='left')

    bureau['NEW_TYPE_DIVERSE'] = bureau['NEW_LOAN_TYPE_CNT'] / bureau['NEW_TOTAL_LOANS_CNT'] 

    ## LOAN FREQUENCY --->> NEW_DAYS_DIFF
    grp = bureau[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])
    grp = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending = False)).reset_index(drop = True)

    # Calculate Difference between the number of Days 
    grp['DAYS_CREDIT1'] = grp['DAYS_CREDIT']*-1
    grp['NEW_DAYS_DIFF'] = grp.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
    bureau = bureau.merge(grp[['SK_ID_BUREAU', 'NEW_DAYS_DIFF']], on=['SK_ID_BUREAU'], how='left')

    bureau['NEW_IS_POS_END_DATE'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype('object')
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean'],
        'NEW_DAYS_DIFF': ['mean', 'min', 'max'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        {**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(
        ['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(
        ['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(
        ['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, 

    # Bureau: future credits - using only numerical aggregations
    future = bureau[bureau['NEW_IS_POS_END_DATE_True'] == 1]
    future_agg = future.groupby('SK_ID_CURR').agg(num_aggregations)
    future_agg.columns = pd.Index(
        ['FUTURE_' + e[0] + "_" + e[1].upper() for e in future_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(future_agg, how='left', on='SK_ID_CURR')
    del future, future_agg, bureau, grp

    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv


def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv('../input/previous_application.csv', nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'APP_CREDIT_PERC': ['max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg(
        {**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(
        ['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(
        ['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(
        ['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv


def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(
        ['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv


def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('../input/installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['mean', 'var'],
        'PAYMENT_DIFF': ['mean', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(
        ['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv


def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('../input/credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper()
                               for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code


def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(
        train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in [
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            # is_unbalance=True,
            # suggested 10000?
            n_estimators=2800,
            learning_rate=0.015,
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            # scale_pos_weight=11
        )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[
            :, 1] / folds.n_splits

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
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(
            submission_file_name, index=False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(
        cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(
        by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances0806.png')


def main(debug=False):
    num_rows = 10000 if debug else None
    # df = application_train_test(num_rows)
    df = pd.read_csv('../input/application_train.csv', nrows=num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    # with timer("Run LightGBM with kfold"):
    #     feat_importance = kfold_lightgbm(
    #         df, num_folds=5, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission_0806_with_2500.csv"
    with timer("Full model run"):
        main(True)
