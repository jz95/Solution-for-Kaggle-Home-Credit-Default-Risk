import numpy as np
import pandas as pd
import gc
from src.utils import timer
from src.feature_selection import DROP_FEATS
import warnings
warnings.filterwarnings(action='ignore')


def one_hot_encoding(df, nan_as_category):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns,
                        dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def batch_box(df, cols):
    level_cols = []
    for col in cols:
        temp = df[col]
        upBnd = temp.quantile(0.99)
        downBnd = temp.quantile(0.01)
        temp[df[col] > upBnd] = upBnd
        temp[df[col] < downBnd] = downBnd
        new_col_name = col + '_LEVEL'
        level_cols.append(new_col_name)
        df[new_col_name] = pd.cut(temp, bins=4)
    return df, level_cols


def batch_agg(df, by_cols, on_cols, prefix=''):
    agg_info = []
    for by_col in by_cols:
        info = dict(by=list(by_col), on=on_cols, agg=['mean', 'std'])
        agg_info.append(info)

    for info in agg_info:
        by_cols = info['by']
        on_cols = info['on']
        gby = df.groupby(by_cols)[on_cols]
        for agg_method in info['agg']:
            new_col_names = [
                "{}{}_ON_{}_BY_{}".
                format(prefix,
                       agg_method,
                       on_col,
                       '_N_'.join(by_cols))
                .upper() for on_col in on_cols
            ]
            nameMap = dict(zip(on_cols, new_col_names))
            temp = gby.agg(agg_method).rename(columns=nameMap).reset_index()
            df = df.merge(temp, on=by_cols, how='left')
    return df


def application_train_test(train_df, test_df, nan_as_category=False):
    # merge the train and test DataFrame
    df = train_df.append(test_df).reset_index()

    del train_df, test_df
    gc.collect()

    # data cleaning
    df = df[df['CODE_GENDER'] != 'XNA']

    social_cols = [col for col in df.columns if 'SOCIAL' in col]
    df.loc[df['DEF_30_CNT_SOCIAL_CIRCLE'] == 34, social_cols] = np.nan

    df['REGION_RATING_CLIENT_W_CITY'].replace(-1, np.nan, inplace=True)
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    # AMT RATIO
    df['APP_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['APP_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['APP_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['APP_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['APP_ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']
    df['APP_GOODS_TO_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']

    # TIME RATIO
    df['APP_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH'] / 365.25)
    df['APP_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED'] / 365.35)

    days_cols = [col for col in df.columns if 'DAYS_' in col]
    for i in range(len(days_cols)):
        for j in range(i + 1, len(days_cols)):
            col1, col2 = days_cols[i], days_cols[j]
            new_col_name = "APP_{}_TO_{}_RATIO".format(col1[5:], col2[5:])
            df[new_col_name] = df[col1] / df[col2]

    df['APP_INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['APP_INCOME_PER_ADULT'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN'])

    # EXT_SCORES
    for function_name in ['mean', 'nanmedian', 'std']:
        df['APP_EXT_SOURCES_{}'.format(function_name.upper())] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    df['APP_SCORE1_TO_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_BIRTH'] / 365.25)
    df['APP_SCORE1_TO_EMPLOY_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_EMPLOYED'] / 365.25)
    df['APP_SCORE1_TO_FAM_CNT_RATIO'] = df['EXT_SOURCE_1'] / df['CNT_FAM_MEMBERS']
    df['APP_SCORE1_TO_GOODS_RATIO'] = df['EXT_SOURCE_1'] / df['AMT_GOODS_PRICE']
    df['APP_SCORE1_TO_CREDIT_RATIO'] = df['EXT_SOURCE_1'] / df['AMT_CREDIT']
    df['APP_SCORE1_TO_SCORE2_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_2']
    df['APP_SCORE1_TO_SCORE3_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_3']

    df['APP_SCORE2_TO_CREDIT_RATIO'] = df['EXT_SOURCE_2'] / df['AMT_CREDIT']
    df['APP_SCORE2_TO_REGION_RATING_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_RATING_CLIENT']
    df['APP_SCORE2_TO_CITY_RATING_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_RATING_CLIENT_W_CITY']
    df['APP_SCORE2_TO_POP_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_POPULATION_RELATIVE']
    df['APP_SCORE2_TO_PHONE_CHANGE_RATIO'] = df['EXT_SOURCE_2'] / df['DAYS_LAST_PHONE_CHANGE']

    df['APP_SCORE3_TO_BIRTH_RATIO'] = df['EXT_SOURCE_3'] / (df['DAYS_BIRTH'] / 365.25)

    # DOCUMENT
    doc_cols = [col for col in df.columns if '_DOCUMENT_' in col]
    df['APP_ALL_DOC_CNT'] = df[doc_cols].sum(axis=1)
    df['APP_DOC3_6_8_CNT'] = df['FLAG_DOCUMENT_3'] + df['FLAG_DOCUMENT_6'] + df['FLAG_DOCUMENT_8']
    df['APP_DOC3_6_8_CNT_RAIO'] = df['APP_DOC3_6_8_CNT'] / df['APP_ALL_DOC_CNT']
    df.drop(columns=doc_cols, inplace=True)

    # HOUSING FEATURES
    housing_info_cols = [col for col in df.columns if df[col].dtype != 'object' and ('_AVG' in col or '_MODE' in col or '_MEDI' in col)]
    housing_info_cols.sort()
    i, j = 0, 1
    while i < len(housing_info_cols) and j < len(housing_info_cols):
        col = housing_info_cols[i]
        k = col.rfind('_')
        col_ = housing_info_cols[j]
        while j < (len(housing_info_cols) - 1) and col[0:k] == col_[0:k]:
            j += 1
            col_ = housing_info_cols[j]
        new_col_name = 'APP_{}_NEW_HOUSING_SCORE'.format(col[0:k])
        df[new_col_name] = df[housing_info_cols[i: j]].mean(axis=1)
        i = j
        j = i + 1

    df.drop(columns=housing_info_cols, inplace=True)

    # OTHER
    df['APP_CAR_PLUS_REALTY'] = df['FLAG_OWN_REALTY'] + df['FLAG_OWN_CAR']
    df['APP_CHILD_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    df['APP_REGION_RATING_TO_POP_RATIO'] = df['REGION_RATING_CLIENT'] / df['REGION_POPULATION_RELATIVE']
    df['APP_CITY_RATING_TO_POP_RATIO'] = df['REGION_RATING_CLIENT_W_CITY'] / df['REGION_POPULATION_RELATIVE']

    df['APP_REGION_ALL_NOT_EQ'] = df['REG_REGION_NOT_LIVE_REGION'] + df['REG_REGION_NOT_WORK_REGION'] + df['LIVE_REGION_NOT_WORK_REGION']
    df['APP_CITY_ALL_NOT_EQ'] = df['REG_CITY_NOT_LIVE_CITY'] + df['REG_CITY_NOT_WORK_CITY'] + df['LIVE_CITY_NOT_WORK_CITY']

    df['APP_CNT_CHILD_TO_BIRTH_RATIO'] = df['CNT_CHILDREN'] / df['DAYS_BIRTH']
    df['APP_CNT_CHILD_TO_REG_RATIO'] = df['CNT_CHILDREN'] / df['DAYS_REGISTRATION']

    df['APP_CREDIT_TO_ANNUITY_RATIO_DIV_SCORE1_TO_BIRTH_RATIO'] = df['APP_CREDIT_TO_ANNUITY_RATIO'] / df['APP_SCORE1_TO_BIRTH_RATIO']
    df['APP_CREDIT_TO_ANNUITY_RATIO_DIV_DAYS_BIRTH'] = df['APP_CREDIT_TO_ANNUITY_RATIO'] / (df['DAYS_BIRTH'] / 365.25)
    df['APP_CREDIT_TO_DAYS_BIRTH_RATIO'] = df['AMT_CREDIT'] / (df['DAYS_BIRTH'] / 365.25)
    df['APP_INCOME_DIV_BIRTH_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['APP_BIRTH_TO_EMPLOYED_RATIO']
    df['APP_ANNUITY_TO_INCOME_RATIO_DIV_BIRTH_TO_EMPLOYED_RATIO'] = df['APP_ANNUITY_TO_INCOME_RATIO'] / df['APP_BIRTH_TO_EMPLOYED_RATIO']

    # CONTACT INFO
    contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
    df['APP_CONTACT_ALL_CNT'] = df[contact_cols].sum(axis=1)

    # SOCIAL CIRCLE
    df['APP_DEF_TO_OBS_RATIO_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE']
    df['APP_DEF_TO_OBS_RATIO_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE']
    df['APP_60_TO_30_RATIO_OBS'] = df['OBS_60_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE']
    df['APP_60_TO_30_RATIO_DEF'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['DEF_30_CNT_SOCIAL_CIRCLE']

    df['APP_DEF_SUB_OBS_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] - df['OBS_30_CNT_SOCIAL_CIRCLE']
    df['APP_DEF_SUB_OBS_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] - df['OBS_60_CNT_SOCIAL_CIRCLE']
    df['APP_60_SUB_30_OBS'] = df['OBS_60_CNT_SOCIAL_CIRCLE'] - df['OBS_30_CNT_SOCIAL_CIRCLE']
    df['APP_60_SUB_30_DEF'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] - df['DEF_30_CNT_SOCIAL_CIRCLE']

    # REQ AMT
    df['APP_REQ_WEEK_GT_MON'] = (df['AMT_REQ_CREDIT_BUREAU_WEEK'] > df['AMT_REQ_CREDIT_BUREAU_YEAR']).astype('float')
    df['APP_REQ_MON_GT_QRT'] = (df['AMT_REQ_CREDIT_BUREAU_MON'] > df['AMT_REQ_CREDIT_BUREAU_YEAR']).astype('float')
    df['APP_REQ_QRT_GT_YEAR'] = (df['AMT_REQ_CREDIT_BUREAU_QRT'] > df['AMT_REQ_CREDIT_BUREAU_YEAR']).astype('float')
    df['APP_REQ_CNT_TOTAL'] = df['APP_REQ_WEEK_GT_MON'] + df['APP_REQ_MON_GT_QRT'] + df['APP_REQ_QRT_GT_YEAR']

    df['APP_CUM_REQ_DAY'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] + df['AMT_REQ_CREDIT_BUREAU_DAY']
    df['APP_CUM_REQ_WEEK'] = df['APP_CUM_REQ_DAY'] + df['AMT_REQ_CREDIT_BUREAU_WEEK']
    df['APP_CUM_REQ_MON'] = df['APP_CUM_REQ_WEEK'] + df['AMT_REQ_CREDIT_BUREAU_MON']
    df['APP_CUM_REQ_QRT'] = df['APP_CUM_REQ_MON'] + df['AMT_REQ_CREDIT_BUREAU_QRT']
    df['APP_CUM_REQ_YEAR'] = df['APP_CUM_REQ_QRT'] + df['AMT_REQ_CREDIT_BUREAU_YEAR']

    df['APP_CUM_REQ_WEEK_TO_QRT_RATIO'] = df['APP_CUM_REQ_WEEK'] / df['APP_CUM_REQ_QRT']
    df['APP_CUM_REQ_WEEK_TO_YEAR_RATIO'] = df['APP_CUM_REQ_WEEK'] / df['APP_CUM_REQ_YEAR']
    df['APP_CUM_REQ_MON_TO_QRT_RATIO'] = df['APP_CUM_REQ_MON'] / df['APP_CUM_REQ_QRT']
    df['APP_CUM_REQ_MON_TO_YEAR_RATIO'] = df['APP_CUM_REQ_MON'] / df['APP_CUM_REQ_YEAR']

    # DO SOME AGGREGATION
    by_cols = [
        ('CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE', 'WALLSMATERIAL_MODE', 'HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE', 'FONDKAPREMONT_MODE'),
        ('CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE'),
        ('CODE_GENDER', 'NAME_TYPE_SUITE', 'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN'),
        ('CODE_GENDER', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE'),
        ]
    on_cols = ['APP_CREDIT_TO_ANNUITY_RATIO', 'DAYS_BIRTH', 'APP_SCORE1_TO_BIRTH_RATIO', 'APP_ANNUITY_TO_INCOME_RATIO', 'AMT_INCOME_TOTAL']

    df = batch_agg(df, by_cols, on_cols, prefix='AGG_APP_')
    # df, _ = one_hot_encoding(df, nan_as_category)
    # df.drop(columns=level_cols, inplace=True)
    return df


def bureau_and_balance(bureau, bb, nan_as_category=True):
    # correct anomalies
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] < -20000, 'DAYS_CREDIT_ENDDATE'] = np.nan
    bureau.loc[bureau['DAYS_ENDDATE_FACT'] < -20000, 'DAYS_ENDDATE_FACT'] = np.nan
    bureau.loc[bureau['DAYS_CREDIT_UPDATE'] < -20000, 'DAYS_ENDDATE_FACT'] = np.nan

    #####################
    ### process on bb ###
    #####################
    # manual feature - has status changed over the records?
    def see_change(x):
        return x.nunique() > 1

    bb_agg = bb.groupby('SK_ID_BUREAU')[['STATUS']].\
                agg(see_change).rename(columns={'STATUS': 'STATUS_CHANGE'}).astype('float')

    for i in range(1, 6):
        bb[bb['STATUS'] == str(i)]['STATUS'] = 'PASS_DUE'

    # get the latest status for a certain bureau id
    def find_lateset_status(df):
        # note that month balance is negative
        return df[df['MONTHS_BALANCE'] == df['MONTHS_BALANCE'].max()]['STATUS'].values[0]

    latest_status = {}
    for sk_id, sub_df in bb.groupby('SK_ID_BUREAU'):
        latest_status[sk_id] = find_lateset_status(sub_df)
    bb_agg['LATEST_STATUS'] = pd.Series(latest_status)

    # respectively one hot encode bb_agg and bb
    bb_agg, _ = one_hot_encoding(bb_agg, nan_as_category)
    bb, bb_cat = one_hot_encoding(bb, nan_as_category)
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}

    status_aggregations = {}
    for col in bb_cat:
        status_aggregations[col] = ['mean', 'sum']

    bb_agg_auto = bb.groupby('SK_ID_BUREAU').agg({**bb_aggregations, **status_aggregations})
    bb_agg_auto.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg_auto.columns.tolist()])
    bb_agg = bb_agg.join(bb_agg_auto, on='SK_ID_BUREAU', how='left')
    del bb_agg_auto
    gc.collect()

    # reset bb_cat
    bb_cat = [col for col in bb_agg.columns if 'STATUS_CHANGE' not in col and 'MONTHS_BALANCE' not in col]

    bureau = bureau.join(bb_agg, on='SK_ID_BUREAU', how='left')
    bureau.drop(columns='SK_ID_BUREAU', inplace=True)
    del bb_agg
    gc.collect()

    #########################
    ### process on bureau ###
    #########################
    # procesing on DAYS related features
    bureau['DAYS_CREDIT_SUB_UPDATE'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_UPDATE']
    bureau['DAYS_CREDIT_TO_UPDATE_RATIO'] = (bureau['DAYS_CREDIT'] - 1) / (bureau['DAYS_CREDIT_UPDATE'] - 1)

    bureau['DAYS_ENDDATE_SUB_UPDATE'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT_UPDATE']
    bureau['DAYS_ENDDATE_TO_UPDATE_RATIO'] = (bureau['DAYS_CREDIT_ENDDATE'] - 1) / (bureau['DAYS_CREDIT_UPDATE'] - 1)

    bureau['DAYS_FACT_SUB_ENDDATE'] = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['DAYS_FACT_SUB_UPDATE'] = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_UPDATE']
    bureau['DAYS_FACT_TO_UPDATE_RATIO'] = bureau['DAYS_ENDDATE_FACT'] / bureau['DAYS_CREDIT_UPDATE']

    bureau['IS_EARLY_PAID'] = (bureau['DAYS_ENDDATE_FACT'] < bureau['DAYS_CREDIT_ENDDATE']).astype('float')
    bureau['IS_LATER_PAID'] = (bureau['DAYS_ENDDATE_FACT'] > bureau['DAYS_CREDIT_ENDDATE']).astype('float')

    bureau['PLAN_TIME_SPAN'] = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT']
    bureau['ACTUAL_TIME_SPAN'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
    bureau['ACTUAL_TIME_SPAN_TO_PLAN_RATIO'] = bureau['ACTUAL_TIME_SPAN'] / bureau['PLAN_TIME_SPAN']

    bureau['DAYS_FACT_SUB_UPDATE_TO_ACTUAL_TIME_SPAN_RATIO'] = bureau['DAYS_FACT_SUB_UPDATE'] / bureau['ACTUAL_TIME_SPAN']
    bureau['DAYS_ENDDATE_SUB_UPDATE_TO_PLAN_TIME_SPAN_RATIO'] = bureau['DAYS_FACT_SUB_UPDATE'] / bureau['PLAN_TIME_SPAN']

    # this means a client applied for a loan in homeCredit before his original creditBureau loan ended
    bureau['DAYS_ENDDATE_TO_DAYS_CREDIT_RATIO'] = bureau['DAYS_CREDIT_ENDDATE'] / bureau['DAYS_CREDIT']
    bureau['DAYS_OVERDUE_TO_CREDIT_RATIO'] = -1 * bureau['CREDIT_DAY_OVERDUE'] / bureau['DAYS_CREDIT']
    bureau['DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO'] = bureau['CREDIT_DAY_OVERDUE'] / bureau['PLAN_TIME_SPAN']
    bureau['DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO'] = bureau['CREDIT_DAY_OVERDUE'] / bureau['ACTUAL_TIME_SPAN']

    # this feature means a user have paid his loan before he applied
    bureau['IS_UNTRUSTWORTHY'] = (bureau['DAYS_CREDIT_ENDDATE'] < bureau['DAYS_CREDIT']).astype('float')
    bureau['IS_END_IN_FUTURE'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype('float')
    bureau['IS_OLD_UPDATE'] = (bureau['DAYS_CREDIT_UPDATE'] < -365).astype('float')

    # do some correction on AMT features
    def is_null_or_zero(val):
        return pd.isnull(val) or val == 0

    def correct_credit_debt(df):
        sum_, limit, debt = df['AMT_CREDIT_SUM'], df['AMT_CREDIT_SUM_LIMIT'], df['AMT_CREDIT_SUM_DEBT']
        if is_null_or_zero(debt) and not is_null_or_zero(limit):
            debt = sum_ - limit
        return debt

    def correct_credit_limit(df):
        sum_, limit, debt = df['AMT_CREDIT_SUM'], df['AMT_CREDIT_SUM_LIMIT'], df['AMT_CREDIT_SUM_DEBT']
        if is_null_or_zero(limit) and not is_null_or_zero(debt):
            limit = sum_ - debt
        return limit

    bureau['AMT_CREDIT_SUM_DEBT'] = bureau.apply(correct_credit_debt, axis=1)
    bureau['AMT_CREDIT_SUM_LIMIT'] = bureau.apply(correct_credit_limit, axis=1)

    # processing on AMT features
    bureau['AMT_OVERDUE_TO_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / bureau['AMT_CREDIT_SUM']
    bureau['AMT_OVERDUE_TO_DEBT_RATIO'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / bureau['AMT_CREDIT_SUM_DEBT']
    bureau['AMT_DEBT_TO_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']
    bureau['AMT_LIMIT_TO_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_LIMIT'] / bureau['AMT_CREDIT_SUM']
    bureau['AMT_MAX_OVERDUE_TO_CREDIT_RATIO'] = bureau['AMT_CREDIT_MAX_OVERDUE'] / bureau['AMT_CREDIT_SUM']

    bureau['AMT_ANNUITY_TO_CREDIT_RATIO'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']
    bureau['AMT_ANNUITY_TO_DEBT_RATIO'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM_DEBT']

    bureau['AVG_ANNUITY_BY_MONTH'] = bureau['AMT_ANNUITY'] / bureau['MONTHS_BALANCE_SIZE']
    bureau['AVG_CREDIT_BY_MONTH'] = bureau['AMT_CREDIT_SUM'] / bureau['MONTHS_BALANCE_SIZE']
    bureau['AVG_DEBT_BY_MONTH'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['MONTHS_BALANCE_SIZE']
    bureau['AVG_LIMIT_BY_MONTH'] = bureau['AMT_CREDIT_SUM_LIMIT'] / bureau['MONTHS_BALANCE_SIZE']

    bureau['IS_DEBT_NEG'] = (bureau['AMT_CREDIT_SUM_DEBT'] < 0).astype('float')
    bureau['IS_LIMIT_NEG'] = (bureau['AMT_CREDIT_SUM_DEBT'] < 0).astype('float')

    # CREDIT TYPE DIVERSITY
    bureau_agg = bureau.groupby('SK_ID_CURR')[['CREDIT_TYPE']]\
                    .nunique().rename(columns={'CREDIT_TYPE': 'BURO_CREDIT_TYPE_CNT'})

    bureau_agg['BURO_TOTAL_CREDIT_CNT'] = bureau.groupby('SK_ID_CURR')['CREDIT_TYPE'].size()
    bureau_agg['BURO_CREDIT_TYPE_DIVERSITY'] = bureau_agg['BURO_TOTAL_CREDIT_CNT'] / bureau_agg['BURO_CREDIT_TYPE_CNT']

    # Calculate the change trend on some numerical cols
    bureau = bureau.groupby('SK_ID_CURR').\
                  apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).\
                  reset_index(drop=True)

    change_ratio_cols = [
        'AMT_CREDIT_SUM',
        'AMT_CREDIT_SUM_DEBT',
        'AMT_ANNUITY',
        'AMT_CREDIT_MAX_OVERDUE',
        'CNT_CREDIT_PROLONG',

        'DAYS_CREDIT_SUB_UPDATE',
        'DAYS_CREDIT_TO_UPDATE_RATIO',
        'DAYS_ENDDATE_SUB_UPDATE',
        'DAYS_ENDDATE_TO_UPDATE_RATIO',

        'DAYS_FACT_SUB_ENDDATE',
        'DAYS_FACT_SUB_UPDATE',
        'DAYS_FACT_TO_UPDATE_RATIO',

        'PLAN_TIME_SPAN',
        'ACTUAL_TIME_SPAN',
        'ACTUAL_TIME_SPAN_TO_PLAN_RATIO',

        'DAYS_FACT_SUB_UPDATE_TO_ACTUAL_TIME_SPAN_RATIO',
        'DAYS_ENDDATE_SUB_UPDATE_TO_PLAN_TIME_SPAN_RATIO',

        'DAYS_ENDDATE_TO_DAYS_CREDIT_RATIO',
        'DAYS_OVERDUE_TO_CREDIT_RATIO',
        'DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO',
        'DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO',

        'AMT_OVERDUE_TO_CREDIT_RATIO',
        'AMT_OVERDUE_TO_DEBT_RATIO',
        'AMT_DEBT_TO_CREDIT_RATIO',
        'AMT_LIMIT_TO_CREDIT_RATIO',
        'AMT_MAX_OVERDUE_TO_CREDIT_RATIO',

        'AMT_ANNUITY_TO_CREDIT_RATIO',
        'AMT_ANNUITY_TO_DEBT_RATIO',

        'AVG_ANNUITY_BY_MONTH',
        'AVG_CREDIT_BY_MONTH',
        'AVG_DEBT_BY_MONTH',

        'MONTHS_BALANCE_SIZE'
    ]

    diff_cols = [
        'DAYS_CREDIT',
        'CREDIT_DAY_OVERDUE',
        'DAYS_CREDIT_ENDDATE',
        'DAYS_ENDDATE_FACT',
        'DAYS_CREDIT_UPDATE',
    ]

    diff_cols += change_ratio_cols

    def change_ratio(df):
        return df.shift(-1) / df - 1

    def diff(df):
        return df.shift(-1) - df

    new_col_cg_ratio = [col + '_CHANGE_RATIO' for col in change_ratio_cols]
    bureau[new_col_cg_ratio] = bureau.groupby('SK_ID_CURR')[change_ratio_cols].apply(change_ratio)

    new_col_diff = [col + '_DIFF' for col in diff_cols]
    bureau[new_col_diff] = bureau.groupby('SK_ID_CURR')[diff_cols].apply(diff)

    # APP TIME SPAN
    bureau_agg['BURO_ALL_CREDIT_TIME_SPAN'] = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].agg(lambda x: x.max() - x.min())

    bureau, bureau_cat = one_hot_encoding(bureau, nan_as_category)

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        # Original Features
        'DAYS_CREDIT': ['mean', 'var', 'max', 'min'],
        'CREDIT_DAY_OVERDUE': ['mean', 'min', 'max'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'min', 'max'],
        'DAYS_ENDDATE_FACT': ['mean', 'min', 'max'],
        'DAYS_CREDIT_UPDATE': ['mean', 'min', 'max'],

        'AMT_CREDIT_MAX_OVERDUE': ['sum', 'mean', 'min', 'max'],
        'AMT_CREDIT_SUM': ['sum', 'mean', 'min', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'min', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'min', 'max'],
        'AMT_CREDIT_SUM_LIMIT': ['sum', 'mean', 'min', 'max'],
        'AMT_ANNUITY': ['sum', 'mean', 'min', 'max'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean', 'min', 'max'],

        # Manual Features on DAYS
        'DAYS_CREDIT_SUB_UPDATE': ['mean', 'min', 'max', 'sum'],
        'DAYS_CREDIT_TO_UPDATE_RATIO': ['mean', 'min', 'max'],
        'DAYS_ENDDATE_SUB_UPDATE': ['mean', 'min', 'max', 'sum'],
        'DAYS_ENDDATE_TO_UPDATE_RATIO': ['mean', 'min', 'max'],


        'DAYS_FACT_SUB_ENDDATE': ['mean', 'min', 'max', 'sum'],
        'DAYS_FACT_SUB_UPDATE': ['mean', 'min', 'max', 'sum'],
        'DAYS_FACT_TO_UPDATE_RATIO': ['mean', 'min', 'max'],

        'IS_EARLY_PAID': ['mean', 'sum'],
        'IS_LATER_PAID': ['mean', 'sum'],

        'PLAN_TIME_SPAN': ['mean', 'min', 'max', 'sum'],
        'ACTUAL_TIME_SPAN': ['mean', 'min', 'max', 'sum'],
        'ACTUAL_TIME_SPAN_TO_PLAN_RATIO': ['mean', 'min', 'max'],

        'DAYS_FACT_SUB_UPDATE_TO_ACTUAL_TIME_SPAN_RATIO': ['mean', 'min', 'max'],
        'DAYS_ENDDATE_SUB_UPDATE_TO_PLAN_TIME_SPAN_RATIO': ['mean', 'min', 'max'],

        'DAYS_ENDDATE_TO_DAYS_CREDIT_RATIO': ['mean', 'min', 'max'],
        'DAYS_OVERDUE_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO': ['mean', 'min', 'max'],
        'DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO': ['mean', 'min', 'max'],

        'IS_UNTRUSTWORTHY': ['mean', 'sum'],
        'IS_END_IN_FUTURE': ['mean', 'sum'],
        'IS_OLD_UPDATE': ['mean', 'sum'],

        # Manual Features on AMT
        'AMT_OVERDUE_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'AMT_OVERDUE_TO_DEBT_RATIO': ['mean', 'min', 'max'],
        'AMT_DEBT_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'AMT_LIMIT_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'AMT_MAX_OVERDUE_TO_CREDIT_RATIO': ['mean', 'min', 'max'],

        'AMT_ANNUITY_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'AMT_ANNUITY_TO_DEBT_RATIO': ['mean', 'min', 'max'],

        'AVG_ANNUITY_BY_MONTH': ['mean', 'min', 'max'],
        'AVG_CREDIT_BY_MONTH': ['mean', 'min', 'max'],
        'AVG_DEBT_BY_MONTH': ['mean', 'min', 'max'],
        'AVG_LIMIT_BY_MONTH': ['mean', 'min', 'max'],

        'IS_DEBT_NEG': ['mean', 'sum'],
        'IS_LIMIT_NEG': ['mean', 'sum'],

        # Numerical Features come from bureau_balance
        'STATUS_CHANGE': ['sum', 'mean'],

        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    for col in new_col_cg_ratio + new_col_diff:
        num_aggregations[col] = ['min', 'max', 'mean']

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean', 'sum']
    for cat in bb_cat:
        cat_aggregations[cat] = ['mean', 'sum']

    bureau_agg_auto = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg_auto.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg_auto.columns.tolist()])
    bureau_agg = bureau_agg.join(bureau_agg_auto, how='left', on='SK_ID_CURR')
    del bureau_agg_auto
    gc.collect()

    bureau_agg['BURO_DAYS_CREDIT_DIFF_MEAN_TO_ACTUAL_TIME_SPAN_MEAN_RATIO'] = bureau_agg['BURO_DAYS_CREDIT_DIFF_MEAN'] / bureau_agg['BURO_ACTUAL_TIME_SPAN_MEAN']
    bureau_agg['BURO_DAYS_CREDIT_DIFF_MEAN_TO_ACTUAL_TIME_SPAN_SUM_RATIO'] = bureau_agg['BURO_DAYS_CREDIT_DIFF_MEAN'] / bureau_agg['BURO_ACTUAL_TIME_SPAN_SUM']
    bureau_agg['BURO_ALL_CREDIT_TIME_SPAN_TO_ACUTAL_TIME_SPAN_SUM_RATIO'] = bureau_agg['BURO_ALL_CREDIT_TIME_SPAN'] / bureau_agg['BURO_ACTUAL_TIME_SPAN_SUM']

    bureau_agg['BURO_AMT_CREDIT_SUM_SUM_TO_LIMIT_SUM_RATIO'] = bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] / bureau_agg['BURO_AMT_CREDIT_SUM_LIMIT_SUM']
    bureau_agg['BURO_AMT_CREDIT_SUM_SUM_TO_DEBT_SUM_RATIO'] = bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] / bureau_agg['BURO_AMT_CREDIT_SUM_DEBT_SUM']
    bureau_agg['BURO_AMT_CREDIT_SUM_SUM_TO_MAX_OVERDUE_SUM_RATIO'] = bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] / bureau_agg['BURO_AMT_CREDIT_MAX_OVERDUE_SUM']
    bureau_agg['BURO_AMT_CREDIT_SUM_SUM_TO_CREDIT_SUM_OVERDUE_SUM_RATIO'] = bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] / bureau_agg['BURO_AMT_CREDIT_SUM_OVERDUE_SUM']
    bureau_agg['BURO_AMT_CREDIT_SUM_SUM_TO_ANNUITYE_SUM_RATIO'] = bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] / bureau_agg['BURO_AMT_ANNUITY_SUM']

    bureau_agg['BURO_AVG_CREDIT_ON_BUREAU_BALANCE_REC'] = bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] / bureau_agg['BURO_MONTHS_BALANCE_SIZE_SUM']
    bureau_agg['BURO_AVG_LIMIT_ON_BUREAU_BALANCE_REC'] = bureau_agg['BURO_AMT_CREDIT_SUM_LIMIT_SUM'] / bureau_agg['BURO_MONTHS_BALANCE_SIZE_SUM']
    bureau_agg['BURO_AVG_DEBT_ON_BUREAU_BALANCE_REC'] = bureau_agg['BURO_AMT_CREDIT_SUM_DEBT_SUM'] / bureau_agg['BURO_MONTHS_BALANCE_SIZE_SUM']
    bureau_agg['BURO_AVG_MAX_OVERDUE_ON_BUREAU_BALANCE_REC'] = bureau_agg['BURO_AMT_CREDIT_MAX_OVERDUE_SUM'] / bureau_agg['BURO_MONTHS_BALANCE_SIZE_SUM']
    bureau_agg['BURO_AVG_OVERDUE_ON_BUREAU_BALANCE_REC'] = bureau_agg['BURO_AMT_CREDIT_SUM_OVERDUE_SUM'] / bureau_agg['BURO_MONTHS_BALANCE_SIZE_SUM']
    bureau_agg['BURO_AVG_ANNUITY_ON_BUREAU_BALANCE_REC'] = bureau_agg['BURO_AMT_ANNUITY_SUM'] / bureau_agg['BURO_MONTHS_BALANCE_SIZE_SUM']

    bureau_agg['BURO_AMT_ANNUITY_SUM_DIV_ACTUAL_TIME_SPAN_SUM'] = bureau_agg['BURO_AMT_ANNUITY_SUM'] / bureau_agg['BURO_ACTUAL_TIME_SPAN_SUM']
    bureau_agg['BURO_AMT_CREDIT_SUM_SUM_DIV_ACTUAL_TIME_SPAN_SUM'] = bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] / bureau_agg['BURO_ACTUAL_TIME_SPAN_SUM']
    bureau_agg['BURO_AMT_DEBT_SUM_DIV_ACTUAL_TIME_SPAN_SUM'] = bureau_agg['BURO_AMT_CREDIT_SUM_DEBT_SUM'] / bureau_agg['BURO_ACTUAL_TIME_SPAN_SUM']

    bureau_agg['BURO_AVG_PROLONG_CREDIT'] = bureau_agg['BURO_AMT_CREDIT_SUM_SUM'] / bureau_agg['BURO_CNT_CREDIT_PROLONG_SUM']
    bureau_agg['BURO_AVG_PROLONG_ANNUITY'] = bureau_agg['BURO_AMT_ANNUITY_SUM'] / bureau_agg['BURO_CNT_CREDIT_PROLONG_SUM']
    bureau_agg['BURO_AVG_PROLONG_CREDIT_OVERDUE'] = bureau_agg['BURO_AMT_CREDIT_SUM_OVERDUE_SUM'] / bureau_agg['BURO_CNT_CREDIT_PROLONG_SUM']

    # Bureau: Active credits
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg({**num_aggregations})
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg({**num_aggregations})
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg
    gc.collect()

    # Bureau: future credits
    future = bureau[bureau['IS_END_IN_FUTURE'] == 1]
    future_agg = future.groupby('SK_ID_CURR').agg({**num_aggregations})
    future_agg.columns = pd.Index(['FUTURE_' + e[0] + "_" + e[1].upper() for e in future_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(future_agg, how='left', on='SK_ID_CURR')
    del future, future_agg
    gc.collect()

    return bureau_agg


def previous_applications(prev, nan_as_category=True):
    # Days 365.243 values -> nan
    prev.loc[prev['AMT_DOWN_PAYMENT'] < 0, 'AMT_DOWN_PAYMENT'] = np.nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev['FLAG_LAST_APPL_PER_CONTRACT'], _ = pd.factorize(prev['FLAG_LAST_APPL_PER_CONTRACT'])

    # handle on AMT features
    prev['APP_TO_ANNUITY_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_ANNUITY']
    prev['APP_TO_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['APP_TO_DOWN_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_DOWN_PAYMENT']
    prev['APP_TO_PRICE_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_GOODS_PRICE']

    prev['ANNUITY_TO_CREDIT_RATIO'] = prev['AMT_ANNUITY'] / prev['AMT_CREDIT']
    prev['ANNUITY_TO_DOWN_RATIO'] = prev['AMT_ANNUITY'] / prev['AMT_DOWN_PAYMENT']
    prev['ANNUITY_TO_PRICE_RATIO'] = prev['AMT_ANNUITY'] / prev['AMT_GOODS_PRICE']

    prev['CREDIT_TO_DOWN_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_DOWN_PAYMENT']
    prev['CREDIT_TO_PRICE_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_GOODS_PRICE']

    prev['DOWN_TO_PRICE_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_GOODS_PRICE']
    prev['APP_SUB_CREDIT'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']

    # AVG AMT
    prev['AVG_PAYMENT_AMT_CREDIT'] = prev['AMT_CREDIT'] / prev['CNT_PAYMENT']
    prev['AVG_PAYMENT_AMT_ANNUITY'] = prev['AMT_ANNUITY'] / prev['CNT_PAYMENT']
    prev['AVG_PAYMENT_TOTAL'] = prev['AVG_PAYMENT_AMT_CREDIT'] + prev['AVG_PAYMENT_AMT_ANNUITY']

    # handle on DAYS features
    prev['PLAN_TIME_SPAN'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_FIRST_DUE']
    prev['ACTUAL_TIME_SPAN'] = prev['DAYS_LAST_DUE'] - prev['DAYS_FIRST_DUE']
    prev['LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE'] - prev['DAYS_LAST_DUE_1ST_VERSION']
    prev['ACTUAL_TIME_SPAN_TO_PLAN_RATIO'] = prev['ACTUAL_TIME_SPAN'] / prev['PLAN_TIME_SPAN']
    prev['DAYS_DESICION_TO_FTRST_DUE_RATIO'] = prev['DAYS_DECISION'] / prev['DAYS_FIRST_DUE']
    prev['DAYS_TERMINATION_SUB_LAST_DUE'] = prev['DAYS_TERMINATION'] - prev['DAYS_LAST_DUE']

    prev['IS_EARLY_PAID'] = (prev['DAYS_LAST_DUE'] < prev['DAYS_LAST_DUE_1ST_VERSION']).astype('float')
    # LAST_DUE was later than planned, might indicate finicial difficulty
    prev['IS_LATER_PAID'] = (prev['DAYS_LAST_DUE'] > prev['DAYS_LAST_DUE_1ST_VERSION']).astype('float')
    prev['IS_FISRT_DRAWING_LATER_THAN_LAST_DUE'] = (prev['DAYS_FIRST_DRAWING'] > prev['DAYS_LAST_DUE']).astype('float')
    prev['IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE'] = (prev['DAYS_FIRST_DRAWING'] > prev['DAYS_FIRST_DUE']).astype('float')

    prev['AVG_PAYMENT_DAYS'] = prev['ACTUAL_TIME_SPAN'] / prev['CNT_PAYMENT']

    prev['AVG_PAYMENT_BY_DAY'] = prev['AMT_CREDIT'] / prev['ACTUAL_TIME_SPAN']
    prev['AVG_ANNUITY_BY_DAY'] = prev['AMT_ANNUITY'] / prev['ACTUAL_TIME_SPAN']
    prev['AVG_TOTAL_PAYMENT_BY_DAY'] = prev['AVG_ANNUITY_BY_DAY'] + prev['AVG_PAYMENT_BY_DAY']

    # OTHER
    prev['IS_SELLERPLACE_AREA_MINUS_1'] = (prev['SELLERPLACE_AREA'] == -1).astype('float')
    prev['IS_SELLERPLACE_AREA_ZERO'] = (prev['SELLERPLACE_AREA'] == 0).astype('float')

    # DO SOME AGG
    prev['IS_X_SELL'] = (prev['NAME_PRODUCT_TYPE'] == 'x-sell').astype('float')
    prev['IS_WALK_IN'] = (prev['NAME_PRODUCT_TYPE'] == 'walk-in').astype('float')
    prev['IS_APPROVED'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype('float')
    prev['IS_REFUSED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype('float')

    by_cols = [
        ('NAME_CONTRACT_TYPE', 'NAME_PAYMENT_TYPE', 'NAME_PORTFOLIO', 'NAME_YIELD_GROUP'),
        ('NAME_CASH_LOAN_PURPOSE', 'NAME_GOODS_CATEGORY', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY'),
        ('WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE'),
    ]

    on_cols = [
        'IS_X_SELL',
        'IS_WALK_IN',
        'IS_APPROVED',
        'IS_REFUSED',
    ]

    by_cols_2 = [
        ('NAME_CONTRACT_TYPE', 'NAME_PAYMENT_TYPE', 'NAME_PORTFOLIO', 'NAME_YIELD_GROUP', 'NAME_CONTRACT_STATUS', 'NAME_PRODUCT_TYPE'),
        ('NAME_CASH_LOAN_PURPOSE', 'NAME_GOODS_CATEGORY', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_CONTRACT_STATUS', 'NAME_PRODUCT_TYPE'),
        ('WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_CONTRACT_STATUS', 'NAME_PRODUCT_TYPE'),
        ('PRODUCT_COMBINATION', 'NAME_GOODS_CATEGORY', 'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS'),
    ]

    on_cols_2 = [
        'AVG_TOTAL_PAYMENT_BY_DAY',
        'ACTUAL_TIME_SPAN',
        'IS_LATER_PAID',
        'IS_EARLY_PAID',
        'AVG_PAYMENT_TOTAL',
        'APP_TO_PRICE_RATIO',
        'DAYS_LAST_DUE_1ST_VERSION',
    ]

    prev = batch_agg(prev, by_cols, on_cols)
    prev = batch_agg(prev, by_cols_2, on_cols_2)

    def app_diversity_on_cate_cols(df, process_info):
        ret = df.groupby('SK_ID_CURR')['SK_ID_PREV'].count().\
            reset_index().\
            rename(index=str, columns={'SK_ID_PREV': 'PREV_USR_APP_CNT'})

        for col_name in process_info:
            new_col_name = 'PREV_N_UNIQUE_ON_' + col_name
            gby = df.groupby('SK_ID_CURR')[col_name].nunique().\
                reset_index().\
                rename(index=str, columns={col_name: new_col_name})
            ret = ret.merge(gby, on='SK_ID_CURR', how='left')
            ret['PREV_USR_APP_DIVERSITY_ON_' +
                col_name] = ret['PREV_USR_APP_CNT'] / ret[new_col_name]

        return ret

    prev_agg = app_diversity_on_cate_cols(
        prev, [col for col in prev.columns if prev[col].dtype == 'object']).set_index('SK_ID_CURR')

    # FIND THE TREND IN PREV 
    prev = prev.groupby('SK_ID_CURR').\
                apply(lambda x: x.sort_values(['DAYS_TERMINATION'], ascending=False)).\
                reset_index(drop=True)
    change_ratio_cols = [
        'AMT_ANNUITY',
        'AMT_APPLICATION',
        'AMT_CREDIT',
        'AMT_DOWN_PAYMENT',
        'AMT_GOODS_PRICE',

        'RATE_DOWN_PAYMENT',
        'RATE_INTEREST_PRIMARY',
        'RATE_INTEREST_PRIVILEGED',

        'DAYS_DECISION',
        'CNT_PAYMENT',

        'APP_TO_ANNUITY_RATIO',
        'APP_TO_CREDIT_RATIO',
        'APP_TO_DOWN_RATIO',
        'APP_TO_PRICE_RATIO',

        'ANNUITY_TO_CREDIT_RATIO',
        'ANNUITY_TO_DOWN_RATIO',
        'ANNUITY_TO_PRICE_RATIO',

        'CREDIT_TO_DOWN_RATIO',
        'CREDIT_TO_PRICE_RATIO',

        'DOWN_TO_PRICE_RATIO',
        'APP_SUB_CREDIT',

        'AVG_PAYMENT_AMT_CREDIT',
        'AVG_PAYMENT_AMT_ANNUITY',
        'AVG_PAYMENT_TOTAL',

        'PLAN_TIME_SPAN',
        'ACTUAL_TIME_SPAN',
        'LAST_DUE_DIFF',
        'ACTUAL_TIME_SPAN_TO_PLAN_RATIO',
        'DAYS_DESICION_TO_FTRST_DUE_RATIO',
        'DAYS_TERMINATION_SUB_LAST_DUE',

        'AVG_PAYMENT_DAYS',
        'AVG_PAYMENT_BY_DAY',
        'AVG_ANNUITY_BY_DAY',
        'AVG_TOTAL_PAYMENT_BY_DAY',
    ]

    diff_cols = [
        'DAYS_TERMINATION',
    ]

    diff_cols += change_ratio_cols

    def change_ratio(df):
        return df.shift(-1) / df - 1

    def diff(df):
        return df.shift(-1) - df

    new_col_cg_ratio = [col + '_CHANGE_RATIO' for col in change_ratio_cols]
    prev[new_col_cg_ratio] = prev.groupby('SK_ID_CURR')[change_ratio_cols].apply(change_ratio)

    new_col_diff = [col + '_DIFF' for col in diff_cols]
    prev[new_col_diff] = prev.groupby('SK_ID_CURR')[diff_cols].apply(diff)

    prev, cat_cols = one_hot_encoding(prev, nan_as_category)

    # EXTRACT THE LATEST APP FOR EACH USER
    cols = [
        'AMT_ANNUITY',
        'AMT_APPLICATION',
        'AMT_CREDIT',
        'AMT_GOODS_PRICE',

        'NFLAG_INSURED_ON_APPROVAL',

        'DAYS_DECISION',
        'CNT_PAYMENT',

        'DAYS_FIRST_DRAWING',
        'DAYS_FIRST_DUE',
        'DAYS_LAST_DUE',
        'DAYS_TERMINATION',

        'APP_TO_ANNUITY_RATIO',
        'APP_TO_CREDIT_RATIO',
        'APP_TO_DOWN_RATIO',
        'APP_TO_PRICE_RATIO',

        'ANNUITY_TO_CREDIT_RATIO',
        'ANNUITY_TO_DOWN_RATIO',
        'ANNUITY_TO_PRICE_RATIO',

        'CREDIT_TO_DOWN_RATIO',
        'CREDIT_TO_PRICE_RATIO',

        'DOWN_TO_PRICE_RATIO',
        'APP_SUB_CREDIT',

        'AVG_PAYMENT_AMT_CREDIT',
        'AVG_PAYMENT_AMT_ANNUITY',
        'AVG_PAYMENT_TOTAL',

        # DAYS
        'PLAN_TIME_SPAN',
        'ACTUAL_TIME_SPAN',
        'ACTUAL_TIME_SPAN_TO_PLAN_RATIO',
        'DAYS_DESICION_TO_FTRST_DUE_RATIO',
        'DAYS_TERMINATION_SUB_LAST_DUE',

        'IS_LATER_PAID',
        'IS_FISRT_DRAWING_LATER_THAN_LAST_DUE',
        'IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE',

        'AVG_PAYMENT_DAYS',
        'AVG_PAYMENT_BY_DAY',
        'AVG_ANNUITY_BY_DAY',
        'AVG_TOTAL_PAYMENT_BY_DAY',

        'IS_SELLERPLACE_AREA_MINUS_1',
        'IS_SELLERPLACE_AREA_ZERO',
    ]

    def find_lateset_app(df, cols):
        # note that month balance is negative
        dest = df['DAYS_LAST_DUE_1ST_VERSION'].max()
        ret = df.loc[df['DAYS_LAST_DUE_1ST_VERSION'] == dest, cols].values.ravel()
        n = len(cols)
        if len(ret) == 0:
            ret = np.empty((n, ))
            ret[:] = np.nan
        elif len(ret) > n:
            ret = ret[:n]
        return ret

    latestDf = {}
    for sk_id, sub_df in prev.groupby('SK_ID_CURR'):
        latestDf[sk_id] = find_lateset_app(sub_df, cols)
    latestDf = pd.DataFrame(latestDf).T
    new_col_names = ['PREV_' + col + '_LATEST' for col in cols]
    latestDf.columns = pd.Index(new_col_names)

    prev_agg = prev_agg.join(latestDf)
    del latestDf
    gc.collect()
    # AGG TO SK_ID_CURR
    num_aggregations = {
        # original features
        'AMT_ANNUITY': ['mean', 'sum', 'min', 'max'],
        'AMT_APPLICATION': ['mean', 'sum', 'min', 'max'],
        'AMT_CREDIT': ['mean', 'sum', 'min', 'max'],
        'AMT_DOWN_PAYMENT': ['mean', 'sum', 'min', 'max'],
        'AMT_GOODS_PRICE': ['mean', 'sum', 'min', 'max'],
        'HOUR_APPR_PROCESS_START': ['mean', 'min', 'max'],

        'FLAG_LAST_APPL_PER_CONTRACT': ['mean', 'sum'],
        'NFLAG_LAST_APPL_IN_DAY': ['mean', 'sum'],
        'NFLAG_INSURED_ON_APPROVAL': ['mean', 'sum'],

        'RATE_DOWN_PAYMENT': ['mean', 'min', 'max'],
        'RATE_INTEREST_PRIMARY': ['mean', 'min', 'max'],
        'RATE_INTEREST_PRIVILEGED': ['mean', 'min', 'max'],

        'DAYS_DECISION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],

        'DAYS_FIRST_DRAWING': ['min', 'max', 'mean'],
        'DAYS_FIRST_DUE': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        'DAYS_LAST_DUE': ['min', 'max', 'mean'],
        'DAYS_TERMINATION': ['min', 'max', 'mean'],

        # manual features
        # AMT
        'APP_TO_ANNUITY_RATIO': ['mean', 'min', 'max'],
        'APP_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'APP_TO_DOWN_RATIO': ['mean', 'min', 'max'],
        'APP_TO_PRICE_RATIO': ['mean', 'min', 'max'],

        'ANNUITY_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'ANNUITY_TO_DOWN_RATIO': ['mean', 'min', 'max'],
        'ANNUITY_TO_PRICE_RATIO': ['mean', 'min', 'max'],

        'CREDIT_TO_DOWN_RATIO': ['mean', 'min', 'max'],
        'CREDIT_TO_PRICE_RATIO': ['mean', 'min', 'max'],

        'DOWN_TO_PRICE_RATIO': ['mean', 'min', 'max'],
        'APP_SUB_CREDIT': ['mean', 'sum', 'min', 'max'],

        'AVG_PAYMENT_AMT_CREDIT': ['mean', 'sum', 'min', 'max'],
        'AVG_PAYMENT_AMT_ANNUITY': ['mean', 'sum', 'min', 'max'],
        'AVG_PAYMENT_TOTAL': ['mean', 'sum', 'min', 'max'],

        # DAYS
        'PLAN_TIME_SPAN': ['mean', 'sum', 'min', 'max'],
        'ACTUAL_TIME_SPAN': ['mean', 'sum', 'min', 'max'],
        'LAST_DUE_DIFF': ['mean', 'sum', 'min', 'max'],
        'ACTUAL_TIME_SPAN_TO_PLAN_RATIO': ['mean', 'min', 'max'],
        'DAYS_DESICION_TO_FTRST_DUE_RATIO': ['mean', 'min', 'max'],
        'DAYS_TERMINATION_SUB_LAST_DUE': ['mean', 'min', 'max'],

        'IS_EARLY_PAID': ['mean', 'sum'],
        'IS_LATER_PAID': ['mean', 'sum'],
        'IS_FISRT_DRAWING_LATER_THAN_LAST_DUE': ['mean', 'sum'],
        'IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE': ['mean', 'sum'],

        'AVG_PAYMENT_DAYS': ['mean', 'min', 'max'],
        'AVG_PAYMENT_BY_DAY': ['mean', 'min', 'max'],
        'AVG_ANNUITY_BY_DAY': ['mean', 'min', 'max'],
        'AVG_TOTAL_PAYMENT_BY_DAY': ['mean', 'min', 'max'],

        'IS_SELLERPLACE_AREA_MINUS_1': ['mean', 'sum'],
        'IS_SELLERPLACE_AREA_ZERO': ['mean', 'sum'],
    }

    for col in new_col_cg_ratio + new_col_diff:
        num_aggregations[col] = ['min', 'max', 'mean']

    agg_aggregations = {}
    agg_cols = [col for col in prev.columns if 'MEAN_ON_' in col or 'STD_ON_' in col]
    for agg in agg_cols:
        agg_aggregations[agg] = ['median', 'mean']

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean', 'sum']

    prev_agg_auto = prev.groupby('SK_ID_CURR').agg(
        {**num_aggregations, **cat_aggregations, **agg_aggregations})
    prev_agg_auto.columns = pd.Index(
        ['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg_auto.columns.tolist()])

    # join back to prev_agg
    prev_agg = prev_agg.join(prev_agg_auto)
    del prev_agg_auto
    gc.collect()

    prev_agg['PREV_DAYS_TERMINATION_DIFF_MAX_DIV_ACTUAL_TIME_SPAN_SUM'] = prev_agg['PREV_DAYS_TERMINATION_DIFF_MAX'] / prev_agg['PREV_ACTUAL_TIME_SPAN_SUM']
    prev_agg['PREV_DAYS_TERMINATION_DIFF_MEAN_DIV_ACTUAL_TIME_SPAN_MEAN'] = prev_agg['PREV_DAYS_TERMINATION_DIFF_MEAN'] / prev_agg['PREV_ACTUAL_TIME_SPAN_MEAN']
    prev_agg['PREV_IS_LATER_PAID_SUM_DIV_IS_EARLY_SUM_PAID'] = prev_agg['PREV_IS_LATER_PAID_SUM'] / prev_agg['PREV_IS_EARLY_PAID_SUM']

    # Previous Applications: Approved Applications
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(
        ['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    del approved, approved_agg
    gc.collect()

    # Previous Applications: Refused Applications
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(
        ['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg
    gc.collect()

    # Previous Applications: X_SELL Applications
    xSell = prev[prev['IS_X_SELL'] == 1]
    xSell_agg = xSell.groupby('SK_ID_CURR').agg({**num_aggregations})
    xSell_agg.columns = pd.Index(
        ['XSELL_' + e[0] + "_" + e[1].upper() for e in xSell_agg.columns.tolist()])
    prev_agg = prev_agg.join(xSell_agg, how='left', on='SK_ID_CURR')
    del xSell, xSell_agg
    gc.collect()
    return prev_agg


def pos_cash(pos, nan_as_category=True):
    latest_status = {}
    for sk_id, df in pos.groupby('SK_ID_CURR'):
        min_ = df['CNT_INSTALMENT_FUTURE'].min()
        if pd.isnull(min_):
            if len(df) == 1:
                latest_status[sk_id] = df['NAME_CONTRACT_STATUS'].values[0]
            else:
                latest_status[sk_id] = np.nan
        else:
            latest_status[sk_id] = df[df['CNT_INSTALMENT_FUTURE'] == min_]['NAME_CONTRACT_STATUS'].values[0]

    pos_agg = pd.DataFrame()
    pos_agg['LATEST_STATUS'] = pd.Series(latest_status)
    pos_agg, _ = one_hot_encoding(pos_agg, nan_as_category)

    pos, cat_cols = one_hot_encoding(pos, nan_as_category)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['sum', 'mean']

    pos_agg_auto = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg_auto.columns = pd.Index(
        ['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg_auto.columns.tolist()])

    pos_agg = pos_agg.join(pos_agg_auto)
    pos_agg['POS_REC_COUNT'] = pos.groupby('SK_ID_CURR').size()
    pos_agg['POS_AVG_DPD'] = pos_agg['POS_SK_DPD_SUM'] / pos_agg['POS_REC_COUNT']
    pos_agg['POS_AVG_DPD_DEF'] = pos_agg['POS_SK_DPD_DEF_SUM'] / pos_agg['POS_REC_COUNT']
    del pos, pos_agg_auto
    gc.collect()
    return pos_agg


def installments_payments(ins, nan_as_category=True):
    ins['AMT_PAYMENT_TO_INSTAL_RATIO'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['AMT_PAYMENT_SUB_INSTAL'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
    ins['IS_PAYMENNT_NOT_ENOUGH'] = (ins['AMT_PAYMENT_SUB_INSTAL'] < 0).astype('float')

    # Days past due and days before due (no negative values)
    ins['DAYS_ENTRY_TO_INSTAL_RATIO'] = ins['DAYS_ENTRY_PAYMENT'] / ins['DAYS_INSTALMENT']
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    ins = ins.groupby('SK_ID_CURR').\
              apply(lambda x: x.sort_values(['DAYS_INSTALMENT'], ascending=False)).\
              reset_index(drop=True)

    change_ratio_cols = [
        'AMT_INSTALMENT',
        'AMT_PAYMENT',
        'DAYS_ENTRY_TO_INSTAL_RATIO',
        'AMT_PAYMENT_TO_INSTAL_RATIO',
        'AMT_PAYMENT_SUB_INSTAL',
    ]

    diff_cols = [
        'DPD',
        'DBD',
        'DAYS_INSTALMENT',
        'DAYS_ENTRY_PAYMENT',
    ]

    diff_cols += change_ratio_cols

    def change_ratio(df):
        return df.shift(-1) / df - 1

    def diff(df):
        return df.shift(-1) - df

    new_col_cg_ratio = [col + '_CHANGE_RATIO' for col in change_ratio_cols]
    ins[new_col_cg_ratio] = ins.groupby('SK_ID_CURR')[change_ratio_cols].apply(change_ratio)

    new_col_diff = [col + '_DIFF' for col in diff_cols]
    ins[new_col_diff] = ins.groupby('SK_ID_CURR')[diff_cols].apply(diff)

    # agg by SK_ID_PREV
    agg_by_prev = ins.groupby('SK_ID_PREV')[['DAYS_INSTALMENT']].\
                agg(lambda x: x.max() - x.min()).reset_index().\
                rename(columns={'DAYS_INSTALMENT': 'TIME_SPAN'}).\
                set_index('SK_ID_PREV')
    agg_by_prev['INSTALL_TIMES'] = ins.groupby('SK_ID_PREV')['NUM_INSTALMENT_NUMBER'].max()
    agg_by_prev['INSTAL_VERSION_N_UNIQUE'] = ins.groupby('SK_ID_PREV')['NUM_INSTALMENT_VERSION'].nunique()
    agg_by_prev['INSTAL_VERSION_CHANGE'] = (agg_by_prev['INSTAL_VERSION_N_UNIQUE'] > 1).astype('float')
    agg_by_prev['INSTAL_DIVERSITY'] = agg_by_prev['INSTALL_TIMES'] / agg_by_prev['INSTAL_VERSION_N_UNIQUE']

    # merge SK_ID_CURR col
    agg_by_prev = agg_by_prev.merge(
                        ins[['SK_ID_PREV', 'SK_ID_CURR']].drop_duplicates('SK_ID_PREV'),
                        on='SK_ID_PREV',
                        how='left')

    ins_agg = ins.groupby('SK_ID_CURR')[['SK_ID_PREV']].count().rename(columns={'SK_ID_PREV': 'INSTAL_USR_REC_CNT'})
    ins_agg['INSTAL_USR_LOAN_CNT'] = ins.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
    ins_agg['INSTAL_REC_CNT_PER_LOAN'] = ins_agg['INSTAL_USR_REC_CNT'] / ins_agg['INSTAL_USR_LOAN_CNT']

    # merge agg_by_prev back to ins_agg on SK_ID_CURR
    prev_aggregation = {
        'TIME_SPAN': ['min', 'max', 'mean', 'sum'],
        'INSTALL_TIMES': ['min', 'max', 'mean'],
        'INSTAL_VERSION_N_UNIQUE': ['min', 'max', 'mean', 'sum'],
        'INSTAL_VERSION_CHANGE': ['sum', 'mean'],
        'INSTAL_DIVERSITY': ['min', 'max', 'mean']
    }

    for col, fcn_lst in prev_aggregation.items():
        for fcn in fcn_lst:
            new_col = col + '_' + fcn.upper()
            ins_agg[new_col] = agg_by_prev.groupby('SK_ID_CURR')[col].agg(fcn)

    # Features: Perform aggregations
    aggregations = {
        # Original Features
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],

        'DAYS_ENTRY_PAYMENT': ['max', 'min'],
        'DAYS_INSTALMENT': ['max', 'min'],

        # Manual Features
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'DAYS_ENTRY_TO_INSTAL_RATIO': ['max', 'mean', 'min'],
        'AMT_PAYMENT_TO_INSTAL_RATIO': ['min', 'mean', 'var'],
        'AMT_PAYMENT_SUB_INSTAL': ['min', 'mean', 'var'],
        'IS_PAYMENNT_NOT_ENOUGH': ['mean', 'sum'],
    }
    for col in new_col_cg_ratio + new_col_diff:
        aggregations[col] = ['min', 'max', 'mean']

    ins_agg_auto = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg_auto.columns = pd.Index(
        ['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg_auto.columns.tolist()])

    recent = ins[ins['DAYS_INSTALMENT'] > -365]
    recent_agg = recent.groupby('SK_ID_CURR').agg(aggregations)
    recent_agg.columns = pd.Index(
        ['RECENT_INSTAL_' + e[0] + "_" + e[1].upper() for e in recent_agg.columns.tolist()])

    ins_agg = ins_agg.merge(ins_agg_auto, on='SK_ID_CURR', how='left')
    ins_agg = ins_agg.merge(recent_agg, on='SK_ID_CURR', how='left')

    del ins, recent, ins_agg_auto, recent_agg, agg_by_prev
    gc.collect()
    return ins_agg


def credit_card_balance(cc, nan_as_category=True):
    # indicate the interest rate
    cc['RECIVABLE_TO_PRINCIPAL_RATIO'] = cc['AMT_RECIVABLE'] / cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['RECIVABLE_SUB_PRINCIPAL'] = cc['AMT_RECIVABLE'] - cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['AMT_RECIVABLE_EQ_TOTAL'] = (cc['AMT_RECIVABLE'] == cc['AMT_TOTAL_RECEIVABLE']).astype('float')

    cc['DRAWINGS_TO_CREDIT_RATIO'] = cc['AMT_DRAWINGS_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['BALANCE_TO_CREDIT_RATIO'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['PAYBACK_LT_INST_MIN'] = (cc['AMT_PAYMENT_CURRENT'] < cc['AMT_INST_MIN_REGULARITY']).astype('float')
    cc['PAYBACK_TO_INST_MIN_RATIO'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']

    cc['IS_DPD_GT_ZERO'] = (cc['SK_DPD'] > 0).astype('float')
    cc['IS_DPD_DEF_GT_ZERO'] = (cc['SK_DPD_DEF'] > 0).astype('float')

    # TREND BY TIME
    cc = cc.groupby('SK_ID_PREV').apply(lambda x: x.sort_values(['MONTHS_BALANCE'], ascending=False)).reset_index(drop=True)

    change_ratio_cols = [
        'AMT_BALANCE',
        'AMT_PAYMENT_CURRENT',
        'AMT_RECIVABLE',
        'AMT_INST_MIN_REGULARITY',

        'AMT_DRAWINGS_ATM_CURRENT',
        'AMT_DRAWINGS_CURRENT',
        'AMT_DRAWINGS_OTHER_CURRENT',
        'AMT_DRAWINGS_POS_CURRENT',

        'CNT_DRAWINGS_ATM_CURRENT',
        'CNT_DRAWINGS_CURRENT',
        'CNT_DRAWINGS_OTHER_CURRENT',
        'CNT_DRAWINGS_POS_CURRENT',

        'SK_DPD',

        'RECIVABLE_TO_PRINCIPAL_RATIO',
        'RECIVABLE_SUB_PRINCIPAL',

        'DRAWINGS_TO_CREDIT_RATIO',
        'BALANCE_TO_CREDIT_RATIO',
    ]

    diff_cols = change_ratio_cols

    def change_ratio(df):
        return df.shift(-1) / df - 1

    def diff(df):
        return df.shift(-1) - df

    new_col_cg_ratio = [col + '_CHANGE_RATIO' for col in change_ratio_cols]
    cc[new_col_cg_ratio] = cc.groupby('SK_ID_PREV')[change_ratio_cols].apply(change_ratio)

    new_col_diff = [col + '_DIFF' for col in diff_cols]
    cc[new_col_diff] = cc.groupby('SK_ID_PREV')[diff_cols].apply(diff)

    # number of loans
    cc_agg = cc.groupby('SK_ID_CURR')['SK_ID_PREV'].\
                nunique().\
                reset_index().\
                rename(index = str, columns = {'SK_ID_PREV': 'CC_USR_LOAN_CNT'})
    cc_agg.set_index('SK_ID_CURR', inplace=True)
    # number of credit balance records
    cc_agg['CC_BLANCE_REC_CNT'] = cc.groupby('SK_ID_CURR').size()

    # handle on payback times
    temp = cc.groupby(['SK_ID_PREV', 'SK_ID_CURR'])['CNT_INSTALMENT_MATURE_CUM'].\
                max().\
                reset_index().\
                rename(index=str, columns={'CNT_INSTALMENT_MATURE_CUM': 'INSTALLMENT_TIMES_PER_LOAN'})
    cc_agg['CC_PAYBACK_TIMES_TOTAL'] = temp.groupby('SK_ID_CURR')['INSTALLMENT_TIMES_PER_LOAN'].sum()
    cc_agg['CC_AVG_PAYBACK_TIMES'] = cc_agg['CC_PAYBACK_TIMES_TOTAL'] / cc_agg['CC_USR_LOAN_CNT']
    cc_agg['CC_PAYBACK_TO_REC_CNT_RATIO'] = cc_agg['CC_PAYBACK_TIMES_TOTAL'] / cc_agg['CC_BLANCE_REC_CNT']


    num_aggregations = {
        'MONTHS_BALANCE': ['min', 'max'],
        'AMT_BALANCE': ['max', 'mean'],
        'AMT_PAYMENT_CURRENT': ['max', 'mean', 'sum'],
        'AMT_RECIVABLE': ['max', 'mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],

        # AMT_DRAWINGS
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum'],

        # CNT_DRAWINGS
        'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum'],

        # DPD
        'SK_DPD': ['max', 'sum', 'mean'],
        'IS_DPD_GT_ZERO': ['sum', 'mean'],
        'SK_DPD_DEF': ['max', 'sum', 'mean'],
        'IS_DPD_DEF_GT_ZERO': ['sum', 'mean'],

        # OTHERS
        'RECIVABLE_TO_PRINCIPAL_RATIO': ['max', 'mean'],
        'RECIVABLE_SUB_PRINCIPAL': ['max', 'mean', 'min'],
        'AMT_RECIVABLE_EQ_TOTAL': ['sum', 'mean'],

        'DRAWINGS_TO_CREDIT_RATIO': ['max', 'sum', 'mean'],
        'BALANCE_TO_CREDIT_RATIO': ['max', 'sum', 'mean'],
        'PAYBACK_LT_INST_MIN': ['sum', 'mean'],
        'PAYBACK_TO_INST_MIN_RATIO': ['max', 'sum', 'mean'],
    }

    cc, cat_cols = one_hot_encoding(cc, nan_as_category)

    cate_aggregations = {}
    for cat in cat_cols:
        cate_aggregations[cat] = ['mean', 'sum']

    for col in new_col_cg_ratio + new_col_diff:
        num_aggregations[col] = ['min', 'max', 'mean']

    cc_agg_auto = cc.groupby('SK_ID_CURR').agg({**num_aggregations, **cate_aggregations})
    cc_agg_auto.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg_auto.columns.tolist()])

    # SOME ADDITIONAL OPERATION
    cc_agg_auto['CC_DRAWINGS_AMT_ATM_RATIO'] = cc_agg_auto['CC_AMT_DRAWINGS_ATM_CURRENT_SUM'] / cc_agg_auto['CC_AMT_DRAWINGS_CURRENT_SUM']
    cc_agg_auto['CC_DRAWINGS_AMT_OTHER_RATIO'] = cc_agg_auto['CC_AMT_DRAWINGS_OTHER_CURRENT_SUM'] / cc_agg_auto['CC_AMT_DRAWINGS_CURRENT_SUM']
    cc_agg_auto['CC_DRAWINGS_AMT_POS_RATIO'] = cc_agg_auto['CC_AMT_DRAWINGS_POS_CURRENT_SUM'] / cc_agg_auto['CC_AMT_DRAWINGS_CURRENT_SUM']

    cc_agg_auto['CC_DRAWINGS_CNT_ATM_RATIO'] = cc_agg_auto['CC_CNT_DRAWINGS_ATM_CURRENT_SUM'] / cc_agg_auto['CC_CNT_DRAWINGS_CURRENT_SUM']
    cc_agg_auto['CC_DRAWINGS_CNT_OTHER_RATIO'] = cc_agg_auto['CC_CNT_DRAWINGS_OTHER_CURRENT_SUM'] / cc_agg_auto['CC_CNT_DRAWINGS_CURRENT_SUM']
    cc_agg_auto['CC_DRAWINGS_CNT_POS_RATIO'] = cc_agg_auto['CC_CNT_DRAWINGS_POS_CURRENT_SUM'] / cc_agg_auto['CC_CNT_DRAWINGS_CURRENT_SUM']

    cc_agg_auto['CC_AVG_DRAWINGS'] = cc_agg_auto['CC_AMT_DRAWINGS_ATM_CURRENT_SUM'] / cc_agg_auto['CC_CNT_DRAWINGS_CURRENT_SUM']
    cc_agg = cc_agg.join(cc_agg_auto)
    del cc, temp, cc_agg_auto
    gc.collect()
    return cc_agg


def final_process(df, nan_as_category=True):
    df['FINAL_BURO_DAYS_CREDIT_ENDDATE_MAX_TO_APP_DAYS_BIRTH_RATIO'] = df['BURO_DAYS_CREDIT_ENDDATE_MAX'] / df['DAYS_BIRTH']
    df['FINAL_BURO_DAYS_CREDIT_ENDDATE_MAX_TO_APP_DAYS_EMPLOYED_RATIO'] = df['BURO_DAYS_CREDIT_ENDDATE_MAX'] / df['DAYS_EMPLOYED']
    df['FINAL_BURO_DAYS_ENDDATE_FACT_MAX_TO_APP_DAYS_BIRTH_RATIO'] = df['BURO_DAYS_ENDDATE_FACT_MAX'] / df['DAYS_BIRTH']
    df['FINAL_BURO_DAYS_ENDDATE_FACT_MAX_TO_APP_DAYS_EMPLOYED_RATIO'] = df['BURO_DAYS_ENDDATE_FACT_MAX'] / df['DAYS_EMPLOYED']

    df['FINAL_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN_TO_APP_AMT_CREDIT_RATIO'] = df['BURO_AMT_CREDIT_MAX_OVERDUE_MEAN'] / df['AMT_CREDIT']
    df['FINAL_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN_TO_APP_AMT_ANNUITY_RATIO'] = df['BURO_AMT_CREDIT_MAX_OVERDUE_MEAN'] / df['AMT_ANNUITY']
    df['FINAL_BURO_AMT_CREDIT_SUM_MAX_TO_APP_AMT_ANNUITY_RATIO'] = df['BURO_AMT_CREDIT_SUM_MAX'] / df['AMT_ANNUITY']
    df['FINAL_BURO_AMT_CREDIT_SUM_MAX_TO_APP_AMT_CREDIT_RATIO'] = df['BURO_AMT_CREDIT_SUM_MAX'] / df['AMT_CREDIT']

    df['FINAL_PREV_AMT_ANNUITY_MEAN_TO_APP_AMT_ANNUITY_RATIO'] = df['PREV_AMT_ANNUITY_MEAN'] / df['AMT_ANNUITY']
    df['FINAL_PREV_AMT_ANNUITY_MEAN_TO_APP_AMT_CREDIT_RATIO'] = df['PREV_AMT_ANNUITY_MEAN'] / df['AMT_CREDIT']
    df['FINAL_PREV_AMT_ANNUITY_MEAN_TO_BURO_AMT_CREDIT_SUM_MAX_RATIO'] = df['PREV_AMT_ANNUITY_MEAN'] / df['BURO_AMT_CREDIT_SUM_MAX']
    df['FINAL_PREV_AMT_APPLICATION_MEAN_TO_APP_AMT_CREDIT_RATIO'] = df['PREV_AMT_APPLICATION_MEAN'] / df['AMT_CREDIT']
    df['FINAL_PREV_AMT_APPLICATION_MEAN_TO_BURO_AMT_CREDIT_SUM_MAX_RATIO'] = df['PREV_AMT_APPLICATION_MEAN'] / df['BURO_AMT_CREDIT_SUM_MAX']

    df['FINAL_APP_AMT_CREDIT_TO_PREV_CNT_PAYMENT_MEAN_RATIO'] = df['AMT_CREDIT'] / df['PREV_CNT_PAYMENT_MEAN']
    df['FINAL_APP_AMT_ANNUITY_TO_PREV_CNT_PAYMENT_MEAN_RATIO'] = df['AMT_ANNUITY'] / df['PREV_CNT_PAYMENT_MEAN']
    df['FINAL_PREV_PLAN_TIME_SPAN_MEAN_TO_APP_DAYS_EMPLOYED_RATIO'] = df['PREV_PLAN_TIME_SPAN_MEAN'] / df['DAYS_EMPLOYED']
    df['FINAL_PREV_PLAN_TIME_SPAN_SUM_TO_APP_DAYS_EMPLOYED_RATIO'] = df['PREV_PLAN_TIME_SPAN_SUM'] / df['DAYS_EMPLOYED']
    df['FINAL_PREV_AVG_ANNUITY_BY_DAY_MEAN_TO_AMT_INCOME_TOTAL_RATIO'] = df['PREV_AVG_ANNUITY_BY_DAY_MEAN'] / df['AMT_INCOME_TOTAL']
    df['FINAL_PREV_AVG_TOTAL_PAYMENT_BY_DAY_MEAN_TO_AMT_INCOME_TOTAL_RATIO'] = df['PREV_AVG_TOTAL_PAYMENT_BY_DAY_MEAN'] / df['AMT_INCOME_TOTAL']

    df['FINAL_INSTAL_AMT_INSTALMENT_SUM_TO_APP_CREDIT_RATIO'] = df['INSTAL_AMT_INSTALMENT_SUM'] / df['AMT_CREDIT']
    df['FINAL_INSTAL_AMT_INSTALMENT_SUM_TO_APP_AMT_ANNUITY_RATIO'] = df['INSTAL_AMT_INSTALMENT_SUM'] / df['AMT_ANNUITY']
    df['FINAL_INSTAL_AMT_PAYMENT_SUM_TO_APP_AMT_CREDIT_RATIO'] = df['INSTAL_AMT_PAYMENT_SUM'] / df['AMT_CREDIT']
    df['FINAL_INSTAL_AMT_PAYMENT_SUM_TO_APP_AMT_INCOME_TOTAL_RATIO'] = df['INSTAL_AMT_PAYMENT_SUM'] / df['AMT_INCOME_TOTAL']
    df['FINAL_INSTAL_AMT_PAYMENT_SUM_TO_APP_AMT_ANNUITY_RATIO'] = df['INSTAL_AMT_PAYMENT_SUM'] / df['AMT_ANNUITY']
    df['FINAL_INSTAL_AMT_PAYMENT_SUM_TO_APP_DAYS_EMPLOYED_RATIO'] = df['INSTAL_AMT_PAYMENT_SUM'] / df['DAYS_EMPLOYED']
    df['FINAL_INSTAL_DBD_SUM_TO_APP_DAYS_EMPLOYED_RATIO'] = df['INSTAL_DBD_SUM'] / df['DAYS_EMPLOYED']
    df['FINAL_RECENT_INSTAL_DBD_MAX_TO_APP_DAYS_EMPLOYED_RATIO'] = df['RECENT_INSTAL_DBD_MAX'] / df['DAYS_EMPLOYED']

    df['FINAL_POS_REC_COUNT_TO_APP_DAYS_EMPLOYED_RATIO'] = df['POS_REC_COUNT'] / df['DAYS_EMPLOYED']

    # DO SOME AGGREGATION
    by_cols = [
        ('CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE'),
        ('CODE_GENDER', 'NAME_TYPE_SUITE', 'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN'),
        ('CODE_GENDER', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE'),
        ]

    on_cols = [
        'PREV_DAYS_LAST_DUE_1ST_VERSION_MAX',
        'PREV_DAYS_TERMINATION_MAX',
        'PREV_CREDIT_TO_PRICE_RATIO_MEAN',
        'PREV_DAYS_DECISION_MAX',
        'PREV_DAYS_DESICION_TO_FTRST_DUE_RATIO_MIN',

        'BURO_DAYS_CREDIT_MAX',
        'BURO_DAYS_CREDIT_DIFF_MEAN',
        'BURO_AMT_CREDIT_SUM_MIN',
        'BURO_DAYS_CREDIT_SUB_UPDATE_MEAN',
        'BURO_AMT_DEBT_TO_CREDIT_RATIO_MAX',
        'BURO_AMT_CREDIT_SUM_SUM_TO_LIMIT_SUM_RATIO',

        'INSTAL_DBD_SUM',
        'INSTAL_DPD_MEAN',
        'INSTAL_DAYS_ENTRY_TO_INSTAL_RATIO_MIN',
        'INSTAL_AMT_PAYMENT_SUB_INSTAL_MEAN',
        'INSTALL_TIMES_MEAN',

        'RECENT_INSTAL_AMT_INSTALMENT_MAX',
        'RECENT_INSTAL_AMT_PAYMENT_MAX',
        ]

    df = batch_agg(df, by_cols, on_cols, 'AGG_FINAL_')

    # Categorical features with One-Hot encode
    df, _ = one_hot_encoding(df, nan_as_category)
    return df


def feature_extract(debug, input_config):
    num_rows = 10000 if debug else None

    with timer("Process applications"):
        train_file = input_config['train_filepath']
        test_file = input_config['test_filepath']
        train_df = pd.read_csv(train_file, nrows=num_rows)
        test_df = pd.read_csv(test_file, nrows=num_rows)
        print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))
        df = application_train_test(train_df, test_df)
        print("Applications df shape:", df.shape)

    with timer("Process bureau and bureau_balance"):
        bureau_balance_file = input_config['bureau_balance_filepath']
        bureau_file = input_config['bureau_filepath']
        bureau = pd.read_csv(bureau_file, nrows=num_rows)
        bb = pd.read_csv(bureau_balance_file, nrows=num_rows)
        bureau = bureau_and_balance(bureau, bb)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()

    with timer("Process previous_applications"):
        prev_app_file = input_config['previous_application_filepath']
        prev = pd.read_csv(prev_app_file, nrows=num_rows)
        prev = previous_applications(prev)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()

    with timer("Process POS-CASH balance"):
        pos_cash_file = input_config['POS_CASH_balance_filepath']
        pos = pd.read_csv(pos_cash_file, nrows=num_rows)
        pos = pos_cash(pos)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()

    with timer("Process installments payments"):
        installments_file = input_config['installments_payments_filepath']
        ins = pd.read_csv(installments_file, nrows=num_rows)
        ins = installments_payments(ins)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()

    with timer("Process credit card balance"):
        credit_card_file = input_config['credit_card_balance_filepath']
        cc = pd.read_csv(credit_card_file, nrows=num_rows)
        cc = credit_card_balance(cc)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()

    with timer("Process Final Stage"):
        df = final_process(df)
        # feature filter
        # df.drop(columns=list(set(DROP_FEATS) & set(df.columns)), inplace=True)

    return df[df['TARGET'].notnull()], df[df['TARGET'].isnull()]
