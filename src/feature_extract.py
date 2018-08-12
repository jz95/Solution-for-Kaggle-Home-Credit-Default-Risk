import numpy as np
import pandas as pd
import gc
from src.utils import timer
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


def application_train_test(train_df, test_df, nan_as_category=False):
    # merge the train and test DataFrame
    df = train_df.append(test_df).reset_index()

    del train_df, test_df
    gc.collect()

    df = df[df['CODE_GENDER'] != 'XNA']
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

    df['APP_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['APP_PHONE_TO_REG_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_REGISTRATION']
    df['APP_PHONE_TO_EMPLOY_RATIO_'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['APP_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['APP_EMPLOY_TO_REG_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_REGISTRATION']
    df['APP_ID_PUB_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['APP_REG_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
    df['APP_ID_PUB_TO_RE_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_REGISTRATION']

    df['APP_INCOME_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['APP_INCOME_PER_ADULT'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN'])

    # EXT_SCORES
    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian', 'std']:
        df['APP_EXT_SOURCES_{}'.format(function_name.upper())] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    df['APP_SCORE1_TO_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_BIRTH'] / 365.25)
    df['APP_SCORE1_TO_EMPLOY_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_EMPLOYED'] / 365.25)
    df['APP_SCORE3_TO_BIRTH_RATIO'] = df['EXT_SOURCE_3'] / (df['DAYS_BIRTH'] / 365.25)
    df['APP_SCORE1_TO_SCORE2_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_2'] 
    df['APP_SCORE1_TO_SCORE3_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_3']
    df['APP_SCORE2_TO_REGION_RATING_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_RATING_CLIENT']
    df['APP_SCORE2_TO_CITY_RATING_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_RATING_CLIENT_W_CITY']
    df['APP_SCORE2_TO_POP_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_POPULATION_RELATIVE']

    # DOCUMENT
    doc_cols = [col for col in df.columns if '_DOCUMENT_' in col]
    df['APP_ALL_DOC_CNT'] = df[doc_cols].sum(axis=1)
    df['APP_DOC3_6_8_CNT'] = df['FLAG_DOCUMENT_3'] + df['FLAG_DOCUMENT_6'] + df['FLAG_DOCUMENT_8']
    df['APP_DOC3_6_8_CNT_RAIO'] = df['APP_DOC3_6_8_CNT'] / df['APP_ALL_DOC_CNT']

    # HOUSING FEATURES
    housing_info_cols = [col for col in df.columns if '_AVG' in col or '_MODE' in col or '_MEDI' in col]
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

    # OTHER
    df['APP_CAR_PLUS_REALTY'] = df['FLAG_OWN_REALTY'] + df['FLAG_OWN_CAR']
    df['APP_CHILD_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    df['APP_REGION_RATING_TO_POP_RATIO'] = df['REGION_RATING_CLIENT'] / df['REGION_POPULATION_RELATIVE']
    df['APP_CITY_RATING_TO_POP_RATIO'] = df['REGION_RATING_CLIENT_W_CITY'] / df['REGION_POPULATION_RELATIVE']

    df['APP_REGION_ALL_NOT_EQ'] = df['REG_REGION_NOT_LIVE_REGION'] + df['REG_REGION_NOT_WORK_REGION'] + df['LIVE_REGION_NOT_WORK_REGION']
    df['APP_CITY_ALL_NOT_EQ'] = df['REG_CITY_NOT_LIVE_CITY'] + df['REG_CITY_NOT_WORK_CITY'] + df['LIVE_CITY_NOT_WORK_CITY']

    contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
    df['APP_CONTACT_ALL_CNT'] = df[contact_cols].sum(axis=1)

    df['APP_DEF_TO_OBS_RATIO_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE']
    df['APP_DEF_TO_OBS_RATIO_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE']

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

    # Categorical features with One-Hot encode
    df, _ = one_hot_encoding(df, nan_as_category)
    return df


def bureau_and_balance(bureau, bb, nan_as_category=True):
    bb_one_year = bb[bb['MONTHS_BALANCE'] >= -12]
    bb_half_year = bb[bb['MONTHS_BALANCE'] >= -6]

    def see_change(x):
        return x.nunique() > 1

    bb_agg = bb.groupby('SK_ID_BUREAU')[['STATUS']].\
                agg(see_change).rename(columns={'STATUS': 'STATUS_CHANGE'}).astype('float')
    bb_agg_one_year = bb_one_year.groupby('SK_ID_BUREAU')[['STATUS']].\
                agg(see_change).rename(columns={'STATUS': 'ONE_YEAR_STATUS_CHANGE'}).astype('float')
    bb_agg_half_year = bb_half_year.groupby('SK_ID_BUREAU')[['STATUS']].\
                agg(see_change).rename(columns={'STATUS': 'HALF_YEAR_STATUS_CHANGE'}).astype('float')

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
    bb_one_year = bb[bb['MONTHS_BALANCE'] >= -12]
    bb_half_year = bb[bb['MONTHS_BALANCE'] >= -6]

    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}

    status_aggregations = {}
    for col in bb_cat:
        status_aggregations[col] = ['mean', 'sum']

    bb_agg_auto = bb.groupby('SK_ID_BUREAU').agg({**bb_aggregations, **status_aggregations})
    bb_agg_auto.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg_auto.columns.tolist()])
    bb_agg = bb_agg.join(bb_agg_auto, on='SK_ID_BUREAU', how='left')

    bb_agg_one_year_auto = bb_one_year.groupby('SK_ID_BUREAU').agg(status_aggregations)
    bb_agg_one_year_auto.columns = pd.Index(['ONE_YEAR_' + e[0] + "_" + e[1].upper() for e in bb_agg_one_year_auto.columns.tolist()])
    bb_agg_one_year = bb_agg_one_year.join(bb_agg_one_year_auto, on='SK_ID_BUREAU', how='left')


    bb_agg_half_year_auto = bb_half_year.groupby('SK_ID_BUREAU').agg(status_aggregations)
    bb_agg_half_year_auto.columns = pd.Index(['HALF_YEAR_' + e[0] + "_" + e[1].upper() for e in bb_agg_half_year_auto.columns.tolist()])
    bb_agg_half_year = bb_agg_half_year.join(bb_agg_half_year_auto, on='SK_ID_BUREAU', how='left')

    del bb_agg_auto, bb_agg_one_year_auto, bb_agg_half_year_auto
    gc.collect()

    # merge all back
    bb_agg = bb_agg.join(bb_agg_one_year, on='SK_ID_BUREAU', how='left')
    bb_agg = bb_agg.join(bb_agg_half_year, on='SK_ID_BUREAU', how='left')
    del bb_agg_one_year, bb_agg_half_year, bb
    gc.collect()

    # reset bb_cat
    bb_cat = [col for col in bb_agg.columns if 'STATUS_CHANGE' not in col and 'MONTHS_BALANCE' not in col]

    bureau = bureau.join(bb_agg, on='SK_ID_BUREAU', how='left')
    bureau.drop(columns='SK_ID_BUREAU', inplace=True)
    del bb_agg
    gc.collect()    


    # procesing on DAYS related features
    bureau['DAYS_CREDIT_TO_UPDATE_RATIO'] = (bureau['DAYS_CREDIT'] - 1) / (bureau['DAYS_CREDIT_UPDATE'] - 1)
    bureau['DAYS_FACT_TO_ENDDATE_RATIO'] = bureau['DAYS_ENDDATE_FACT'] / bureau['DAYS_CREDIT_ENDDATE']
    bureau['IS_FACT_LT_ENDDATE'] = (bureau['DAYS_ENDDATE_FACT'] < bureau['DAYS_CREDIT_ENDDATE']).astype('float')
    # this means a client applied for a loan in homeCredit before his original creditBureau loan ended
    bureau['DAYS_ENDDATE_TO_DAYS_CREDIT_RATIO'] = bureau['DAYS_CREDIT_ENDDATE'] / bureau['DAYS_CREDIT']
    bureau['DAYS_OVERDUE_TO_CREDIT_RATIO'] = -1 * bureau['CREDIT_DAY_OVERDUE'] / bureau['DAYS_CREDIT']
    bureau['IS_ENDDATE_GT_CREDIT'] = (bureau['DAYS_CREDIT_ENDDATE'] > bureau['DAYS_CREDIT']).astype('float')

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

    bureau['IS_DEBT_NEG'] = (bureau['AMT_CREDIT_SUM_DEBT'] < 0).astype('float')
    bureau['IS_LIMIT_NEG'] = (bureau['AMT_CREDIT_SUM_DEBT'] < 0).astype('float')

    # CREDIT TYPE DIVERSITY
    bureau_agg = bureau.groupby('SK_ID_CURR')[['CREDIT_TYPE']]\
                    .nunique().rename(columns={'CREDIT_TYPE': 'BURO_CREDIT_TYPE_CNT'})

    bureau_agg['BURO_TOTAL_CREDIT_CNT'] = bureau.groupby('SK_ID_CURR')['CREDIT_TYPE'].size()
    bureau_agg['BURO_CREDIT_TYPE_DIVERSITY'] = bureau_agg['BURO_TOTAL_CREDIT_CNT'] / bureau_agg['BURO_CREDIT_TYPE_CNT']

    # LOAN FREQUENCY
    temp = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].groupby('SK_ID_CURR')
    temp = temp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(drop=True)

    temp['DAYS_CREDIT'] = -1 * temp['DAYS_CREDIT']
    bureau_agg['DAYS_DIFF'] = temp.groupby('SK_ID_CURR')['DAYS_CREDIT'].diff()


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
        'DAYS_CREDIT_TO_UPDATE_RATIO': ['mean', 'min', 'max'],
        'DAYS_FACT_TO_ENDDATE_RATIO': ['mean', 'min', 'max'],
        'IS_FACT_LT_ENDDATE': ['mean', 'sum'],

        'DAYS_ENDDATE_TO_DAYS_CREDIT_RATIO': ['mean', 'min', 'max'],
        'DAYS_OVERDUE_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'IS_ENDDATE_GT_CREDIT': ['mean', 'sum'],

        'IS_END_IN_FUTURE': ['mean', 'sum'],
        'IS_OLD_UPDATE': ['mean', 'sum'],

        # Manual Features on AMT
        'AMT_OVERDUE_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'AMT_OVERDUE_TO_DEBT_RATIO': ['mean', 'min', 'max'],
        'AMT_DEBT_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'AMT_LIMIT_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'AMT_MAX_OVERDUE_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'AMT_ANNUITY_TO_CREDIT_RATIO': ['mean', 'min', 'max'],

        'IS_DEBT_NEG': ['mean', 'sum'],
        'IS_LIMIT_NEG': ['mean', 'sum'],

        # Numerical Features come from bureau_balance
        'STATUS_CHANGE': ['sum', 'mean'],
        'ONE_YEAR_STATUS_CHANGE': ['sum', 'mean'],
        'HALF_YEAR_STATUS_CHANGE': ['sum', 'mean'],

        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean']
    for cat in bb_cat:
        cat_aggregations[cat] = ['mean', 'median']

    bureau_agg_auto = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg_auto.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg_auto.columns.tolist()])
    bureau_agg = bureau_agg.join(bureau_agg_auto, how='left', on='SK_ID_CURR')
    del bureau_agg_auto
    gc.collect()

    # Bureau: Active credits
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg,

    # Bureau: future credits
    future = bureau[bureau['IS_END_IN_FUTURE'] == 1]
    future_agg = future.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    future_agg.columns = pd.Index(['FUTURE_' + e[0] + "_" + e[1].upper() for e in future_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(future_agg, how='left', on='SK_ID_CURR')
    del future, future_agg

    gc.collect()
    return bureau_agg


def previous_applications(prev, nan_as_category=True):
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev['FLAG_LAST_APPL_PER_CONTRACT'], _ = pd.factorize(prev['FLAG_LAST_APPL_PER_CONTRACT'])

    # handle on AMT features
    prev['APP_TO_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['ANNUITY_TO_CREDIT_RATIO'] = prev['AMT_ANNUITY'] / prev['AMT_CREDIT']
    # there is still some tiny difference between DOWN_TO_CREDIT & RATE_DOWN_PAYMENT
    # so we just keep it
    prev['DOWN_TO_CREDIT_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']
    prev['PRICE_TO_CREDIT_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_CREDIT']
    prev['PRICE_TO_APP_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_APPLICATION']
    prev['DOWN_TO_APP_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_APPLICATION']

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
    prev['AVG_TOTAL_PAYMENT_BY_DAY'] = prev['AVG_ANNUITY_BY_DAY'] / prev['AVG_PAYMENT_BY_DAY']

    # OTHER
    prev['IS_SELLERPLACE_AREA_MINUS_1'] = (prev['SELLERPLACE_AREA'] == -1).astype('float')
    prev['IS_SELLERPLACE_AREA_ZERO'] = (prev['SELLERPLACE_AREA'] == 0).astype('float')

    def app_diversity_on_cate_cols(df, process_info):
        ret = df.groupby('SK_ID_CURR')['SK_ID_PREV'].count().\
            reset_index().\
            rename(index=str, columns={'SK_ID_PREV': 'NEW_USR_APP_CNT'})

        for col_name in process_info:
            new_col_name = 'NEW_N_UNIQUE_ON_' + col_name
            gby = df.groupby('SK_ID_CURR')[col_name].nunique().\
                reset_index().\
                rename(index=str, columns={col_name: new_col_name})
            ret = ret.merge(gby, on='SK_ID_CURR', how='left')
            ret['NEW_USR_APP_DIVERSITY_ON_' +
                col_name] = ret['NEW_USR_APP_CNT'] / ret[new_col_name]

        return ret

    diversity_df = app_diversity_on_cate_cols(
        prev, [col for col in prev.columns if prev[col].dtype == 'object'])
    prev, cat_cols = one_hot_encoding(prev, nan_as_category)

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

        'DAYS_FIRST_DRAWING': ['min', 'max'],
        'DAYS_FIRST_DUE': ['min', 'max'],
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max'],
        'DAYS_LAST_DUE': ['min', 'max'],
        'DAYS_TERMINATION': ['min', 'max'],

        # manual features
        'APP_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'ANNUITY_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'DOWN_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'PRICE_TO_CREDIT_RATIO': ['mean', 'min', 'max'],
        'PRICE_TO_APP_RATIO': ['mean', 'min', 'max'],
        'DOWN_TO_APP_RATIO': ['mean', 'min', 'max'],

        'AVG_PAYMENT_AMT_CREDIT': ['mean', 'min', 'max'],
        'AVG_PAYMENT_AMT_ANNUITY': ['mean', 'min', 'max'],
        'AVG_PAYMENT_TOTAL': ['mean', 'min', 'max'],

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

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean', 'sum']
    prev_agg = prev.groupby('SK_ID_CURR').agg(
        {**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(
        ['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # add the diversity features
    prev_agg = prev_agg.merge(diversity_df, on='SK_ID_CURR', how='left')

    # Previous Applications: Approved Applications
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    approved_agg.columns = pd.Index(
        ['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    refused_agg.columns = pd.Index(
        ['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev, diversity_df
    gc.collect()
    return prev_agg.set_index('SK_ID_CURR')


def pos_cash(pos, nan_as_category=True):
    pos, cat_cols = one_hot_encoding(pos, nan_as_category)
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
    'TIME_SPAN' : ['min', 'max', 'mean', 'sum'],
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



# Preprocess credit_card_balance.csv

def credit_card_balance(cc, nan_as_category=True):
    # number of loans
    cc_agg = cc.groupby('SK_ID_CURR')['SK_ID_PREV'].\
                nunique().\
                reset_index().\
                rename(index = str, columns = {'SK_ID_PREV': 'CC_USR_LOAN_CNT'})
    cc_agg.set_index('SK_ID_CURR', inplace=True)
    # number of credit balance records
    cc_agg['CC_BLANCE_REC_CNT'] = cc.groupby('SK_ID_CURR').size()
    cc_agg['CC_PAYBACK_TIMES_MAX'] = cc.groupby('SK_ID_CURR')['CNT_INSTALMENT_MATURE_CUM'].max()

    # handle on payback times
    temp = cc.groupby(['SK_ID_PREV', 'SK_ID_CURR'])['CNT_INSTALMENT_MATURE_CUM'].\
                max().\
                reset_index().\
                rename(index=str, columns={'CNT_INSTALMENT_MATURE_CUM': 'INSTALLMENT_TIMES_PER_LOAN'})
    cc_agg['CC_PAYBACK_TIMES_TOTAL'] = temp.groupby('SK_ID_CURR')['INSTALLMENT_TIMES_PER_LOAN'].sum()
    cc_agg['CC_AVG_PAYBACK_TIMES'] = cc_agg['CC_PAYBACK_TIMES_TOTAL'] / cc_agg['CC_USR_LOAN_CNT']

    # handle on DPD
    cc_agg['CC_DPD_MAX'] = cc.groupby('SK_ID_CURR')['SK_DPD'].max()
    cc['SK_DPD_GT_ZERO'] = cc['SK_DPD'] > 0
    temp = cc.groupby(['SK_ID_PREV', 'SK_ID_CURR'])['SK_DPD_GT_ZERO'].\
                sum().\
                reset_index().\
                rename(index=str, columns={'SK_DPD_GT_ZERO': 'CNT_DPD_PER_LOAN'})
    cc_agg['CC_USR_AVG_DPD_CNT'] = temp.groupby('SK_ID_CURR')['CNT_DPD_PER_LOAN'].mean()
    cc_agg['CC_USR_AVG_DPD'] = cc.groupby('SK_ID_CURR')['SK_DPD'].mean()    

    # handle on DPD DEF
    cc_agg['CC_DPD_DEF_MAX'] = cc.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
    cc['SK_DPD_DEF_GT_ZERO'] = cc['SK_DPD_DEF'] > 0
    temp = cc.groupby(['SK_ID_PREV', 'SK_ID_CURR'])['SK_DPD_DEF_GT_ZERO'].\
                sum().\
                reset_index().\
                rename(index=str, columns={'SK_DPD_DEF_GT_ZERO': 'CNT_DPD_DEF_PER_LOAN'})
    cc_agg['CC_USR_AVG_DPD_DEF_CNT'] = temp.groupby('SK_ID_CURR')['CNT_DPD_DEF_PER_LOAN'].mean()
    cc_agg['CC_USR_AVG_DPD_DEF'] = cc.groupby('SK_ID_CURR')['SK_DPD_DEF'].mean()

    # the ratio of minimum installment missed
#     cc_agg['CC_USR_MINIMUM_PAYMENT_MISS_RATIO'] = \
#         cc.groupby('SK_ID_CURR')[['AMT_PAYMENT_CURRENT', 'AMT_INST_MIN_REGULARITY']].\
#         apply(lambda x: (cc['AMT_PAYMENT_CURRENT'] < cc['AMT_INST_MIN_REGULARITY']).sum() / len(cc))

    cc['RECIVABLE_TO_PRINCIPAL_RATIO'] = cc['AMT_RECIVABLE'] / cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['RECIVABLE_TO_TOTAL_RATIO'] = cc['AMT_RECIVABLE'] / cc['AMT_TOTAL_RECEIVABLE']
    cc['BALANCE_TO_CREDIT_RATIO'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['PAYBACK_LT_INST_MIN'] = cc['AMT_PAYMENT_CURRENT'] < cc['AMT_INST_MIN_REGULARITY']
    cc['PAYBACK_TO_INST_MIN_RATIO'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']

    num_aggregations = {
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

        # OTHER AMT & RATIO
        'AMT_BALANCE': ['max', 'mean', 'std'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean', 'std'],
        'BALANCE_TO_CREDIT_RATIO': ['max', 'mean', 'std'],
        'RECIVABLE_TO_PRINCIPAL_RATIO': ['max', 'mean', 'std'],
        'RECIVABLE_TO_TOTAL_RATIO': ['max', 'mean', 'std'],
        'PAYBACK_LT_INST_MIN': ['mean', 'sum'],
        'PAYBACK_TO_INST_MIN_RATIO': ['max', 'mean', 'std'],
    }

    cc, cat_cols = one_hot_encoding(cc, nan_as_category)
    
    cate_aggregations = {}
    for cat in cat_cols:
        cate_aggregations[cat] = ['mean']
    
    cc_agg_auto = cc.groupby('SK_ID_CURR').agg({**num_aggregations, **cate_aggregations})
    cc_agg_auto.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg_auto.columns.tolist()])
    
    # SOME ADDITIONAL OPERATION
    cc_agg_auto['CC_DRAWINGS_AMT_ATM_RATIO'] = cc_agg_auto['CC_AMT_DRAWINGS_ATM_CURRENT_SUM'] \
                                        / cc_agg_auto['CC_AMT_DRAWINGS_CURRENT_SUM']
    cc_agg_auto['CC_DRAWINGS_AMT_OTHER_RATIO'] = cc_agg_auto['CC_AMT_DRAWINGS_OTHER_CURRENT_SUM'] \
                                        / cc_agg_auto['CC_AMT_DRAWINGS_CURRENT_SUM']
    cc_agg_auto['CC_DRAWINGS_AMT_POS_RATIO'] = cc_agg_auto['CC_AMT_DRAWINGS_POS_CURRENT_SUM'] \
                                        / cc_agg_auto['CC_AMT_DRAWINGS_CURRENT_SUM']
    
    cc_agg_auto['CC_DRAWINGS_CNT_ATM_RATIO'] = cc_agg_auto['CC_CNT_DRAWINGS_ATM_CURRENT_SUM'] \
                                        / cc_agg_auto['CC_CNT_DRAWINGS_CURRENT_SUM']
    cc_agg_auto['CC_DRAWINGS_CNT_OTHER_RATIO'] = cc_agg_auto['CC_CNT_DRAWINGS_OTHER_CURRENT_SUM'] \
                                        / cc_agg_auto['CC_CNT_DRAWINGS_CURRENT_SUM']
    cc_agg_auto['CC_DRAWINGS_CNT_POS_RATIO'] = cc_agg_auto['CC_CNT_DRAWINGS_POS_CURRENT_SUM'] \
                                        / cc_agg_auto['CC_CNT_DRAWINGS_CURRENT_SUM']

    cc_agg = cc_agg.join(cc_agg_auto)
    del cc, temp, cc_agg_auto
    gc.collect()
    return cc_agg


def feature_extract(debug, input_config):
    num_rows = 30000 if debug else None

    with timer("Process applications"):
        train_file = input_config['train_filepath']
        test_file = input_config['test_filepath']
        train_df = pd.read_csv(train_file, nrows=num_rows)
        test_df = pd.read_csv(test_file, nrows=num_rows)
        print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))
        df = application_train_test(train_df, test_df)


    with timer("Process bureau and bureau_balance"):
        bureau_balance_file = input_config['bureau_balance_filepath']
        bureau_file = input_config['bureau_filepath']
        bureau = pd.read_csv(bureau_file, nrows=num_rows)
        bb = pd.read_csv(bureau_balance_file, nrows=num_rows)
        bureau = bureau_and_balance(bureau, bb)
        # with open('./notebook/bureau_agg.pkl', 'rb') as f:
        #     import pickle
        #     bureau = pickle.load(f)
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

    return df[df['TARGET'].notnull()], df[df['TARGET'].isnull()]
