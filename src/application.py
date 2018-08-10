def application_train_test(df, test_df, nan_as_category=False):
    # Read data and merge
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
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
    df['APP_ALL_DOC_CNT'] = df[doc_cols].sum(axis=1)
    df['APP_DOC3_6_8_CNT'] = df['FLAG_DOCUMENT_3'] + df['FLAG_DOCUMENT_6'] + df['FLAG_DOCUMENT_8']
    df['APP_DOC3_6_8_CNT_RAIO'] = df['APP_DOC3_6_8_CNT'] / df['APP_ALL_DOC_CNT']

    # HOUSING FEATURES
    housing_info_cols = [col for col in app.columns if '_AVG' in col or '_MODE' in col or '_MEDI' in col]
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

    df['APP_CONTACT_ALL_CNT'] = df[contact_cols].sum(axis=1)

    df['APP_DEF_TO_OBS_RATIO_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE']
    df['APP_DEF_TO_OBS_RATIO_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE']

    # REQ AMT
    df['APP_REQ_WEEK_GT_MON'] = df['AMT_REQ_CREDIT_BUREAU_WEEK'] > df['AMT_REQ_CREDIT_BUREAU_YEAR']
    df['APP_REQ_MON_GT_QRT'] = df['AMT_REQ_CREDIT_BUREAU_MON'] > df['AMT_REQ_CREDIT_BUREAU_YEAR']
    df['APP_REQ_QRT_GT_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_QRT'] > df['AMT_REQ_CREDIT_BUREAU_YEAR']
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

    del test_df
    gc.collect()
    return df