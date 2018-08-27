DROP_FEATS = \
[
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'FLAG_CONT_MOBILE',
    'FLAG_EMAIL',
    'FLAG_EMP_PHONE',
    'FLAG_MOBIL',
    'LIVE_REGION_NOT_WORK_REGION',
    'REG_REGION_NOT_LIVE_REGION',
    'APP_60_SUB_30_OBS',
    'APP_60_SUB_30_DEF',
    'APP_REQ_WEEK_GT_MON',
    'APP_REQ_MON_GT_QRT',
    'APP_REQ_QRT_GT_YEAR',
    'APP_REQ_CNT_TOTAL',
    'APP_CUM_REQ_DAY',
    'APP_CUM_REQ_WEEK',
    'BURO_CREDIT_DAY_OVERDUE_MEAN',
    'BURO_CREDIT_DAY_OVERDUE_MIN',
    'BURO_CREDIT_DAY_OVERDUE_MAX',
    'BURO_AMT_CREDIT_SUM_OVERDUE_MIN',
    'BURO_CNT_CREDIT_PROLONG_SUM',
    'BURO_CNT_CREDIT_PROLONG_MEAN',
    'BURO_CNT_CREDIT_PROLONG_MIN',
    'BURO_CNT_CREDIT_PROLONG_MAX',
    'BURO_DAYS_OVERDUE_TO_CREDIT_RATIO_MEAN',
    'BURO_DAYS_OVERDUE_TO_CREDIT_RATIO_MIN',
    'BURO_DAYS_OVERDUE_TO_CREDIT_RATIO_MAX',
    'BURO_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MEAN',
    'BURO_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MIN',
    'BURO_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MAX',
    'BURO_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MEAN',
    'BURO_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MIN',
    'BURO_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MAX',
    'BURO_IS_UNTRUSTWORTHY_MEAN',
    'BURO_IS_UNTRUSTWORTHY_SUM',
    'BURO_AMT_OVERDUE_TO_CREDIT_RATIO_MIN',
    'BURO_AMT_OVERDUE_TO_CREDIT_RATIO_MAX',
    'BURO_AMT_OVERDUE_TO_DEBT_RATIO_MEAN',
    'BURO_AMT_OVERDUE_TO_DEBT_RATIO_MIN',
    'BURO_AMT_OVERDUE_TO_DEBT_RATIO_MAX',
    'BURO_IS_DEBT_NEG_MEAN',
    'BURO_IS_DEBT_NEG_SUM',
    'BURO_IS_LIMIT_NEG_MEAN',
    'BURO_IS_LIMIT_NEG_SUM',
    'BURO_CREDIT_ACTIVE_Bad debt_MEAN',
    'BURO_CREDIT_ACTIVE_Bad debt_SUM',
    'BURO_CREDIT_ACTIVE_nan_MEAN',
    'BURO_CREDIT_ACTIVE_nan_SUM',
    'BURO_CREDIT_CURRENCY_currency 1_MEAN',
    'BURO_CREDIT_CURRENCY_currency 2_MEAN',
    'BURO_CREDIT_CURRENCY_currency 2_SUM',
    'BURO_CREDIT_CURRENCY_currency 3_MEAN',
    'BURO_CREDIT_CURRENCY_currency 3_SUM',
    'BURO_CREDIT_CURRENCY_currency 4_MEAN',
    'BURO_CREDIT_CURRENCY_currency 4_SUM',
    'BURO_CREDIT_CURRENCY_nan_MEAN',
    'BURO_CREDIT_CURRENCY_nan_SUM',
    'BURO_CREDIT_TYPE_Another type of loan_MEAN',
    'BURO_CREDIT_TYPE_Another type of loan_SUM',
    'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
    'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_SUM',
    'BURO_CREDIT_TYPE_Interbank credit_MEAN',
    'BURO_CREDIT_TYPE_Interbank credit_SUM',
    'BURO_CREDIT_TYPE_Loan for business development_MEAN',
    'BURO_CREDIT_TYPE_Loan for business development_SUM',
    'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN',
    'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_SUM',
    'BURO_CREDIT_TYPE_Loan for the purchase of equipment_MEAN',
    'BURO_CREDIT_TYPE_Loan for the purchase of equipment_SUM',
    'BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN',
    'BURO_CREDIT_TYPE_Loan for working capital replenishment_SUM',
    'BURO_CREDIT_TYPE_Mobile operator loan_MEAN',
    'BURO_CREDIT_TYPE_Mobile operator loan_SUM',
    'BURO_CREDIT_TYPE_Real estate loan_MEAN',
    'BURO_CREDIT_TYPE_Real estate loan_SUM',
    'BURO_CREDIT_TYPE_Unknown type of loan_MEAN',
    'BURO_CREDIT_TYPE_Unknown type of loan_SUM',
    'BURO_CREDIT_TYPE_nan_MEAN',
    'BURO_CREDIT_TYPE_nan_SUM',
    'BURO_LATEST_STATUS_1_SUM',
    'BURO_LATEST_STATUS_2_MEAN',
    'BURO_LATEST_STATUS_2_SUM',
    'BURO_LATEST_STATUS_3_MEAN',
    'BURO_LATEST_STATUS_3_SUM',
    'BURO_LATEST_STATUS_4_MEAN',
    'BURO_LATEST_STATUS_4_SUM',
    'BURO_LATEST_STATUS_5_MEAN',
    'BURO_LATEST_STATUS_5_SUM',
    'BURO_LATEST_STATUS_nan_MEAN',
    'BURO_LATEST_STATUS_nan_SUM',
    'BURO_STATUS_2_MEAN_SUM',
    'BURO_STATUS_2_SUM_SUM',
    'BURO_STATUS_3_MEAN_MEAN',
    'BURO_STATUS_3_MEAN_SUM',
    'BURO_STATUS_3_SUM_MEAN',
    'BURO_STATUS_3_SUM_SUM',
    'BURO_STATUS_4_MEAN_MEAN',
    'BURO_STATUS_4_MEAN_SUM',
    'BURO_STATUS_4_SUM_MEAN',
    'BURO_STATUS_4_SUM_SUM',
    'BURO_STATUS_5_MEAN_MEAN',
    'BURO_STATUS_5_MEAN_SUM',
    'BURO_STATUS_5_SUM_MEAN',
    'BURO_STATUS_5_SUM_SUM',
    'BURO_STATUS_nan_MEAN_MEAN',
    'BURO_STATUS_nan_MEAN_SUM',
    'BURO_STATUS_nan_SUM_MEAN',
    'BURO_STATUS_nan_SUM_SUM',
    'BURO_AVG_OVERDUE_ON_BUREAU_BALANCE_REC',
    'BURO_AVG_PROLONG_ANNUITY',
    'BURO_AVG_PROLONG_CREDIT_OVERDUE',
    'ACTIVE_CREDIT_DAY_OVERDUE_MEAN',
    'ACTIVE_CREDIT_DAY_OVERDUE_MIN',
    'ACTIVE_CREDIT_DAY_OVERDUE_MAX',
    'ACTIVE_DAYS_ENDDATE_FACT_MEAN',
    'ACTIVE_DAYS_ENDDATE_FACT_MIN',
    'ACTIVE_DAYS_ENDDATE_FACT_MAX',
    'ACTIVE_AMT_CREDIT_SUM_OVERDUE_MIN',
    'ACTIVE_CNT_CREDIT_PROLONG_SUM',
    'ACTIVE_CNT_CREDIT_PROLONG_MEAN',
    'ACTIVE_CNT_CREDIT_PROLONG_MIN',
    'ACTIVE_CNT_CREDIT_PROLONG_MAX',
    'ACTIVE_DAYS_FACT_SUB_ENDDATE_MEAN',
    'ACTIVE_DAYS_FACT_SUB_ENDDATE_MIN',
    'ACTIVE_DAYS_FACT_SUB_ENDDATE_MAX',
    'ACTIVE_DAYS_FACT_SUB_ENDDATE_SUM',
    'ACTIVE_DAYS_FACT_SUB_UPDATE_MEAN',
    'ACTIVE_DAYS_FACT_SUB_UPDATE_MIN',
    'ACTIVE_DAYS_FACT_SUB_UPDATE_MAX',
    'ACTIVE_DAYS_FACT_SUB_UPDATE_SUM',
    'ACTIVE_DAYS_FACT_TO_UPDATE_RATIO_MEAN',
    'ACTIVE_DAYS_FACT_TO_UPDATE_RATIO_MIN',
    'ACTIVE_DAYS_FACT_TO_UPDATE_RATIO_MAX',
    'ACTIVE_IS_EARLY_PAID_MEAN',
    'ACTIVE_IS_EARLY_PAID_SUM',
    'ACTIVE_IS_LATER_PAID_MEAN',
    'ACTIVE_IS_LATER_PAID_SUM',
    'ACTIVE_PLAN_TIME_SPAN_MEAN',
    'ACTIVE_PLAN_TIME_SPAN_MIN',
    'ACTIVE_PLAN_TIME_SPAN_MAX',
    'ACTIVE_PLAN_TIME_SPAN_SUM',
    'ACTIVE_ACTUAL_TIME_SPAN_TO_PLAN_RATIO_MEAN',
    'ACTIVE_ACTUAL_TIME_SPAN_TO_PLAN_RATIO_MIN',
    'ACTIVE_ACTUAL_TIME_SPAN_TO_PLAN_RATIO_MAX',
    'ACTIVE_DAYS_FACT_SUB_UPDATE_TO_ACTUAL_TIME_SPAN_RATIO_MEAN',
    'ACTIVE_DAYS_FACT_SUB_UPDATE_TO_ACTUAL_TIME_SPAN_RATIO_MIN',
    'ACTIVE_DAYS_FACT_SUB_UPDATE_TO_ACTUAL_TIME_SPAN_RATIO_MAX',
    'ACTIVE_DAYS_ENDDATE_SUB_UPDATE_TO_PLAN_TIME_SPAN_RATIO_MEAN',
    'ACTIVE_DAYS_ENDDATE_SUB_UPDATE_TO_PLAN_TIME_SPAN_RATIO_MIN',
    'ACTIVE_DAYS_ENDDATE_SUB_UPDATE_TO_PLAN_TIME_SPAN_RATIO_MAX',
    'ACTIVE_DAYS_OVERDUE_TO_CREDIT_RATIO_MEAN',
    'ACTIVE_DAYS_OVERDUE_TO_CREDIT_RATIO_MIN',
    'ACTIVE_DAYS_OVERDUE_TO_CREDIT_RATIO_MAX',
    'ACTIVE_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MEAN',
    'ACTIVE_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MIN',
    'ACTIVE_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MAX',
    'ACTIVE_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MEAN',
    'ACTIVE_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MIN',
    'ACTIVE_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MAX',
    'ACTIVE_IS_UNTRUSTWORTHY_MEAN',
    'ACTIVE_IS_UNTRUSTWORTHY_SUM',
    'ACTIVE_AMT_OVERDUE_TO_CREDIT_RATIO_MEAN',
    'ACTIVE_AMT_OVERDUE_TO_CREDIT_RATIO_MIN',
    'ACTIVE_AMT_OVERDUE_TO_CREDIT_RATIO_MAX',
    'ACTIVE_AMT_OVERDUE_TO_DEBT_RATIO_MIN',
    'ACTIVE_IS_DEBT_NEG_MEAN',
    'ACTIVE_IS_DEBT_NEG_SUM',
    'ACTIVE_IS_LIMIT_NEG_MEAN',
    'ACTIVE_IS_LIMIT_NEG_SUM',
    'CLOSED_CREDIT_DAY_OVERDUE_MEAN',
    'CLOSED_CREDIT_DAY_OVERDUE_MIN',
    'CLOSED_CREDIT_DAY_OVERDUE_MAX',
    'CLOSED_AMT_CREDIT_SUM_DEBT_MIN',
    'CLOSED_AMT_CREDIT_SUM_OVERDUE_SUM',
    'CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN',
    'CLOSED_AMT_CREDIT_SUM_OVERDUE_MIN',
    'CLOSED_AMT_CREDIT_SUM_OVERDUE_MAX',
    'CLOSED_AMT_CREDIT_SUM_LIMIT_MIN',
    'CLOSED_CNT_CREDIT_PROLONG_SUM',
    'CLOSED_CNT_CREDIT_PROLONG_MEAN',
    'CLOSED_CNT_CREDIT_PROLONG_MIN',
    'CLOSED_CNT_CREDIT_PROLONG_MAX',
    'CLOSED_DAYS_OVERDUE_TO_CREDIT_RATIO_MEAN',
    'CLOSED_DAYS_OVERDUE_TO_CREDIT_RATIO_MIN',
    'CLOSED_DAYS_OVERDUE_TO_CREDIT_RATIO_MAX',
    'CLOSED_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MEAN',
    'CLOSED_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MIN',
    'CLOSED_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MAX',
    'CLOSED_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MEAN',
    'CLOSED_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MIN',
    'CLOSED_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MAX',
    'CLOSED_IS_UNTRUSTWORTHY_MEAN',
    'CLOSED_IS_UNTRUSTWORTHY_SUM',
    'CLOSED_IS_END_IN_FUTURE_SUM',
    'CLOSED_AMT_OVERDUE_TO_CREDIT_RATIO_MEAN',
    'CLOSED_AMT_OVERDUE_TO_CREDIT_RATIO_MIN',
    'CLOSED_AMT_OVERDUE_TO_CREDIT_RATIO_MAX',
    'CLOSED_AMT_OVERDUE_TO_DEBT_RATIO_MEAN',
    'CLOSED_AMT_OVERDUE_TO_DEBT_RATIO_MIN',
    'CLOSED_AMT_OVERDUE_TO_DEBT_RATIO_MAX',
    'CLOSED_AMT_DEBT_TO_CREDIT_RATIO_MIN',
    'CLOSED_AMT_LIMIT_TO_CREDIT_RATIO_MIN',
    'CLOSED_AMT_ANNUITY_TO_DEBT_RATIO_MEAN',
    'CLOSED_AMT_ANNUITY_TO_DEBT_RATIO_MIN',
    'CLOSED_AMT_ANNUITY_TO_DEBT_RATIO_MAX',
    'CLOSED_AVG_DEBT_BY_MONTH_MEAN',
    'CLOSED_AVG_DEBT_BY_MONTH_MIN',
    'CLOSED_AVG_DEBT_BY_MONTH_MAX',
    'CLOSED_AVG_LIMIT_BY_MONTH_MEAN',
    'CLOSED_AVG_LIMIT_BY_MONTH_MIN',
    'CLOSED_IS_DEBT_NEG_MEAN',
    'CLOSED_IS_DEBT_NEG_SUM',
    'CLOSED_IS_LIMIT_NEG_MEAN',
    'CLOSED_IS_LIMIT_NEG_SUM',
    'FUTURE_CREDIT_DAY_OVERDUE_MEAN',
    'FUTURE_CREDIT_DAY_OVERDUE_MIN',
    'FUTURE_CREDIT_DAY_OVERDUE_MAX',
    'FUTURE_AMT_CREDIT_SUM_OVERDUE_MIN',
    'FUTURE_CNT_CREDIT_PROLONG_SUM',
    'FUTURE_CNT_CREDIT_PROLONG_MIN',
    'FUTURE_CNT_CREDIT_PROLONG_MAX',
    'FUTURE_IS_LATER_PAID_MEAN',
    'FUTURE_IS_LATER_PAID_SUM',
    'FUTURE_DAYS_OVERDUE_TO_CREDIT_RATIO_MEAN',
    'FUTURE_DAYS_OVERDUE_TO_CREDIT_RATIO_MIN',
    'FUTURE_DAYS_OVERDUE_TO_CREDIT_RATIO_MAX',
    'FUTURE_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MEAN',
    'FUTURE_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MIN',
    'FUTURE_DAYS_OVERDUE_TO_PLAN_TIME_SPAN_RATIO_MAX',
    'FUTURE_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MEAN',
    'FUTURE_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MIN',
    'FUTURE_DAYS_OVERDUE_TO_ACTUAL_TIME_SPAN_RATIO_MAX',
    'FUTURE_IS_UNTRUSTWORTHY_MEAN',
    'FUTURE_IS_UNTRUSTWORTHY_SUM',
    'FUTURE_IS_END_IN_FUTURE_MEAN',
    'FUTURE_AMT_OVERDUE_TO_CREDIT_RATIO_MIN',
    'FUTURE_AMT_OVERDUE_TO_CREDIT_RATIO_MAX',
    'FUTURE_AMT_OVERDUE_TO_DEBT_RATIO_MEAN',
    'FUTURE_AMT_OVERDUE_TO_DEBT_RATIO_MIN',
    'FUTURE_IS_DEBT_NEG_MEAN',
    'FUTURE_IS_DEBT_NEG_SUM',
    'FUTURE_IS_LIMIT_NEG_MEAN',
    'FUTURE_IS_LIMIT_NEG_SUM',
    'FUTURE_MONTHS_BALANCE_MAX_MAX',
    'PREV_FLAG_LAST_APPL_PER_CONTRACT_MEAN',
    'PREV_FLAG_LAST_APPL_PER_CONTRACT_SUM',
    'PREV_NFLAG_LAST_APPL_IN_DAY_MEAN',
    'PREV_RATE_INTEREST_PRIMARY_MEAN',
    'PREV_RATE_INTEREST_PRIMARY_MIN',
    'PREV_RATE_INTEREST_PRIMARY_MAX',
    'PREV_RATE_INTEREST_PRIVILEGED_MEAN',
    'PREV_RATE_INTEREST_PRIVILEGED_MIN',
    'PREV_RATE_INTEREST_PRIVILEGED_MAX',
    'PREV_APP_TO_PRICE_RATIO_MEAN',
    'PREV_APP_TO_PRICE_RATIO_MIN',
    'PREV_APP_TO_PRICE_RATIO_MAX',
    'PREV_IS_LATER_PAID_MEAN',
    'PREV_IS_LATER_PAID_SUM',
    'PREV_IS_FISRT_DRAWING_LATER_THAN_LAST_DUE_MEAN',
    'PREV_IS_FISRT_DRAWING_LATER_THAN_LAST_DUE_SUM',
    'PREV_IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE_MEAN',
    'PREV_IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE_SUM',
    'PREV_NAME_CONTRACT_TYPE_XNA_MEAN',
    'PREV_NAME_CONTRACT_TYPE_XNA_SUM',
    'PREV_NAME_CONTRACT_TYPE_nan_MEAN',
    'PREV_NAME_CONTRACT_TYPE_nan_SUM',
    'PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN',
    'PREV_WEEKDAY_APPR_PROCESS_START_nan_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Business development_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a used car_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a used car_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Education_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Education_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Furniture_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Hobby_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Journey_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Medicine_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Other_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Payments on other loans_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Payments on other loans_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Repairs_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_SUM',
    'PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_nan_SUM',
    'PREV_NAME_CONTRACT_STATUS_Unused offer_SUM',
    'PREV_NAME_CONTRACT_STATUS_nan_MEAN',
    'PREV_NAME_CONTRACT_STATUS_nan_SUM',
    'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN',
    'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_SUM',
    'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN',
    'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_SUM',
    'PREV_NAME_PAYMENT_TYPE_nan_MEAN',
    'PREV_NAME_PAYMENT_TYPE_nan_SUM',
    'PREV_CODE_REJECT_REASON_CLIENT_MEAN',
    'PREV_CODE_REJECT_REASON_CLIENT_SUM',
    'PREV_CODE_REJECT_REASON_SYSTEM_MEAN',
    'PREV_CODE_REJECT_REASON_SYSTEM_SUM',
    'PREV_CODE_REJECT_REASON_VERIF_MEAN',
    'PREV_CODE_REJECT_REASON_VERIF_SUM',
    'PREV_CODE_REJECT_REASON_XNA_MEAN',
    'PREV_CODE_REJECT_REASON_XNA_SUM',
    'PREV_CODE_REJECT_REASON_nan_MEAN',
    'PREV_CODE_REJECT_REASON_nan_SUM',
    'PREV_NAME_TYPE_SUITE_Group of people_MEAN',
    'PREV_NAME_TYPE_SUITE_Group of people_SUM',
    'PREV_NAME_TYPE_SUITE_Other_A_MEAN',
    'PREV_NAME_TYPE_SUITE_Other_A_SUM',
    'PREV_NAME_TYPE_SUITE_Other_B_SUM',
    'PREV_NAME_CLIENT_TYPE_XNA_MEAN',
    'PREV_NAME_CLIENT_TYPE_XNA_SUM',
    'PREV_NAME_CLIENT_TYPE_nan_MEAN',
    'PREV_NAME_CLIENT_TYPE_nan_SUM',
    'PREV_NAME_GOODS_CATEGORY_Additional Service_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Additional Service_SUM',
    'PREV_NAME_GOODS_CATEGORY_Animals_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Animals_SUM',
    'PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Auto Accessories_SUM',
    'PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_SUM',
    'PREV_NAME_GOODS_CATEGORY_Direct Sales_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Direct Sales_SUM',
    'PREV_NAME_GOODS_CATEGORY_Education_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Education_SUM',
    'PREV_NAME_GOODS_CATEGORY_Fitness_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Fitness_SUM',
    'PREV_NAME_GOODS_CATEGORY_Gardening_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Gardening_SUM',
    'PREV_NAME_GOODS_CATEGORY_Homewares_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Homewares_SUM',
    'PREV_NAME_GOODS_CATEGORY_House Construction_MEAN',
    'PREV_NAME_GOODS_CATEGORY_House Construction_SUM',
    'PREV_NAME_GOODS_CATEGORY_Insurance_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Insurance_SUM',
    'PREV_NAME_GOODS_CATEGORY_Jewelry_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Jewelry_SUM',
    'PREV_NAME_GOODS_CATEGORY_Medical Supplies_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Medical Supplies_SUM',
    'PREV_NAME_GOODS_CATEGORY_Medicine_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Medicine_SUM',
    'PREV_NAME_GOODS_CATEGORY_Office Appliances_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Office Appliances_SUM',
    'PREV_NAME_GOODS_CATEGORY_Other_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Other_SUM',
    'PREV_NAME_GOODS_CATEGORY_Tourism_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Tourism_SUM',
    'PREV_NAME_GOODS_CATEGORY_Vehicles_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Vehicles_SUM',
    'PREV_NAME_GOODS_CATEGORY_Weapon_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Weapon_SUM',
    'PREV_NAME_GOODS_CATEGORY_nan_MEAN',
    'PREV_NAME_GOODS_CATEGORY_nan_SUM',
    'PREV_NAME_PORTFOLIO_Cars_MEAN',
    'PREV_NAME_PORTFOLIO_Cars_SUM',
    'PREV_NAME_PORTFOLIO_nan_MEAN',
    'PREV_NAME_PORTFOLIO_nan_SUM',
    'PREV_NAME_PRODUCT_TYPE_nan_MEAN',
    'PREV_NAME_PRODUCT_TYPE_nan_SUM',
    'PREV_CHANNEL_TYPE_Car dealer_MEAN',
    'PREV_CHANNEL_TYPE_Car dealer_SUM',
    'PREV_CHANNEL_TYPE_nan_MEAN',
    'PREV_CHANNEL_TYPE_nan_SUM',
    'PREV_NAME_SELLER_INDUSTRY_Clothing_SUM',
    'PREV_NAME_SELLER_INDUSTRY_Construction_SUM',
    'PREV_NAME_SELLER_INDUSTRY_Furniture_SUM',
    'PREV_NAME_SELLER_INDUSTRY_Industry_SUM',
    'PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Jewelry_SUM',
    'PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_MLM partners_SUM',
    'PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Tourism_SUM',
    'PREV_NAME_SELLER_INDUSTRY_nan_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_nan_SUM',
    'PREV_NAME_YIELD_GROUP_nan_MEAN',
    'PREV_NAME_YIELD_GROUP_nan_SUM',
    'PREV_PRODUCT_COMBINATION_Card X-Sell_SUM',
    'PREV_PRODUCT_COMBINATION_POS mobile without interest_SUM',
    'PREV_PRODUCT_COMBINATION_POS other with interest_SUM',
    'PREV_PRODUCT_COMBINATION_POS others without interest_MEAN',
    'PREV_PRODUCT_COMBINATION_POS others without interest_SUM',
    'PREV_PRODUCT_COMBINATION_nan_MEAN',
    'PREV_PRODUCT_COMBINATION_nan_SUM',
    'PREV_MEAN_ON_APP_TO_PRICE_RATIO_BY_[ NAME_CONTRACT_TYPE, NAME_PAYMENT_TYPE, NAME_PORTFOLIO, NAME_YIELD_GROUP, NAME_CONTRACT_STATUS, NAME_PRODUCT_TYPE ]_MEDIAN',
    'PREV_MEAN_ON_APP_TO_PRICE_RATIO_BY_[ NAME_CASH_LOAN_PURPOSE, NAME_GOODS_CATEGORY, CHANNEL_TYPE, NAME_SELLER_INDUSTRY, NAME_CONTRACT_STATUS, NAME_PRODUCT_TYPE ]_MEDIAN',
    'PREV_MEAN_ON_APP_TO_PRICE_RATIO_BY_[ PRODUCT_COMBINATION, NAME_GOODS_CATEGORY, NAME_CASH_LOAN_PURPOSE, NAME_CONTRACT_STATUS ]_MEDIAN',
    'APPROVED_FLAG_LAST_APPL_PER_CONTRACT_MEAN',
    'APPROVED_FLAG_LAST_APPL_PER_CONTRACT_SUM',
    'APPROVED_NFLAG_LAST_APPL_IN_DAY_MEAN',
    'APPROVED_RATE_INTEREST_PRIMARY_MEAN',
    'APPROVED_RATE_INTEREST_PRIMARY_MIN',
    'APPROVED_RATE_INTEREST_PRIMARY_MAX',
    'APPROVED_RATE_INTEREST_PRIVILEGED_MEAN',
    'APPROVED_RATE_INTEREST_PRIVILEGED_MIN',
    'APPROVED_RATE_INTEREST_PRIVILEGED_MAX',
    'APPROVED_APP_TO_PRICE_RATIO_MEAN',
    'APPROVED_APP_TO_PRICE_RATIO_MIN',
    'APPROVED_APP_TO_PRICE_RATIO_MAX',
    'APPROVED_IS_LATER_PAID_MEAN',
    'APPROVED_IS_LATER_PAID_SUM',
    'APPROVED_IS_FISRT_DRAWING_LATER_THAN_LAST_DUE_MEAN',
    'APPROVED_IS_FISRT_DRAWING_LATER_THAN_LAST_DUE_SUM',
    'APPROVED_IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE_MEAN',
    'APPROVED_IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE_SUM',
    'APPROVED_IS_SELLERPLACE_AREA_ZERO_MEAN',
    'APPROVED_IS_SELLERPLACE_AREA_ZERO_SUM',
    'REFUSED_FLAG_LAST_APPL_PER_CONTRACT_MEAN',
    'REFUSED_FLAG_LAST_APPL_PER_CONTRACT_SUM',
    'REFUSED_NFLAG_LAST_APPL_IN_DAY_MEAN',
    'REFUSED_NFLAG_INSURED_ON_APPROVAL_MEAN',
    'REFUSED_NFLAG_INSURED_ON_APPROVAL_SUM',
    'REFUSED_RATE_INTEREST_PRIMARY_MEAN',
    'REFUSED_RATE_INTEREST_PRIMARY_MIN',
    'REFUSED_RATE_INTEREST_PRIMARY_MAX',
    'REFUSED_RATE_INTEREST_PRIVILEGED_MEAN',
    'REFUSED_RATE_INTEREST_PRIVILEGED_MIN',
    'REFUSED_RATE_INTEREST_PRIVILEGED_MAX',
    'REFUSED_DAYS_FIRST_DRAWING_MIN',
    'REFUSED_DAYS_FIRST_DRAWING_MAX',
    'REFUSED_DAYS_FIRST_DUE_MIN',
    'REFUSED_DAYS_FIRST_DUE_MAX',
    'REFUSED_DAYS_LAST_DUE_1ST_VERSION_MIN',
    'REFUSED_DAYS_LAST_DUE_1ST_VERSION_MAX',
    'REFUSED_DAYS_LAST_DUE_MIN',
    'REFUSED_DAYS_LAST_DUE_MAX',
    'REFUSED_DAYS_TERMINATION_MIN',
    'REFUSED_DAYS_TERMINATION_MAX',
    'REFUSED_APP_TO_PRICE_RATIO_MEAN',
    'REFUSED_APP_TO_PRICE_RATIO_MIN',
    'REFUSED_APP_TO_PRICE_RATIO_MAX',
    'REFUSED_PLAN_TIME_SPAN_MEAN',
    'REFUSED_PLAN_TIME_SPAN_SUM',
    'REFUSED_PLAN_TIME_SPAN_MIN',
    'REFUSED_PLAN_TIME_SPAN_MAX',
    'REFUSED_ACTUAL_TIME_SPAN_MEAN',
    'REFUSED_ACTUAL_TIME_SPAN_SUM',
    'REFUSED_ACTUAL_TIME_SPAN_MIN',
    'REFUSED_ACTUAL_TIME_SPAN_MAX',
    'REFUSED_LAST_DUE_DIFF_MEAN',
    'REFUSED_LAST_DUE_DIFF_SUM',
    'REFUSED_LAST_DUE_DIFF_MIN',
    'REFUSED_LAST_DUE_DIFF_MAX',
    'REFUSED_ACTUAL_TIME_SPAN_TO_PLAN_RATIO_MEAN',
    'REFUSED_ACTUAL_TIME_SPAN_TO_PLAN_RATIO_MIN',
    'REFUSED_ACTUAL_TIME_SPAN_TO_PLAN_RATIO_MAX',
    'REFUSED_DAYS_DESICION_TO_FTRST_DUE_RATIO_MEAN',
    'REFUSED_DAYS_DESICION_TO_FTRST_DUE_RATIO_MIN',
    'REFUSED_DAYS_DESICION_TO_FTRST_DUE_RATIO_MAX',
    'REFUSED_DAYS_TERMINATION_SUB_LAST_DUE_MEAN',
    'REFUSED_DAYS_TERMINATION_SUB_LAST_DUE_MIN',
    'REFUSED_DAYS_TERMINATION_SUB_LAST_DUE_MAX',
    'REFUSED_IS_EARLY_PAID_MEAN',
    'REFUSED_IS_EARLY_PAID_SUM',
    'REFUSED_IS_LATER_PAID_MEAN',
    'REFUSED_IS_LATER_PAID_SUM',
    'REFUSED_IS_FISRT_DRAWING_LATER_THAN_LAST_DUE_MEAN',
    'REFUSED_IS_FISRT_DRAWING_LATER_THAN_LAST_DUE_SUM',
    'REFUSED_IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE_MEAN',
    'REFUSED_IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE_SUM',
    'REFUSED_AVG_PAYMENT_DAYS_MEAN',
    'REFUSED_AVG_PAYMENT_DAYS_MIN',
    'REFUSED_AVG_PAYMENT_DAYS_MAX',
    'REFUSED_AVG_PAYMENT_BY_DAY_MEAN',
    'REFUSED_AVG_PAYMENT_BY_DAY_MIN',
    'REFUSED_AVG_PAYMENT_BY_DAY_MAX',
    'REFUSED_AVG_ANNUITY_BY_DAY_MEAN',
    'REFUSED_AVG_ANNUITY_BY_DAY_MIN',
    'REFUSED_AVG_ANNUITY_BY_DAY_MAX',
    'REFUSED_AVG_TOTAL_PAYMENT_BY_DAY_MEAN',
    'REFUSED_AVG_TOTAL_PAYMENT_BY_DAY_MIN',
    'REFUSED_AVG_TOTAL_PAYMENT_BY_DAY_MAX',
    'REFUSED_IS_SELLERPLACE_AREA_ZERO_MEAN',
    'REFUSED_IS_SELLERPLACE_AREA_ZERO_SUM',
    'XSELL_AMT_DOWN_PAYMENT_MEAN',
    'XSELL_AMT_DOWN_PAYMENT_SUM',
    'XSELL_AMT_DOWN_PAYMENT_MIN',
    'XSELL_AMT_DOWN_PAYMENT_MAX',
    'XSELL_FLAG_LAST_APPL_PER_CONTRACT_MEAN',
    'XSELL_FLAG_LAST_APPL_PER_CONTRACT_SUM',
    'XSELL_NFLAG_LAST_APPL_IN_DAY_MEAN',
    'XSELL_RATE_DOWN_PAYMENT_MEAN',
    'XSELL_RATE_DOWN_PAYMENT_MIN',
    'XSELL_RATE_DOWN_PAYMENT_MAX',
    'XSELL_RATE_INTEREST_PRIMARY_MEAN',
    'XSELL_RATE_INTEREST_PRIMARY_MIN',
    'XSELL_RATE_INTEREST_PRIMARY_MAX',
    'XSELL_RATE_INTEREST_PRIVILEGED_MEAN',
    'XSELL_RATE_INTEREST_PRIVILEGED_MIN',
    'XSELL_RATE_INTEREST_PRIVILEGED_MAX',
    'XSELL_APP_TO_DOWN_RATIO_MEAN',
    'XSELL_APP_TO_DOWN_RATIO_MIN',
    'XSELL_APP_TO_DOWN_RATIO_MAX',
    'XSELL_APP_TO_PRICE_RATIO_MEAN',
    'XSELL_APP_TO_PRICE_RATIO_MIN',
    'XSELL_APP_TO_PRICE_RATIO_MAX',
    'XSELL_ANNUITY_TO_DOWN_RATIO_MEAN',
    'XSELL_ANNUITY_TO_DOWN_RATIO_MIN',
    'XSELL_ANNUITY_TO_DOWN_RATIO_MAX',
    'XSELL_CREDIT_TO_DOWN_RATIO_MEAN',
    'XSELL_CREDIT_TO_DOWN_RATIO_MIN',
    'XSELL_CREDIT_TO_DOWN_RATIO_MAX',
    'XSELL_DOWN_TO_PRICE_RATIO_MEAN',
    'XSELL_DOWN_TO_PRICE_RATIO_MIN',
    'XSELL_DOWN_TO_PRICE_RATIO_MAX',
    'XSELL_IS_LATER_PAID_MEAN',
    'XSELL_IS_LATER_PAID_SUM',
    'XSELL_IS_FISRT_DRAWING_LATER_THAN_LAST_DUE_MEAN',
    'XSELL_IS_FISRT_DRAWING_LATER_THAN_LAST_DUE_SUM',
    'XSELL_IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE_MEAN',
    'XSELL_IS_FISRT_DRAWING_LATER_THAN_FIRST_DUE_SUM',
    'XSELL_IS_SELLERPLACE_AREA_ZERO_SUM',
    'LATEST_STATUS_Amortized debt',
    'LATEST_STATUS_Approved',
    'LATEST_STATUS_Demand',
    'LATEST_STATUS_Returned to the store',
    'LATEST_STATUS_Signed',
    'LATEST_STATUS_nan',
    'POS_NAME_CONTRACT_STATUS_Amortized debt_SUM',
    'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN',
    'POS_NAME_CONTRACT_STATUS_Approved_SUM',
    'POS_NAME_CONTRACT_STATUS_Approved_MEAN',
    'POS_NAME_CONTRACT_STATUS_Canceled_SUM',
    'POS_NAME_CONTRACT_STATUS_Canceled_MEAN',
    'POS_NAME_CONTRACT_STATUS_Demand_SUM',
    'POS_NAME_CONTRACT_STATUS_Demand_MEAN',
    'POS_NAME_CONTRACT_STATUS_XNA_SUM',
    'POS_NAME_CONTRACT_STATUS_XNA_MEAN',
    'POS_NAME_CONTRACT_STATUS_nan_SUM',
    'POS_NAME_CONTRACT_STATUS_nan_MEAN',
    'CC_USR_LOAN_CNT',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_MAX',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_SUM',
    'CC_SK_DPD_DEF_MAX',
    'CC_AMT_RECIVABLE_EQ_TOTAL_MEAN',
    'CC_NAME_CONTRACT_STATUS_Approved_MEAN',
    'CC_NAME_CONTRACT_STATUS_Approved_SUM',
    'CC_NAME_CONTRACT_STATUS_Completed_MEAN',
    'CC_NAME_CONTRACT_STATUS_Completed_SUM',
    'CC_NAME_CONTRACT_STATUS_Demand_MEAN',
    'CC_NAME_CONTRACT_STATUS_Demand_SUM',
    'CC_NAME_CONTRACT_STATUS_Refused_MEAN',
    'CC_NAME_CONTRACT_STATUS_Refused_SUM',
    'CC_NAME_CONTRACT_STATUS_Sent proposal_MEAN',
    'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM',
    'CC_NAME_CONTRACT_STATUS_Signed_SUM',
    'CC_NAME_CONTRACT_STATUS_nan_MEAN',
    'CC_NAME_CONTRACT_STATUS_nan_SUM',
    'AGG_FINAL_MEAN_ON_PREV_CREDIT_TO_PRICE_RATIO_MEAN_BY_[ CODE_GENDER, NAME_INCOME_TYPE, NAME_CONTRACT_TYPE ]',
    'AGG_FINAL_MEAN_ON_BURO_AMT_CREDIT_SUM_SUM_TO_LIMIT_SUM_RATIO_BY_[ CODE_GENDER, NAME_INCOME_TYPE, NAME_CONTRACT_TYPE ]',
    'AGG_FINAL_MEAN_ON_BURO_AMT_CREDIT_SUM_SUM_TO_LIMIT_SUM_RATIO_BY_[ CODE_GENDER, NAME_TYPE_SUITE, NAME_HOUSING_TYPE, NAME_FAMILY_STATUS, CNT_CHILDREN ]',
    'EMERGENCYSTATE_MODE_Yes',
    'EMERGENCYSTATE_MODE_nan',
    'FONDKAPREMONT_MODE_not specified',
    'FONDKAPREMONT_MODE_org spec account',
    'FONDKAPREMONT_MODE_reg oper account',
    'FONDKAPREMONT_MODE_reg oper spec account',
    'FONDKAPREMONT_MODE_nan',
    'HOUSETYPE_MODE_specific housing',
    'HOUSETYPE_MODE_terraced house',
    'NAME_CONTRACT_TYPE_Revolving loans',
    'NAME_CONTRACT_TYPE_nan',
    'NAME_EDUCATION_TYPE_Academic degree',
    'NAME_EDUCATION_TYPE_Incomplete higher',
    'NAME_EDUCATION_TYPE_nan',
    'NAME_FAMILY_STATUS_Unknown',
    'NAME_FAMILY_STATUS_Widow',
    'NAME_FAMILY_STATUS_nan',
    'NAME_HOUSING_TYPE_Co-op apartment',
    'NAME_HOUSING_TYPE_House / apartment',
    'NAME_HOUSING_TYPE_Rented apartment',
    'NAME_HOUSING_TYPE_With parents',
    'NAME_HOUSING_TYPE_nan',
    'NAME_INCOME_TYPE_Businessman',
    'NAME_INCOME_TYPE_Commercial associate',
    'NAME_INCOME_TYPE_Maternity leave',
    'NAME_INCOME_TYPE_Pensioner',
    'NAME_INCOME_TYPE_Student',
    'NAME_INCOME_TYPE_Unemployed',
    'NAME_INCOME_TYPE_nan',
    'NAME_TYPE_SUITE_Children',
    'NAME_TYPE_SUITE_Family',
    'NAME_TYPE_SUITE_Group of people',
    'NAME_TYPE_SUITE_Other_A',
    'NAME_TYPE_SUITE_Other_B',
    'NAME_TYPE_SUITE_nan',
    'OCCUPATION_TYPE_Accountants',
    'OCCUPATION_TYPE_Cleaning staff',
    'OCCUPATION_TYPE_Cooking staff',
    'OCCUPATION_TYPE_HR staff',
    'OCCUPATION_TYPE_IT staff',
    'OCCUPATION_TYPE_Laborers',
    'OCCUPATION_TYPE_Low-skill Laborers',
    'OCCUPATION_TYPE_Medicine staff',
    'OCCUPATION_TYPE_Private service staff',
    'OCCUPATION_TYPE_Realty agents',
    'OCCUPATION_TYPE_Secretaries',
    'OCCUPATION_TYPE_Security staff',
    'OCCUPATION_TYPE_Waiters/barmen staff',
    'OCCUPATION_TYPE_nan',
    'ORGANIZATION_TYPE_Advertising',
    'ORGANIZATION_TYPE_Agriculture',
    'ORGANIZATION_TYPE_Business Entity Type 1',
    'ORGANIZATION_TYPE_Business Entity Type 2',
    'ORGANIZATION_TYPE_Cleaning',
    'ORGANIZATION_TYPE_Culture',
    'ORGANIZATION_TYPE_Electricity',
    'ORGANIZATION_TYPE_Emergency',
    'ORGANIZATION_TYPE_Government',
    'ORGANIZATION_TYPE_Hotel',
    'ORGANIZATION_TYPE_Housing',
    'ORGANIZATION_TYPE_Industry: type 1',
    'ORGANIZATION_TYPE_Industry: type 10',
    'ORGANIZATION_TYPE_Industry: type 11',
    'ORGANIZATION_TYPE_Industry: type 12',
    'ORGANIZATION_TYPE_Industry: type 13',
    'ORGANIZATION_TYPE_Industry: type 2',
    'ORGANIZATION_TYPE_Industry: type 3',
    'ORGANIZATION_TYPE_Industry: type 4',
    'ORGANIZATION_TYPE_Industry: type 5',
    'ORGANIZATION_TYPE_Industry: type 6',
    'ORGANIZATION_TYPE_Industry: type 7',
    'ORGANIZATION_TYPE_Industry: type 8',
    'ORGANIZATION_TYPE_Insurance',
    'ORGANIZATION_TYPE_Legal Services',
    'ORGANIZATION_TYPE_Medicine',
    'ORGANIZATION_TYPE_Mobile',
    'ORGANIZATION_TYPE_Postal',
    'ORGANIZATION_TYPE_Realtor',
    'ORGANIZATION_TYPE_Religion',
    'ORGANIZATION_TYPE_Restaurant',
    'ORGANIZATION_TYPE_Security',
    'ORGANIZATION_TYPE_Services',
    'ORGANIZATION_TYPE_Telecom',
    'ORGANIZATION_TYPE_Trade: type 1',
    'ORGANIZATION_TYPE_Trade: type 2',
    'ORGANIZATION_TYPE_Trade: type 3',
    'ORGANIZATION_TYPE_Trade: type 4',
    'ORGANIZATION_TYPE_Trade: type 5',
    'ORGANIZATION_TYPE_Trade: type 6',
    'ORGANIZATION_TYPE_Trade: type 7',
    'ORGANIZATION_TYPE_Transport: type 1',
    'ORGANIZATION_TYPE_Transport: type 2',
    'ORGANIZATION_TYPE_Transport: type 4',
    'ORGANIZATION_TYPE_University',
    'ORGANIZATION_TYPE_XNA',
    'ORGANIZATION_TYPE_nan',
    'WALLSMATERIAL_MODE_Block',
    'WALLSMATERIAL_MODE_Mixed',
    'WALLSMATERIAL_MODE_Monolithic',
    'WALLSMATERIAL_MODE_Others',
    'WALLSMATERIAL_MODE_Wooden',
    'WEEKDAY_APPR_PROCESS_START_nan',
]
