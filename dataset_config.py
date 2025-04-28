"""
if specified bin columns are not in `categorical_indicator`, those
columns will be still considered as `num_cols` instead of `bin_cols`
"""
DATA_CONFIGS = {
    'credit-g': {'bin': ['own_telephone', 'foreign_worker'], 'column_map': {'age': 'applicant_age'}},
    'Click_prediction_small': None,
    'credit-approval': {'bin': ['A9', 'A10', 'A12']},
    'dresses-sales': {'column_map': {'V2': 'dress_style', 'V3': 'dress_att2', 'V4': 'dress_att3', 'V5': 'size', 'V6': 'season', 'V7': 'dress_att6', 'V8': 'dress_att7', 'V9': 'dress_att8', 'V10': 'dress_att9', 'V11': 'dress_att10', 'V12': 'dresss_att11', 'V13': 'dress_att12'}},
    'adult': {'column_map': {'age': 'adult_age'}},
    'pendigits': None,
    'steel-plates-fault': {'bin': ['V28', 'V29', 'V30', 'V31', 'V32', 'V33']},
    'SpeedDating': {'bin': ['has_null'], 'column_map': {'age': 'age'}},
    'cjs': None,
    'KDDCup09_upselling': None,
    'blood-transfusion-service-center': None,
    'Satellite': None,
    'amazon-commerce-reviews': None,
    'bank-marketing': {'bin': ['V5', 'V7', 'V8']},
    'eeg-eye-state': None,
    'one-hundred-plants-margin': None,
    'one-hundred-plants-shape': None,
    'one-hundred-plants-texture': None,
    'monks-problems-2': None,
    'tic-tac-toe': None,
    'kr-vs-kp': None,
    'qsar-biodeg': None,
    'phoneme': None,
    'hill-valley': None,
    'spambase': None,
    'ilpd': None,
    'pc1': None,
    # ozone-level-8hr
    1487: None,
    # spambase
    44: None,
    # adult
    1590: None,
    # telco-customer-churn
    42178: {"bin": ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]},
    # KDDCup09_appetency
    1111: None,
    # credit-g
    31: {"bin": ["own_telephone", "foreign_worker"]},
    # Click_prediction_small
    42733: None,
    # qsar-biodeg
    #1494: {"bin": ["V24", "V25", "V29"]},
    1494: None,
    # arrhythmia
    1017: {"bin": [
        "chDI_RRwaveExists", "chDI_DD_RRwaveExists", "chDI_RPwaveExists",
        "chDI_DD_RPwaveExists", "chDI_RTwaveExists", "chDI_DD_RTwaveExists",
        "chDII_RRwaveExists", "chDII_DD_RRwaveExists", "chDII_RPwaveExists",
        "chDII_DD_RPwaveExists", "chDII_RTwaveExists", "chDII_DD_RTwaveExists",
        "chDIII_RRwaveExists", "chDIII_DD_RRwaveExists", "chDIII_RPwaveExists",
        "chDIII_DD_RPwaveExists", "chDIII_RTwaveExists", "chDIII_DD_RTwaveExists",
        "chAVR_RRwaveExists", "chAVR_DD_RRwaveExists", "chAVR_RPwaveExists",
        "chAVR_DD_RPwaveExists", "chAVR_RTwaveExists", "chAVR_DD_RTwaveExists",
        "chAVL_DD_RRwaveExists", "chAVL_RPwaveExists", "chAVL_DD_RPwaveExists",
        "chAVL_RTwaveExists", "chAVL_DD_RTwaveExists", "chAVF_RRwaveExists",
        "chAVF_DD_RRwaveExists", "chAVF_DD_RPwaveExists", "chAVF_RTwaveExists",
        "chAVF_DD_RTwaveExists", "chV1_RRwaveExists", "chV1_DD_RRwaveExists",
        "chV1_RPwaveExists", "chV1_DD_RPwaveExists", "chV1_RTwaveExists",
        "chV1_DD_RTwaveExists", "chV2_RRwaveExists", "chV2_DD_RRwaveExists",
        "chV2_RPwaveExists", "chV2_DD_RPwaveExists", "chV2_RTwaveExists",
        "chV2_DD_RTwaveExists", "chV3_RRwaveExists", "chV3_DD_RRwaveExists",
        "chV3_RPwaveExists", "chV3_DD_RPwaveExists", "chV3_RTwaveExists",
        "chV3_DD_RTwaveExists", "chV4_RRwaveExists", "chV4_DD_RRwaveExists",
        "chV4_RTwaveExists", "chV4_DD_RTwaveExists", "chV5_DD_RRwaveExists",
        "chV5_DD_RPwaveExists", "chV5_DD_RTwaveExists", "chV6_RRwaveExists",
        "chV6_DD_RRwaveExists", "chV6_RPwaveExists", "chV6_DD_RTwaveExists"
        ]
    },
    # Bioresponse
    4134: None,
    # blood-transfusion
    1464: None,
    # monks-problems-2
    334: None,
    # tic-tac-toe
    50: None,
    # steel-plates-fault
    1504: {"bin": [
        "V12", "V13", "V28", "V29", "V30", "V31", "V32", "V33"
        ]
    },
    # wdbc
    1510: None,
    #
    1489: None,
    # kc2
    1063: None,
    # kc1
    1467: None,
    # ilpd
    1480: None,
    # pc1
    1068: None,
    # pc4
    1049: None,
    # pc3
    1050: None,
    # scene 
    312: {"bin": ["Beach", "Sunset", "FallFoliage", "Field", "Mountain"]},
    # sick
    38: {"bin": [
        "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
        "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
        "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary",
        "psych", "TSH_measured", "T3_measured", "TT4_measured", "T4U_measured",
        "FTI_measured"
        ]
    },
    # mushroom
    24: {"bin": ["bruises%3F"]},
    # churn 
    40701: {"bin": ["international_plan", "voice_mail_plan"]},
    # delta_ailerons
    803: None,
    # banknote-authentication
    1462: None,
    # wilt
    40983: None,
    # Satellite
    40900: None,
    # pollen
    871: None,
    # bank marketing
    1558: {"bin": ["V5", "V7", "V8"]},
    # JapaneseVowels
    976: None,
    # mc1
    1056: {"bin": ["DESIGN_DENSITY", "ESSENTIAL_DENSITY", "GLOBAL_DATA_DENSITY", "MAINTENANCE_SEVERITY"]}, 
    # kin8nm
    807: None,
    # mfeat-karhunen
    1020: None,
    # delta_elevators
    819: None,
    # eeg-eye-state
    1471: None,
    # nomao
    1046: {"bin": ["event"]},
    # jm1
    1053: None,
    # bank marketing
    1461: {"bin": ["V5", "V7", "V8"]},
    # click_prediction_small
    1220: None,
    # higgs
    23512: None,
    # numerai
    23517: None,
    # --------------------------------------------------------------------------
    # eucalyptus 
    188: None,
    # covertype
    1596: None,
    # Diabetes130US
    4541: {"bin": ["diabetesMed"]},
    # car-evaluation
    40664: {"bin": [
        "buying_price_vhigh", "buying_price_high", "buying_price_med",
        "buying_price_low", "maintenance_price_vhigh", "maintenance_price_high",
        "maintenance_price_med", "maintenance_price_low", "luggage_boot_size_small",
        "luggage_boot_size_med", "luggage_boot_size_big", "safety_low",
        "safety_med", "safety_high"
        ]
    },
    # shuttle
    40685: None,
    # solar-flare
    40687: None,
    # car
    40975: None,
    # volkert
    41166: None,
    # one-hundred-plants-margin
    1491: None,
    # one-hundred-plants-shape
    1492: None,
    # one-hundred-plants-texture
    1493: None,
    # letter
    6: None,
    # isolet
    300: None,
    # helena
    41169: None,
    # okcupid-stem
    42734: None,
    # soybean
    42: {"bin": ["hail", "lodging"]},
    # mfeat-karhunen
    16: None,
    # mfeat-fourier
    14: None,
    # mfeat-factor
    12: None,
    # mfeat-morphological
    18: None,
    # optdigits
    28: {"bin": ["input25", "input33", "input57"]},
    # mice-protein
    40966: None,
    # autoUniv-au7-1100
    1552: None,
    # autoUniv-au4-2500
    1548: None,
    # baseball
    185: None,
    # mfeat-zernike
    22: None,
    # satimage
    182: None,
    # first-order-theorem-proving
    1475: None,
    # wall-robot-navigation
    1497: None,
    # Abalone
    183: None,
    # Gesture
    4538: None,
    # Pixel
    20: None,
    # Characters
    1459: None,
    # Gas
    1476: None,
    # Nursery
    26: None,
    # Kropt
    184: None,
    # RMFTSA
    679: None,
    # Splice
    46: None,
    # IPUMS
    381: {"bin": ["workedyr", "vetstat"]},
    # CJS
    473: None,
    # Cardiotocography
    1560: None,
    # Volcano
    1529: None,
    # Volcano-d3
    1540: None,
    # Volcano-d1
    1538: None,
    # Nursery
    1568: None,
    # RobotNavigation
    1525: None,
    # ThyroidAllbp
    40474: None,
    # ThyroidAllhyper
    40475: None,
    # --------------------------------------------------------------------------
    # kin8nm
    189: None,
    # socmob
    541: None,
    # abalone
    42726: None,
    # colleges
    42727: None,
    # topo_2_1
    422: None,
    # live-disorders
    8: None,
    # cholesterol
    204: None,
    # meta
    566: None,
    # chscase_census2
    673: None,
    # plasma_retinol
    511: None,
    # stock
    223: None,
    # cpmp-2015
    41700: None,
    # space_ga
    507: None
}

NUM_CLASSES = {
    'credit-g': 2,
    'Click_prediction_small': 2,
    'credit-approval': 2,
    'dresses-sales': 2,
    'adult': 2,
    'pendigits': 10,
    'steel-plates-fault': 2,
    'SpeedDating': 2,
    'cjs': 6,
    'KDDCup09_upselling': 2,
    'blood-transfusion-service-center': 2,
    'Satellite': 2,
    'amazon-commerce-reviews': 50,
    'bank-marketing': 2,
    'eeg-eye-state': 2,
    'one-hundred-plants-margin': 100,
    'one-hundred-plants-shape': 100,
    'one-hundred-plants-texture': 100,
    'monks-problems-2': 2,
    'tic-tac-toe': 2,
    'kr-vs-kp': 2,
    'qsar-biodeg': 2,
    'phoneme': 2,
    'hill-valley': 2,
    'spambase': 2,
    'ilpd': 2,
    'pc1': 2,
    # binary classification
    1487: 2,
    44: 2,
    1590: 2,
    42178: 2,
    1111: 2,
    31: 2,
    42733: 2,
    1494: 2,
    1017: 2,
    4134: 2,
    1464: 2,
    334: 2,
    50: 2,
    1504: 2,
    1510: 2,
    1489: 2,
    1063: 2,
    1467: 2,
    1480: 2,
    1068: 2,
    1049: 2,
    1050: 2,
    312: 2,
    38: 2,
    24: 2,
    40701: 2,
    803: 2,
    1462: 2,
    40983: 2,
    40900: 2,
    871: 2,
    1558: 2,
    976: 2,
    1056: 2,
    807: 2,
    1020: 2,
    819: 2,
    1471: 2,
    1046: 2,
    1053: 2,
    1461: 2,
    1220: 2,
    23512: 2,
    23517: 2,
    # multiclass classification
    188: 5,
    1596: 7,
    4541: 3,
    40664: 4,
    40685: 7,
    40687: 6,
    40975: 4,
    41166: 10,
    41169: 100,
    1491: 100,
    1492: 100,
    1493: 100,
    6: 26,
    300: 26,
    42734: 3,
    42: 19,
    16: 10,
    14: 10,
    12: 10,
    18: 10,
    28: 10,
    40966: 8,
    1552: 5,
    1548: 3,
    185: 3,
    22: 10,
    182: 6,
    1475: 6,
    1497: 4,
    183: 28,
    4538: 5,
    20: 10,
    1459: 10,
    1476: 6,
    26: 5,
    184: 18,
    679: 4,
    381: 7,
    473: 6,
    46: 3,
    1560: 3,
    1529: 5,
    1540: 5,
    1538: 5,
    1568: 4,
    1525: 4,
    40474: 5,
    40475: 5,
    # regression
    189: 1,
    541: 1,
    42726: 1,
    42727: 1,
    422: 1,
    8: 1,
    204: 1,
    566: 1,
    673: 1,
    511: 1,
    223: 1,
    41700: 1,
    507: 1
}