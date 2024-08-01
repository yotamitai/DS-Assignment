import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
import pyarrow.feather as feather
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore', category=RuntimeWarning)


def get_selected_columns(df):
    selected_cols = ['MSAHYEA', 'DJANPLY', 'LJWUXOV', 'VZOZKWX', 'MRUWBZT', 'DVSZBLN', 'DMTXDGF',
                     'SJZKEVZ', 'PRVIFOQ', 'VMWDQGF', 'UNFEABR', 'HKTLMYY', 'XIRNYHK', 'OOASVXJ',
                     'FXACWUA', 'KDQUJOB', 'KPQSPBC', 'XEUMIIT', 'DUOKWOI', 'STMKCSJ', 'WLULXFN',
                     'AVSMOFQ', 'CNGIUEP', 'LKFNGDB', 'LLZRQRY', 'NYWUAUO', 'LGVRVXB', 'CMSGVTL',
                     'EULURHL', 'NSERJIK', 'WGQRBHJ', 'SBUETBP', 'DDGIRQA', 'JIOWSAX', 'EYILHOU',
                     'FFJOGRA', 'FPBFOEB', 'JEKQIKE', 'VYTOISY', 'AIKOJYC', 'AHVIXII', 'RTVQHPO',
                     'HFYZKSZ', 'YLHQTEA', 'LCAOHWW', 'AWMGPPL', 'IIYLALO', 'SRQKQTX', 'JNEVNAR',
                     'HWKCDPO', 'UTGKBXG', 'NIVQSOG', 'TPUASCQ', 'FCMPHJR', 'BYEQVGG', 'LLHXSPF',
                     'PDWQQDP', 'LQEHWDR', 'VFXSABJ', 'DTJITAO', 'DAEPCOR', 'OFEEPAC', 'RNKJCEN',
                     'AYAJVQL', 'KPZEEJR', 'MOSECGI', 'XLVHGLO', 'DXWRFDA', 'JTIJWNL', 'AKUNFFN',
                     'KPXMEBJ', 'GBGZJZO', 'GRBGIZR', 'KNFIDTO', 'UOGNHSF', 'PSXOLCG', 'FAIOOOV',
                     'UHMRZHI', 'FBBFYZM', 'SQODRRP', 'NAWMDDT', 'MDBYIRV', 'SPKCFPP', 'BGDAMPC',
                     'DANYGJI', 'OXRCIYW', 'IKMWIOV', 'MQXCIBE', 'OOVNYOI', 'LTCKGYN', 'HBQPQTD',
                     'SDEUDHY', 'BHSHCHU', 'DDPBVDN', 'JIWJHVI', 'ZRYDRXT', 'GOBJALH', 'LOGULZS',
                     'ARVLGNZ', 'MWHYOSB', 'JMFGDPB', 'IHCEXCN', 'GQHLWWM', 'AWJPBPO', 'AGTCLZR',
                     'AEXRRBM', 'OGYQNUB', 'APJFLOK', 'GKACPXS', 'JWXNCNT', 'FUTFIRO', 'ENIRWLT',
                     'NVBZJEU', 'BGHDMAS', 'TKQUNLP', 'RIEVFEX', 'QASZGHA', 'MRBOALK', 'DNHHKRL',
                     'ZKXMWHB', 'SMHHBFH', 'EDASGHM', 'RCJGZLT', 'UZGUYFK', 'LCFCVCB', 'VEUJYWN',
                     'IJRFPEK', 'UMJYMGD', 'XWQPLHB', 'ITWTNIT', 'VTMSCPQ', 'DPPXTGF', 'BGPVFMN',
                     'KEJOIIS', 'HOVFLAR', 'HIFPGGQ', 'ZYMFJDH', 'WLFAYHX', 'NGVHWDO', 'HCSXZKW',
                     'JYSKSPX', 'OKIKPOJ', 'KGJACPV', 'XJJAKPY', 'CWLCCPL', 'HHSFOPV', 'ONKWSSO',
                     'GVDVKJH', 'KXISVUL', 'MUEKGHC', 'YCJYNVQ', 'HEBFRRA', 'KYRXYOZ', 'OADGFBJ',
                     'MKHJGMF', 'JIMYAME', 'ULYIQYO', 'VPAZKWG', 'BEUAZOI', 'TPYJDFW', 'UXZNNNB',
                     'QODSZMV', 'GUEUYTS', 'HSCCVTR', 'LBMUJNZ', 'BQLPPBV', 'RLHGEVY', 'ZCRFMLI',
                     'JOJRNMZ', 'IHLMFTI', 'OPBTWFJ', 'NQQYONU', 'ZPBDTHO', 'QWVFRRZ', 'GYFZSXY',
                     'POUQMMG', 'FWILIBF', 'ALKYTAY', 'IWHIYNA', 'HPCWOKU', 'NYRKOCF', 'JUMKXBC',
                     'LVMHJCI', 'VCPMKTP', 'KUXSPYJ', 'OULTOYT', 'ACVSDTK', 'HCBDBBS', 'RQTEIMY',
                     'DTSZUFG', 'JORKKSF', 'CQZHXYY', 'RAGOFAC', 'RIEGYBR', 'NTYHQRF', 'NGIJPET',
                     'ZMEJKZD', 'UEIYMWS', 'RSQGZAF', 'TTQVFLL', 'DDZYPGM', 'WBCMBKW', 'VOHXBZJ',
                     'UZPPCUZ', 'WTMETCB', 'VRNGUZU', 'PAIOUXI', 'PZYGETW', 'OYSQKAH', 'NMWUWJL',
                     'ACEFRZA', 'VXWAKJT', 'CWWUCQG', 'MHLQVAB', 'NDWFVEZ', 'MIIVLBT', 'PPNLAEY',
                     'XLHIYUD', 'KBNVGJL', 'UJTXHTS', 'UJYQCMY', 'TLJYWBE']
    df_selected = df[selected_cols].copy()
    return df_selected


# %%
def add_significant_features(df):
    significant_interactions = ['IKMWIOV_APJFLOK', 'IKMWIOV_KNFIDTO', 'IKMWIOV_ACEFRZA',
                                'OOASVXJ_AIKOJYC', 'AKUNFFN_AIKOJYC', 'IKMWIOV_QASZGHA',
                                'ALKYTAY_AIKOJYC', 'IKMWIOV_LLZRQRY', 'FFJOGRA_AIKOJYC',
                                'AIKOJYC_ACEFRZA', 'AVSMOFQ_AKUNFFN']

    df_sig = df.copy()
    for pair in significant_interactions:
        i, j = pair.split('_')
        df_sig[pair] = df.eval(f'{i} * {j}')
    return df


def log_transform_label(df):
    """Transform target ot log scale"""
    df['TLJYWBE'] = np.log(df['TLJYWBE'])
    print("Target Transformed to log scale")
    return df


def remove_empty(df):
    """remove empty columns"""
    empty = df.columns[df.isna().all()]
    df_empty = df.drop(columns=empty)
    print(f"Empty columns dropped: {len(empty)}")
    return df_empty


def handle_object_cols(df):
    """Handeling object columns"""
    df_obj = df.copy()
    df_obj.replace('nan', np.nan, inplace=True)
    obj_cols_to_lower = df_obj.select_dtypes(include='object').nunique().loc[
        lambda x: x <= 10].index
    df_obj[obj_cols_to_lower] = df_obj[obj_cols_to_lower].apply(lambda x: x.str.lower())
    binary_col, cat_col = [], []
    binary_map = {'true': 1, 'false': 0, np.nan: np.nan}
    for c in obj_cols_to_lower:
        unique_values = df_obj[c].dropna().unique()
        num_unique_values = len(unique_values)
        if num_unique_values == 2 or (
                num_unique_values == 3 and set(unique_values) == {'true', 'false'}):
            df_obj[c] = df_obj[c].map(binary_map)
            df_obj[c] = pd.to_numeric(df_obj[c], errors='coerce').astype('Int64')
            binary_col.append(c)
        else:
            dummy_columns = pd.get_dummies(df_obj[c], dummy_na=True, prefix=c)
            df_obj = pd.concat([df_obj, dummy_columns], axis=1)
            df_obj.drop(c, axis=1, inplace=True)

    """convert other object columns to numerical"""
    for c in df_obj.select_dtypes(include=[object]).columns:
        df_obj[c] = pd.to_numeric(df_obj[c], errors='coerce')

    return df_obj


def remove_missing(df):
    """Remove columns with more than 30% missing values"""
    missing_cols = list(df.columns[df.isnull().mean() > 0.3])
    df_missing = df.drop(columns=missing_cols)
    print(f"Columns with more than 30% missing values dropped: {len(missing_cols)}")
    return df_missing


def zero_variance(df):
    """zero Variance"""
    # Create masks for conditions
    cols = df.columns
    zero_std_mask = df[cols].std() == 0
    count_mask = df[cols].count() == df.shape[0]
    zero_std_all = cols[zero_std_mask & count_mask].tolist()
    zero_std = cols[zero_std_mask & ~count_mask].tolist()
    for c in zero_std:
        val = df[c].mode()[0]
        df[c] = df[c].fillna(0)
        df[c] = (df[c] == val).astype(bool)
    df_var = df.drop(columns=zero_std_all)
    print(f"Zero Variance columns dropped: {len(zero_std_all)}")
    return df_var


def bool_cols(df):
    """Boolean columns"""
    df_bool = df.copy()
    bool_cols = df_bool.columns[df_bool.nunique(dropna=True) == 2]
    for col in bool_cols:
        unique_vals = df_bool[col].dropna().unique()
        mapping = {unique_vals[0]: False, unique_vals[1]: True}
        df_bool[col] = df_bool[col].map(mapping).astype(bool)
    print("Boolean cols: ", len(bool_cols))
    return df_bool


def normal_outliers(df):
    """Outliers in Normal distribution"""
    normal_cols = []
    non_bool_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for column in non_bool_cols:
        data = df[column].dropna()
        if len(data) > 7:
            stat, p = stats.normaltest(data)
            if p > 0.05:
                normal_cols.append(column)

    mask = pd.Series(True, index=df.index)
    means = df[normal_cols].mean()
    std_devs = df[normal_cols].std()
    for col in normal_cols:
        col_mask = ((df[col] >= means[col] - 3 * std_devs[col]) &
                    (df[col] <= means[col] + 3 * std_devs[col]))
        mask &= col_mask

    print("Outliers dropped:", len(df.index[~mask]))
    df_out = df.drop(df.index[~mask])
    return df_out


def impute_missing(df):
    """Impute missing values"""
    imputer = SimpleImputer(strategy='median')
    print("number of missing values:", df.isnull().sum().sum())
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    # Check if any missing values remain
    print("remaining missing value:", df_imputed.isnull().sum().max())
    return df_imputed


def standardize(df):
    """standardizing numeric data"""
    # Identify non-boolean columns
    df_scaled = df.copy()
    non_bool_columns = df_scaled.select_dtypes(exclude=['bool']).columns
    scaler = StandardScaler()
    # Apply StandardScaler to non-boolean columns
    df_scaled[non_bool_columns] = scaler.fit_transform(df_scaled[non_bool_columns])
    return df_scaled


def remove_duplicates(df):
    """remove duplicate columns"""
    duplicates = set()
    df_scaled = df.copy()
    # Compare each column against all others
    for i in range(df_scaled.shape[1]):
        col1 = df_scaled.iloc[:, i]
        for j in range(i + 1, df_scaled.shape[1]):
            col2 = df_scaled.iloc[:, j]
            if col1.equals(col2):
                duplicates.add(df_scaled.columns[j])

    # Drop the duplicate columns
    df_reduced = df_scaled.drop(columns=list(duplicates))
    print(f"Removed {len(duplicates)} duplicate columns")
    return df_reduced


def process(args):
    df = feather.read_feather(args.path)
    print('Dataset Loaded', df.shape)
    # sample
    if args.sample:
        df = df.sample(frac=args.sample, random_state=args.random_state)
        print(f"Using a sample of {100 * args.sample}% of data")

    df = log_transform_label(df)
    df = get_selected_columns(df)
    df = remove_empty(df)
    df = handle_object_cols(df)
    df = add_significant_features(df)
    df = remove_missing(df)
    df = zero_variance(df)
    df = bool_cols(df)
    df = normal_outliers(df)
    df = standardize(df)
    df = impute_missing(df)
    df = remove_duplicates(df)

    """save results"""
    path, filename = os.path.split(args.path)
    new_path = os.path.join(path, filename.replace('.', '_processed.'))
    df.to_feather(new_path)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pipeline for preprocessing dataset')
    parser.add_argument('--path', type=str, help='Input file', default='home_assignment.feather')
    parser.add_argument('--random_state', type=int, help='random state for sampling', default=42)
    parser.add_argument('--sample', type=float, help='sample from the dataset', default=False)
    args = parser.parse_args()
    process(args)
