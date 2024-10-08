import numpy as np
import pandas as pd

df = pd.read_csv('data_dir/liver.csv', low_memory=False)
print('size (raw) = {}'.format(df.index.size))

df.AGE = pd.to_numeric(df.AGE, errors='coerce')
df.AGE_DON = pd.to_numeric(df.AGE_DON, errors='coerce')
df.CREAT_TX = pd.to_numeric(df.CREAT_TX, errors='coerce')
df.INR_TX = pd.to_numeric(df.INR_TX, errors='coerce')
df.PX_STAT_DATE = pd.to_datetime(df.PX_STAT_DATE, errors='coerce')
df.TBILI_TX = pd.to_numeric(df.TBILI_TX, errors='coerce')
df.TX_DATE = pd.to_datetime(df.TX_DATE, errors='coerce')
df.WGT_KG_CALC = pd.to_numeric(df.WGT_KG_CALC, errors='coerce')
df.WGT_KG_DON_CALC = pd.to_numeric(df.WGT_KG_DON_CALC, errors='coerce')

df = df[df.ABO != 'UNK']
df = df[df.ABO_DON != 'UNK']
df = df[df.AGE >= 18]
df = df[df.AGE_DON >= 18]
df = df[~df.CREAT_TX.isna()]
df = df[(df.DIAL_TX == 'N') | (df.DIAL_TX == 'Y')]
df = df[~df.END_DATE.isna()]
df = df[~df.INIT_DATE.isna()]
df = df[~df.INR_TX.isna()]
df = df[(df.LIFE_SUP_TRR == 'N') | (df.LIFE_SUP_TRR == 'Y')]
df = df[~df.TBILI_TX.isna()]
df = df[~df.TX_DATE.isna()]
df = df[~df.WGT_KG_CALC.isna()]
df = df[~df.WGT_KG_DON_CALC.isna()]

df = df[~(df.INIT_DATE == ' ')]
df = df[~(df.END_DATE == ' ')]
df = df[~(df.TX_DATE == ' ')]


df = df[df.PX_STAT == 'D']
df['SURVIVAL'] = (df.PX_STAT_DATE - df.TX_DATE) / np.timedelta64(365,'D')
df['TX_DATE_REC'] = df.TX_DATE
df['ABO_MISMATCH'] = (df.ABO != df.ABO_DON).replace({True: 1, False: -1})
df['DIAL_TX'] = (df.DIAL_TX == 'Y').replace({True: 1, False: -1})
df['LIFE_SUP_TRR'] = (df.LIFE_SUP_TRR == 'Y').replace({True: 1, False: -1})
df['WGT_DIFF'] = np.abs(df.WGT_KG_CALC - df.WGT_KG_DON_CALC)


df2 = df[['SURVIVAL','ABO_MISMATCH', 'AGE', 'CREAT_TX', 'DIAL_TX', 'INR_TX', 'LIFE_SUP_TRR', 'TBILI_TX', 'WGT_DIFF']]
df2 = (df2 - df2.mean()) / df2.std()


print('size = {}'.format(df2.index.size))

from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd


# training data 1: before 2002
tr1_idx = np.where(df['TX_DATE'] <  np.datetime64(date.fromisoformat('2002-02-26')))[0]
print('training data 1 examples:',len(tr1_idx))

tr2_idx = np.where((np.datetime64(date.fromisoformat('2002-02-26')) < df['TX_DATE']) & (df['TX_DATE'] <  np.datetime64(date.fromisoformat('2003-05-01'))))[0]
print('training data 2 examples:',len(tr2_idx))

test_idx = np.where((np.datetime64(date.fromisoformat('2003-05-01')) < df['TX_DATE']) & (df['TX_DATE'] <  np.datetime64(date.fromisoformat('2005-01-01'))))[0]
print('test data 1 examples:',len(test_idx))