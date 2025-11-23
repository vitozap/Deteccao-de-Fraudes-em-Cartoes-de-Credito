import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv('dados.csv')

print("\nPrimeiras linhas do dataset:")
print(df.head())
print("\nInformações gerais:")
print(df.info())

def tratar(df):
    df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    dias_de_vida = (df['trans_date'] - df['dob']).dt.days
    idade_com_nan = dias_de_vida / 365.25
    idade_com_nan.fillna(idade_com_nan.median(), inplace=True)
    df['idade'] = idade_com_nan.astype(int)

    colunas_para_remover = [
        'gender', 'city', 'state', 'zip', 'profile', 'merchant', 'ssn',
        'cc_num', 'first', 'last', 'street', 'acct_num', 'trans_num',
        'job', 'dob', 'trans_date'
    ]
    df = df.drop(columns=colunas_para_remover)

    df = pd.get_dummies(df, columns=['category'], prefix='categoria', dtype=int)

    df['trans_time'] = pd.to_timedelta(df['trans_time']).dt.total_seconds().astype(int)

    return df

df = tratar(df)

indices_para_remover = df.sample(n=4000000, random_state=42).index
df = df.drop(indices_para_remover)

colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
colunas_numericas.remove("is_fraud")

df_before = df[colunas_numericas].copy()

scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[colunas_numericas])
df_scaled = pd.DataFrame(df_scaled, columns=colunas_numericas, index=df.index)

df_after = df.copy()
df_after[colunas_numericas] = df_scaled



print("\n=== PRIMEIRAS LINHAS (ANTES DO ROBUST SCALER) ===")
print(df_before.head())

print("\n=== PRIMEIRAS LINHAS (DEPOIS DO ROBUST SCALER) ===")
print(df_after[colunas_numericas].head())
