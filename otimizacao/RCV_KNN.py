from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv("dados.csv")

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

    df = pd.get_dummies(df, columns=['category'], prefix='categoria', dtype=int, drop_first=True)
    df['trans_time'] = pd.to_timedelta(df['trans_time']).dt.total_seconds().astype(int)
    return df

df = tratar(df)

N = 4000000  # quantidade total a remover

df_0 = df[df['is_fraud'] == 0]
df_1 = df[df['is_fraud'] == 1]
n0, n1 = len(df_0), len(df_1)
total = n0 + n1
p0, p1 = n0 / total, n1 / total
remove_0 = int(N * p0)
remove_1 = int(N * p1)

idx_remove_0 = df_0.sample(n=remove_0, random_state=33).index
idx_remove_1 = df_1.sample(n=remove_1, random_state=33).index

df = df.drop(list(idx_remove_0) + list(idx_remove_1))

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=33
)


pipeline = ImbPipeline(steps=[
    ("scaler", RobustScaler()),
    ("smote", SMOTE(random_state=33)),
    ("knn", KNeighborsClassifier())
])


param_dist = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["minkowski", "euclidean", "manhattan"],
}

scorer = make_scorer(f1_score)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=6,
    scoring=scorer,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=33),
    verbose=2,
    n_jobs=-1,
    random_state=33
)

print("\nIniciando RandomizedSearch...\n")
random_search.fit(X_train, y_train)

print("\n====================================")
print("MELHORES PARÂMETROS ENCONTRADOS:")
print("====================================")
print(random_search.best_params_)

print("\nMelhor F1 médio (CV):")
print(random_search.best_score_)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=== Desempenho no conjunto de teste ===")
print(classification_report(y_test, y_pred))


print("\nKNN não possui feature importance — nenhuma plotagem será exibida.")
