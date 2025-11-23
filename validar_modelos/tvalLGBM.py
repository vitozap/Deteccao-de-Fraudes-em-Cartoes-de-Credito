from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgbm
from sklearn.metrics import (
    accuracy_score, auc, f1_score, recall_score, roc_auc_score,
    average_precision_score, precision_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv('dados.csv')

print("\nPrimeiras linhas do dataset:")
print(df.head())
print("\nInformações:")
print(df.info())
print("\nDistribuição das classes:")
print(df['is_fraud'].value_counts())


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

N = 4000000  # quantidade total a remover

# separa por classe
df_0 = df[df['is_fraud'] == 0]
df_1 = df[df['is_fraud'] == 1]

# tamanhos
n0 = len(df_0)
n1 = len(df_1)
total = n0 + n1

# proporção por classe
p0 = n0 / total
p1 = n1 / total

# quantidade a remover por classe
remove_0 = int(N * p0)
remove_1 = int(N * p1)

# seleciona linhas a remover
idx_remove_0 = df_0.sample(n=remove_0, random_state=42).index
idx_remove_1 = df_1.sample(n=remove_1, random_state=42).index

# junta e remove
df = df.drop(list(idx_remove_0) + list(idx_remove_1))

print("\nInformações APÓS tratamento:")
print(df.info())
print("\nDistribuição das classes APÓS tratamento:")
print(df['is_fraud'].value_counts())


X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

lgbm_model = lgbm.LGBMClassifier(
    subsample=0.9,
    reg_lambda=0,
    reg_alpha=0,
    num_leaves=31,
    n_estimators=1000,
    min_child_samples=20,
    max_depth=10,
    learning_rate=0.1,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', RobustScaler()),
    ('model', lgbm_model)
])


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

resultados = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'specificity': [],
    'error_rate': [],
    'f1': [],
    'roc_auc': [],
    'pr_auc': []
}

y_true_total = []
y_pred_total = []
y_prob_total = []


fold = 1
for train_idx, test_idx in kfold.split(X, y):
    print(f"\n========== FOLD {fold} ==========")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    y_true_total.extend(y_test)
    y_pred_total.extend(y_pred)
    y_prob_total.extend(y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    error_rate = (fp + fn) / (tp + tn + fp + fn)

    resultados['accuracy'].append(acc)
    resultados['precision'].append(prec)
    resultados['recall'].append(rec)
    resultados['f1'].append(f1)
    resultados['roc_auc'].append(roc)
    resultados['pr_auc'].append(pr)
    resultados['specificity'].append(specificity)
    resultados['error_rate'].append(error_rate)


    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"PR-AUC:    {pr:.4f}")
    print(f"Especificidade: {specificity:.4f}")
    print(f"Taxa de Erro:   {error_rate:.4f}")


    fold += 1


print("\n================== RESULTADOS FINAIS (MÉDIA DOS 5 FOLDS) ==================\n")
for metrica, valores in resultados.items():
    print(f"{metrica.upper():10} | MÉDIA = {np.mean(valores):.4f} | DESVIO = {np.std(valores):.4f}")


cm_final = confusion_matrix(y_true_total, y_pred_total)

plt.figure(figsize=(5,4))
sns.heatmap(cm_final, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão (Soma dos 5 Folds)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


pipeline.fit(X, y)
modelo_final = pipeline.named_steps['model']

importances = pd.DataFrame({
    'feature': X.columns,
    'importance': modelo_final.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n=== 20 Variáveis Mais Importantes ===")
print(importances.head(20))

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=importances.head(20), palette='viridis')
plt.title('Top 20 Variáveis Mais Relevantes (LightGBM)')
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_true_total, y_prob_total)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_value:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC (Soma dos 5 Folds)")
plt.legend()
plt.tight_layout()
plt.show()