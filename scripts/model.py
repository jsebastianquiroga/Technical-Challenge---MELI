import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Función para seleccionar características más relevantes
def select_temporal_features(df, target='failure', num_features=10):
    """
    Selecciona las características más relevantes basadas en información mutua.
    
    Args:
        df (pd.DataFrame): Dataset con características y variable objetivo.
        target (str): Nombre de la columna objetivo.
        num_features (int): Número de características principales a seleccionar.
    
    Returns:
        pd.DataFrame: Dataset con las características seleccionadas.
    """
    X = df.drop(columns=['device', 'date', target])
    y = df[target]
    scores = mutual_info_classif(X, y, random_state=42)
    top_indices = np.argsort(scores)[-num_features:]
    selected_features = X.columns[top_indices]
    return df[selected_features.tolist() + ['device', 'date', target]]

# Función para entrenar y evaluar el modelo
def train_and_evaluate_logistic_regression(df, target='failure', num_features=10, n_splits=5):
    """
    Entrena y evalúa un modelo de regresión logística con validación cruzada temporal.
    
    Args:
        df (pd.DataFrame): Dataset con características y variable objetivo.
        target (str): Nombre de la columna objetivo.
        num_features (int): Número de características principales a seleccionar.
        n_splits (int): Número de splits para validación cruzada.
    
    Returns:
        dict: Resultados del experimento.
    """
    # Seleccionar características más relevantes
    df_selected = select_temporal_features(df, target=target, num_features=num_features)

    # Separar características y variable objetivo
    X = df_selected.drop(columns=['device', 'date', target])
    y = df_selected[target]

    # Escalar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configurar validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    log_reg = LogisticRegression(class_weight='balanced', random_state=42)

    roc_auc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    specificity_scores = []

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Verificar si el split contiene ambas clases
        if len(np.unique(y_test)) < 2:
            print("Split omitido: contiene una sola clase en y_test.")
            continue

        # Entrenar el modelo
        log_reg.fit(X_train, y_train)
        y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

        # Calcular métricas
        metrics = calculate_model_metrics(y_test, y_pred_proba)
        roc_auc_scores.append(metrics['roc_auc'])
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
        specificity_scores.append(metrics['specificity'])

    # Resumen de métricas
    results = {
        'Modelo': 'Logistic Regression',
        'ROC-AUC': np.mean(roc_auc_scores),
        'Precision': np.mean(precision_scores),
        'Recall': np.mean(recall_scores),
        'Specificity': np.mean(specificity_scores),
        'F1-Score': np.mean(f1_scores)
    }

    # Importancia de características
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': log_reg.coef_.flatten()
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    return results, feature_importance


import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_model_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calcula todas las métricas requeridas para la evaluación del modelo.
    
    Args:
        y_true: Valores reales
        y_pred_proba: Probabilidades predichas
        threshold: Umbral para clasificación binaria
    
    Returns:
        dict: Diccionario con todas las métricas
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calcular especificidad manualmente
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'specificity': specificity,
        'f1': f1_score(y_true, y_pred)
    }