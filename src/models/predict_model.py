import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predire_modele(modele, X_test):    
    # Génère les prédictions d'un modèle entraîné.
    return modele.predict(X_test)
 
def calculer_metriques_modele(y_test, y_pred, nom_modele, nom_dataset):
    
    # Calcule les métriques d'un modèle.
    resultats = pd.DataFrame({
        "Dataset": [nom_dataset],
        "Modèle": [nom_modele],
        "MAE (€/m²)": [mean_absolute_error(y_test, y_pred)],
        "RMSE (€/m²)": [np.sqrt(mean_squared_error(y_test, y_pred))],
        "R²": [r2_score(y_test, y_pred)]
    })

    return resultats