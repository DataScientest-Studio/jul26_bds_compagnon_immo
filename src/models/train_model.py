# Préprocessing
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
# Modèles
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Séparation des données et optimisation
from sklearn.model_selection import (train_test_split, RandomizedSearchCV, GridSearchCV)

# Distribution pour RandomizedSearchCV
from scipy.stats import randint, uniform


def preparer_donnees(df, target="prix_m2", test_size=0.20, random_state=42):
    # Sépare les variables explicatives, la cible et les jeux train/test
    X = df.drop(columns=[target, "valeur_fonciere"]).copy()
    y = df[target].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=X["type_local"]) 


def creer_preprocessor(X_train):
    # Crée le preprocessing numérique et catégoriel
    variables_num = X_train.select_dtypes(include=["number"]).columns.tolist()
    variables_cat = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    pipeline_num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    pipeline_cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", pipeline_num, variables_num), ("cat", pipeline_cat, variables_cat)])


def valider_modeles_cv(modeles, X_train, y_train, n_splits=5):    

    # Création de la méthode de validation croisée
    validation_croisee = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Création d'une liste vide pour stocker les résultats
    resultats = []

    # Parcours de chaque modèle contenu dans le dictionnaire
    for nom_modele, modele in modeles.items():

        # Calcul des scores du modèle sur chaque fold
        scores = cross_val_score(
            estimator=modele,
            X=X_train, 
            y=y_train,
            cv=validation_croisee,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        # Scikit-learn retourne une MAE négative
        # Le signe moins permet de la transformer en valeur positive
        mae_folds = -scores

        # Ajout des résultats du modèle dans la liste
        resultats.append({
            "Modèle": nom_modele,
            "Nombre de folds": n_splits,
            "MAE moyenne (€/m²)": mae_folds.mean(),
            "Écart-type (€/m²)": mae_folds.std()
        })

        # Affichage du nom du modèle
        # print(f"\n{nom_modele}")

        # # Affichage de la MAE obtenue sur chaque fold
        # print("MAE de chaque fold :", mae_folds.round(2))

        # # Affichage de la MAE moyenne
        # print("MAE moyenne :",round(mae_folds.mean(), 2), "€/m²" )

        # # Affichage de l'écart-type
        # print("Écart-type :", round(mae_folds.std(), 2), "€/m²")

    # Transformation de la liste des résultats en DataFrame
    tableau_resultats = pd.DataFrame(resultats)

    # La fonction renvoie le tableau final
    return tableau_resultats

#######################################
# Modelisation baseline 
######################################

def entrainer_modeles(X_train, y_train):
    # Entraîne le DummyRegressor 
    modele_dummy = DummyRegressor(strategy="median")
    modele_dummy.fit(X_train, y_train)

    # Entraîne la régression linéaire
    modele_lr = Pipeline([("preprocessor", creer_preprocessor(X_train)), ("model", LinearRegression())])
    modele_lr.fit(X_train, y_train)

    return modele_dummy, modele_lr

def regression_ridge(X_train, y_train):

    # Créer un pipeline contenant le préprocesseur et la régression Ridge
    pipeline_ridge = Pipeline([("preprocessor", creer_preprocessor(X_train)), ("model", Ridge())])

    # Définir les différentes valeurs du paramètre alpha à tester
    parametres = {"model__alpha": [0.01, 0.1, 1, 10, 100]}

    # Rechercher le meilleur alpha avec une validation croisée en 5 folds
    recherche = GridSearchCV(estimator=pipeline_ridge, param_grid=parametres, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)

    # Entraîner et évaluer les différentes configurations sur les données d’entraînement
    recherche.fit(X_train, y_train)

    # Retourner le meilleur modèle, ses paramètres et sa MAE moyenne en validation croisée
    return recherche.best_estimator_, recherche.best_params_, -recherche.best_score_
 

#######################################
# Modelisation avancée 
######################################

def entrainer_XGBRegressor(X_train, y_train):
    # Entraîne un modèle XGBoost avec le préprocesseur.
    modele_xgb = Pipeline([("preprocessor", creer_preprocessor(X_train)),
                    ("model", XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1 
                    ))
    ])

    modele_xgb.fit(X_train, y_train)
    return modele_xgb

def optimiser_XGBRegressor(X_train, y_train):

    pipeline_xgb = Pipeline([
        ("preprocessor", creer_preprocessor(X_train)),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        ))
    ])

    parametres = {
        # Ton modèle actuel = 500
        "model__n_estimators": randint(450, 801),
        # Ton modèle actuel = 0.05
        "model__learning_rate": uniform(0.025, 0.04),
        # Ton modèle actuel = 6
        "model__max_depth": randint(4, 8),
        # Régularisation de l'arbre
        "model__min_child_weight": randint(1, 8),
        # Échantillonnage
        "model__subsample": uniform(0.75, 0.20),
        "model__colsample_bytree": uniform(0.75, 0.20),
        # Régularisation supplémentaire
        "model__reg_alpha": uniform(0, 0.5),
        "model__reg_lambda": uniform(0.5, 2),
        # Limite la création de divisions peu utiles
        "model__gamma": uniform(0, 0.3)
    }

    recherche_xgb = RandomizedSearchCV(
        estimator=pipeline_xgb,
        param_distributions=parametres,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    recherche_xgb.fit(X_train, y_train)

    meilleur_modele = recherche_xgb.best_estimator_
    meilleurs_parametres = recherche_xgb.best_params_
    mae_cv = -recherche_xgb.best_score_

    return meilleur_modele, meilleurs_parametres, mae_cv

def optimiser_XGBRegressor_2(X_train, y_train):

    pipeline_xgb = Pipeline([
        ("preprocessor", creer_preprocessor(X_train)),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        ))
    ])

    # Recherche resserrée autour des meilleurs paramètres obtenus
    parametres = {
        "model__n_estimators": randint(780, 951),
        "model__learning_rate": uniform(0.045, 0.03),   # 0.045 à 0.075
        "model__max_depth": [7, 8, 9],
        "model__min_child_weight": [1, 2],

        "model__subsample": uniform(0.78, 0.12),        # 0.78 à 0.90
        "model__colsample_bytree": uniform(0.75, 0.12),# 0.75 à 0.87

        "model__reg_alpha": uniform(0.20, 0.20),        # 0.20 à 0.40
        "model__reg_lambda": uniform(2.2, 1.0),         # 2.2 à 3.2

        "model__gamma": uniform(0.02, 0.10)             # 0.02 à 0.12
    }

    recherche_xgb = RandomizedSearchCV(
        estimator=pipeline_xgb,
        param_distributions=parametres,
        n_iter=20,                 # 40 au lieu de 20
        scoring="neg_mean_absolute_error",
        cv=3,                      # 5 folds au lieu de 3
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    recherche_xgb.fit(X_train, y_train)

    meilleur_modele = recherche_xgb.best_estimator_
    meilleurs_parametres = recherche_xgb.best_params_
    mae_cv = -recherche_xgb.best_score_

    print("Meilleurs paramètres :", meilleurs_parametres)
    print("MAE CV :", round(mae_cv, 2), "€/m²")

    return meilleur_modele, meilleurs_parametres, mae_cv

# def optimiser_XGBRegressor_2(X_train, y_train):

#     pipeline_xgb = Pipeline([
#         ("preprocessor", creer_preprocessor(X_train)),
#         ("model", XGBRegressor(
#             objective="reg:squarederror",
#             random_state=42,
#             n_jobs=1
#         ))
#     ])

#     parametres = {
#         "model__n_estimators": randint(700, 901),
#         "model__learning_rate": uniform(0.05, 0.02),
#         "model__max_depth": [6, 7, 8],
#         "model__min_child_weight": [1, 2, 3],
#         "model__subsample": uniform(0.75, 0.15),
#         "model__colsample_bytree": uniform(0.75, 0.15),
#         "model__reg_alpha": uniform(0.10, 0.20),
#         "model__reg_lambda": uniform(1.8, 1.2),
#         "model__gamma": uniform(0.05, 0.15)
#     }

#     recherche_xgb = RandomizedSearchCV(
#         estimator=pipeline_xgb,
#         param_distributions=parametres,
#         n_iter=20,
#         scoring="neg_mean_absolute_error",
#         cv=3,
#         random_state=42,
#         n_jobs=-1,
#         verbose=1
#     )

#     recherche_xgb.fit(X_train, y_train)

#     meilleur_modele = recherche_xgb.best_estimator_
#     meilleurs_parametres = recherche_xgb.best_params_
#     mae_cv = -recherche_xgb.best_score_

#     return meilleur_modele, meilleurs_parametres, mae_cv

def entrainer_random_forest(X_train, y_train):
    
    # Entraîne RandomForestRegressor    
    modele_rf = Pipeline([("preprocessor", creer_preprocessor(X_train)), ("model", RandomForestRegressor(n_estimators=100, max_depth=13, min_samples_leaf=5, random_state=42, n_jobs=-1))])
    modele_rf.fit(X_train, y_train)

    return modele_rf

def entrainer_CatBoostRegressor(X_train, y_train):
    # Entraîne un modèle CatBoost avec le préprocesseur.

    modele_catboost = Pipeline([
        ("preprocessor", creer_preprocessor(X_train)),

        ("model", CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            subsample=0.8, 
            rsm=0.8,
            bootstrap_type="Bernoulli",
            loss_function="RMSE",
            random_seed=42,
            thread_count=-1,
            verbose=0,
            allow_writing_files=False
        ))
    ])

    modele_catboost.fit(X_train, y_train)

    return modele_catboost
