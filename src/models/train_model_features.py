import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

def preparer_X_y(df, target="prix_m2"):

    # Variable cible
    y = df[target]

    # Variables explicatives
    X = df.drop(
        columns=[target, "valeur_fonciere"],
        errors="ignore"
    )

    return X, y


def creer_preprocessor(X_train):
    # Crée le preprocessing numérique et catégoriel
    variables_num = X_train.select_dtypes(include=["number"]).columns.tolist()
    variables_cat = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    pipeline_num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    pipeline_cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", pipeline_num, variables_num), ("cat", pipeline_cat, variables_cat)])


def creer_modele_XGB(X_train):

    modele_xgb = Pipeline([
        ("preprocessor", creer_preprocessor(X_train)),

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

    return modele_xgb

def calculer_mae_cv(X_train, y_train):

    modele = creer_modele_XGB(X_train)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        modele,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    mae_folds = -scores

    return mae_folds.mean(), mae_folds.std()

def tester_apport_features(X_train, y_train):

    resultats = []

    # --------------------------------------------------
    # MAE avec toutes les variables
    # --------------------------------------------------

    mae_reference, std_reference = calculer_mae_cv(
        X_train,
        y_train
    )

    print(
        "MAE avec toutes les features :",
        round(mae_reference, 2),
        "€/m²"
    )

    # --------------------------------------------------
    # Retirer chaque feature une par une
    # --------------------------------------------------

    for feature in X_train.columns:

        print("Test sans :", feature)

        X_train_sans_feature = X_train.drop(
            columns=[feature]
        )

        mae, std = calculer_mae_cv(
            X_train_sans_feature,
            y_train
        )

        difference = mae - mae_reference

        resultats.append({
            "Feature retirée": feature,
            "MAE CV": mae,
            "Écart-type": std,
            "Différence MAE": difference
        })

    resultats = pd.DataFrame(resultats)

    resultats = resultats.sort_values(
        by="Différence MAE",
        ascending=False
    )

    return resultats

def tester_groupes_features(X_train, y_train, groupes):

    resultats = []

    mae_reference, std_reference = calculer_mae_cv(
        X_train,
        y_train
    )

    for nom, colonnes in groupes.items():

        X_test_features = X_train.drop(
            columns=colonnes,
            errors="ignore"
        )

        mae, std = calculer_mae_cv(
            X_test_features,
            y_train
        )

        resultats.append({
            "Configuration": nom,
            "MAE CV": mae,
            "Écart-type": std,
            "Différence MAE": mae - mae_reference
        })

    return pd.DataFrame(resultats).sort_values(
        "MAE CV"
    )