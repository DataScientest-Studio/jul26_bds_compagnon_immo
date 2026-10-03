from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
# ============================================================
# CONFIGURATION ET CHEMINS
# ============================================================

RACINE_PROJET = Path(__file__).resolve().parents[2]
 
if str(RACINE_PROJET) not in sys.path:
    sys.path.append(str(RACINE_PROJET))

from src.features.build_features import creer_features_prediction

st.set_page_config(page_title="Compagnon Immobilier", page_icon="🏠", layout="wide")

# ============================================================
# CHARGEMENT DU RÉFÉRENTIEL + DONNÉES INSEE
# ============================================================

@st.cache_data
def charger_communes():
    chemin_referentiel = RACINE_PROJET / "data" / "processed" / "referentiel.csv"
    chemin_dvf_insee = RACINE_PROJET / "data" / "processed" / "dvf_insee_enrichi.csv"

    if not chemin_referentiel.exists():
        raise FileNotFoundError(f"Référentiel introuvable : {chemin_referentiel}")

    if not chemin_dvf_insee.exists():
        raise FileNotFoundError(f"Fichier DVF + INSEE introuvable : {chemin_dvf_insee}")

    communes = pd.read_csv(chemin_referentiel, sep=None, engine="python", dtype={"code_commune": "string"})
    communes.columns = communes.columns.str.strip()

    colonnes_referentiel = {"code_commune", "nom_commune", "latitude", "longitude"}
    colonnes_manquantes = colonnes_referentiel - set(communes.columns)

    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes dans referentiel.csv : {sorted(colonnes_manquantes)}")

    communes["code_commune"] = communes["code_commune"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)

    dvf_insee = pd.read_csv(chemin_dvf_insee, dtype={"code_commune": "string"})
    dvf_insee.columns = dvf_insee.columns.str.strip()
    dvf_insee["code_commune"] = dvf_insee["code_commune"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)

    colonnes_insee = ["code_commune", "niveau_vie_median", "densite_population", "population", "superficie_km2", "nb_logements", "nb_residences_principales", "nb_residences_secondaires", "nb_logements_vacants", "nb_proprietaires", "part_proprietaires", "part_logements_vacants", "nb_etablissements"]
    colonnes_insee_manquantes = [colonne for colonne in colonnes_insee if colonne not in dvf_insee.columns]

    if colonnes_insee_manquantes:
        raise ValueError(f"Colonnes manquantes dans dvf_insee_enrichi.csv : {colonnes_insee_manquantes}")

    donnees_insee_communes = dvf_insee[colonnes_insee].drop_duplicates(subset="code_commune").copy()
    communes = communes.merge(donnees_insee_communes, on="code_commune", how="left")
    communes = communes.dropna(subset=["code_commune", "nom_commune", "latitude", "longitude"]).copy()
    communes["libelle"] = communes["code_commune"].str.strip() + " — " + communes["nom_commune"].astype(str)
    communes = communes.sort_values("nom_commune").reset_index(drop=True)

    return communes, dvf_insee

# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def charger_modele():
    chemin_modele = RACINE_PROJET / "models" / "modele_xgb_final.joblib"

    if not chemin_modele.exists():
        raise FileNotFoundError(f"Modèle introuvable : {chemin_modele}")

    return joblib.load(chemin_modele)

# ============================================================
# EXÉCUTION DES CHARGEMENTS
# ============================================================

try:
    communes, dvf_insee = charger_communes()

except Exception as erreur:
    st.error("Les données des communes n’ont pas pu être chargées. Vérifiez referentiel.csv et dvf_insee_enrichi.csv dans data/processed.")
    st.exception(erreur)
    st.stop()

try:
    modele = charger_modele()

except Exception as erreur:
    st.error("Le modèle n’a pas pu être chargé. Vérifiez que modele_xgb_final.joblib se trouve dans models.")
    st.exception(erreur)
    st.stop()

# ============================================================
# TITRE
# ============================================================

st.title("🏠 Compagnon Immobilier")
st.write("Une application d’estimation du prix immobilier dans le département du Rhône.")

# ============================================================
# ONGLETS
# ============================================================

onglet_estimation, onglet_performances, onglet_projet = st.tabs(["🏠 Estimation", "📊 Performances", "ℹ️ À propos"])

# ============================================================
# ONGLET 1 : ESTIMATION
# ============================================================

with onglet_estimation:
    st.subheader("Estimer la valeur d’un logement")
    st.markdown('<div class="introduction">Renseignez les principales caractéristiques du logement pour obtenir une estimation de son prix au m² et de sa valeur totale.</div>', unsafe_allow_html=True)

    with st.form("formulaire_estimation"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            type_local = st.selectbox("Type de logement", options=["Appartement", "Maison"])

        with col2:
            libelle_commune = st.selectbox("Commune", options=communes["libelle"].tolist(), index=None, placeholder="Sélectionnez une commune")

        with col3:
            surface = st.number_input("Surface habitable (m²)", min_value=10.0, max_value=500.0, value=70.0, step=1.0)

        with col4:
            nombre_pieces = st.number_input("Nombre de pièces", min_value=1, max_value=20, value=3, step=1)

        bouton_estimation = st.form_submit_button("🔍 Estimer le logement", type="primary")

    if bouton_estimation:
        if libelle_commune is None:
            st.error("Veuillez sélectionner une commune.")

        else:
            try:
                commune_selectionnee = communes.loc[communes["libelle"] == libelle_commune].iloc[0]
                code_commune = str(commune_selectionnee["code_commune"])
                latitude = float(commune_selectionnee["latitude"])
                longitude = float(commune_selectionnee["longitude"])
                niveau_vie_median = float(commune_selectionnee["niveau_vie_median"])

                logement = pd.DataFrame()
                logement["code_commune"] = [code_commune]
                logement["type_local"] = [type_local]
                logement["surface_reelle_bati"] = [float(surface)]
                logement["nombre_pieces_principales"] = [int(nombre_pieces)]
                logement["surface_terrain"] = [np.nan]
                logement["lot1_surface_carrez"] = [np.nan]
                logement["nombre_lots"] = [np.nan]
                logement["latitude"] = [latitude]
                logement["longitude"] = [longitude]
                logement["annee_mutation"] = [2025]
                logement["mois_mutation"] = [9]
                logement["niveau_vie_median"] = [commune_selectionnee["niveau_vie_median"]]
                logement["densite_population"] = [commune_selectionnee["densite_population"]]
                logement["population"] = [commune_selectionnee["population"]]
                logement["superficie_km2"] = [commune_selectionnee["superficie_km2"]]
                logement["nb_logements"] = [commune_selectionnee["nb_logements"]]
                logement["nb_residences_principales"] = [commune_selectionnee["nb_residences_principales"]]
                logement["nb_residences_secondaires"] = [commune_selectionnee["nb_residences_secondaires"]]
                logement["nb_logements_vacants"] = [commune_selectionnee["nb_logements_vacants"]]
                logement["nb_proprietaires"] = [commune_selectionnee["nb_proprietaires"]]
                logement["part_proprietaires"] = [commune_selectionnee["part_proprietaires"]]
                logement["part_logements_vacants"] = [commune_selectionnee["part_logements_vacants"]]
                logement["nb_etablissements"] = [commune_selectionnee["nb_etablissements"]]

                logement = creer_features_prediction(logement)

                if "terrain_renseigne" not in logement.columns:
                    logement["terrain_renseigne"] = logement["surface_terrain"].notna().astype(int)

                prix_m2 = float(modele.predict(logement)[0])
                prix_total = prix_m2 * surface

                st.markdown('<p class="resultat-titre">Résultat de l’estimation</p>', unsafe_allow_html=True)
                # st.success("✅ Estimation réalisée avec succès")

                resultat1, resultat2 = st.columns(2)

                with resultat1:
                    st.metric("Prix estimé au m²", f"{prix_m2:,.0f} €/m²")

                with resultat2:
                    st.metric("Valeur totale estimée", f"{prix_total:,.0f} €")

                transactions_commune = dvf_insee[(dvf_insee["code_commune"] == code_commune) & (dvf_insee["type_local"] == type_local)].copy()

                # Transactions de la commune
                if not transactions_commune.empty:
                    prix_median_commune = transactions_commune["prix_m2"].median()
                    nombre_transactions = len(transactions_commune)
                    ecart_marche = ((prix_m2 - prix_median_commune) / prix_median_commune) * 100

                    st.markdown("### 📊 Positionnement par rapport au marché local")

                    comparaison_marche = pd.DataFrame({
                        "Indicateur": ["Estimation du modèle", "Médiane du marché local"],
                        "Prix": [prix_m2, prix_median_commune]
                    })

                    graphique_marche = alt.Chart(comparaison_marche).mark_bar(cornerRadiusEnd=6, size=28).encode(
                        y=alt.Y("Indicateur:N", sort=None, title="", axis=alt.Axis(labelLimit=250)),
                        x=alt.X("Prix:Q", title="Prix au m² (€)", scale=alt.Scale(domain=[0, max(prix_m2, prix_median_commune) * 1.15])),
                        tooltip=["Indicateur", alt.Tooltip("Prix:Q", title="Prix au m²", format=",.0f")]
                    ).properties(height=140)

                    comparaison_marche["Prix_affiche"] = comparaison_marche["Prix"].apply(lambda x: f"{x:,.0f} €/m²")

                    valeurs = alt.Chart(comparaison_marche).mark_text(align="left", dx=5, fontSize=14, fontWeight="bold").encode(
                        y=alt.Y("Indicateur:N", sort=None),
                        x="Prix:Q",
                        text="Prix_affiche:N"
                    )

                    st.altair_chart(graphique_marche + valeurs, use_container_width=True)

                    if ecart_marche > 0:
                        st.info(f"📈 L'estimation se situe à **{ecart_marche:.1f} % au-dessus** de la médiane locale.")
                    elif ecart_marche < 0:
                        st.info(f"📉 L'estimation se situe à **{abs(ecart_marche):.1f} % en dessous** de la médiane locale.")
                    else:
                        st.info("L'estimation est proche de la médiane locale.")

                    st.caption(f"Comparaison basée sur {nombre_transactions} transactions DVF de type « {type_local} » disponibles dans cette commune.")
                
                else:
                    st.info("Aucune transaction comparable disponible pour cette commune.")

                with st.expander("📍 Voir le contexte de la commune", expanded=True):
                    st.write(f"**Code INSEE :** {code_commune}")
                    st.write(f"**Niveau de vie médian :** {niveau_vie_median:,.0f} €")

                    if pd.notna(commune_selectionnee["population"]):
                        st.write(f"**Population :** {commune_selectionnee['population']:,.0f}")

                    if pd.notna(commune_selectionnee["densite_population"]):
                        st.write(f"**Densité de population :** {commune_selectionnee['densite_population']:,.0f} hab./km²")

                    st.write(f"**Distance par rapport à Lyon :** {logement['distance_lyon_km'].iloc[0]:.2f} km")
                    # st.write(f"**Latitude :** {latitude:.6f}")
                    # st.write(f"**Longitude :** {longitude:.6f}")

                st.caption(f"Estimation pour {libelle_commune}. Cette estimation est indicative et ne remplace pas l’évaluation d’un professionnel.")

            except Exception as erreur:
                st.error("Une erreur est survenue pendant l’estimation.")
                st.exception(erreur)

# ============================================================
# ONGLET 2 : PERFORMANCES
# ============================================================

with onglet_performances:
    st.subheader("Performances du modèle final")
    st.write("Le modèle retenu est un **XGBoost optimisé**. Il a été évalué sur un jeu de test qui n’a pas été utilisé pendant son entraînement.")

    metrique1, metrique2, metrique3 = st.columns(3)
    metrique1.metric("MAE", "642,13 €/m²")
    metrique2.metric("RMSE", "921,15 €/m²")
    metrique3.metric("R²", "0,63")

    st.info("La MAE signifie que l’écart absolu moyen entre le prix réel et le prix prédit est d’environ 642 €/m².")
   
    st.markdown("### Évolution des performances de XGBoost")

    comparaison_modeles = pd.DataFrame({
        "Modèle": ["XGBoost initial", "XGBoost optimisé", "XGBoost optimisé 2"],
        "MAE": [673.98, 649.39, 642.13]
    })

    graphique = alt.Chart(comparaison_modeles).mark_line(point=True).encode(
        x=alt.X("Modèle:N", sort=None, title="", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("MAE:Q", scale=alt.Scale(domain=[600, 700]), title="MAE (€/m²)"),
        tooltip=["Modèle", alt.Tooltip("MAE:Q", title="MAE", format=".2f")]
    )

    st.altair_chart(graphique, use_container_width=True)

    st.caption("Évolution de la MAE : 673,98 → 649,39 → 642,13 €/m². Plus la MAE est faible, meilleure est la précision moyenne du modèle.")


    st.markdown("### Explicabilité du modèle avec SHAP")

    st.write(
        "Ce graphique montre comment les principales variables influencent les prédictions du modèle. "
        "Les variables sont classées de la plus influente à la moins influente. "
        "Chaque point représente une observation : le **rouge correspond à une valeur élevée de la variable** "
        "et le **bleu à une valeur faible**. "
        "Un impact SHAP positif augmente le prix au m² prédit, tandis qu'un impact négatif le diminue."
    )

    st.info(
        "📊 **Interprétation :** la distance à Lyon et la surface habitable sont les variables les plus influentes. "
        "Une distance élevée à Lyon tend globalement à diminuer le prix au m² prédit, tandis que les biens proches de Lyon "
        "tendent à avoir un impact positif. Les grandes surfaces habitables tendent également à réduire le prix au m². "
        "À l'inverse, un niveau de vie médian élevé dans la commune tend à augmenter le prix prédit. "
        "La surface du terrain, l'année de mutation et le nombre de lots contribuent également aux estimations."
    )

    chemin_shap = RACINE_PROJET / "reports" / "figures" / "shap.png"

    if chemin_shap.exists():
        st.image(str(chemin_shap), caption="Interprétation des prédictions avec SHAP", use_container_width=True)

    else:
        st.warning("Le graphique shap.png est introuvable.")

# ============================================================
# ONGLET 3 : À PROPOS
# ============================================================

with onglet_projet:
    st.subheader("À propos de Compagnon Immobilier")
    st.write("**Compagnon Immobilier** est un projet de Data Science dont l’objectif est d’estimer le prix au m² d’un logement situé dans le département du Rhône.")

    st.markdown("### Données utilisées")
    st.write("Le projet repose principalement sur les données **DVF (Demandes de valeurs foncières)**, qui décrivent les transactions immobilières réalisées dans le Rhône.")
    st.write("Ces données ont été enrichies avec des **indicateurs INSEE à l’échelle communale**, afin d’intégrer au modèle des informations sur le contexte socio-économique et territorial des communes.")

    st.markdown("### Informations prises en compte")
    st.markdown("- Type de logement\n- Surface habitable\n- Nombre de pièces principales\n- Commune\n- Coordonnées géographiques\n- Surface du terrain\n- Distance par rapport au centre de Lyon\n- Année et mois de la transaction\n- Niveau de vie médian\n- Population et densité\n- Caractéristiques du parc de logements\n- Nombre d’établissements")

    st.markdown("### Enrichissement avec les données INSEE")
    st.info("L’enrichissement avec les données INSEE permet de compléter les caractéristiques propres au logement par des informations sur son environnement communal. L’objectif est de mieux représenter les différences territoriales susceptibles d’influencer les prix immobiliers.")

    st.markdown("### Démarche de modélisation")
    st.write("Plusieurs modèles ont été entraînés et comparés : DummyRegressor, régression linéaire, Ridge, Random Forest, CatBoost et XGBoost.")
    st.write("Le modèle **XGBoost optimisé** a été retenu, car il présente les meilleures performances parmi les modèles étudiés.")

    st.markdown("### Fonctionnement de l’application")
    st.write("L’application charge directement le pipeline déjà entraîné depuis le fichier `modele_xgb_final.joblib`.")
    st.write("Elle ne réentraîne donc pas le modèle. Elle récupère les informations saisies par l’utilisateur et les complète avec les indicateurs INSEE de la commune avant d’effectuer la prédiction.")

    # st.warning("Les résultats proposés par cette application sont des estimations et doivent être interprétés avec prudence.")
    st.markdown("---")
    st.caption("Projet réalisé dans le cadre de la formation Data Scientist par Kawthar CHOUKRI et Luc MARTIAS.")

# ============================================================
# STYLE CSS
# ============================================================

st.markdown("""
<style>
.stApp { background-color: #F7F9FC; }
.block-container { max-width: 1050px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stForm"] { background-color: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 25px; box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06); }
.stButton > button, [data-testid="stFormSubmitButton"] > button { width: 100%; background-color: #E85D4A; color: white; border: none; border-radius: 8px; font-weight: 600; padding: 0.7rem; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { background-color: #CC4938; color: white; border: none; }
[data-testid="stMetric"] { background-color: white; border: 1px solid #E2E8F0; border-left: 5px solid #E85D4A; border-radius: 12px; padding: 18px; box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05); }
.stTabs [data-baseweb="tab-list"] { gap: 12px; }
.stTabs [data-baseweb="tab"] { background-color: white; border-radius: 8px 8px 0 0; padding: 10px 18px; }
.stTabs [aria-selected="true"] { color: #E85D4A; font-weight: 700; }
.introduction { background: linear-gradient(135deg, #FFF3F0, #FFFFFF); border-left: 5px solid #E85D4A; border-radius: 12px; padding: 18px; margin-bottom: 22px; }
.resultat-titre { color: #1E293B; font-size: 1.3rem; font-weight: 700; margin-top: 25px; }
</style>
""", unsafe_allow_html=True)