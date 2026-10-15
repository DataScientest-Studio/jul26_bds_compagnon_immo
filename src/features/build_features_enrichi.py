import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.neighbors import BallTree



#--------------------------------------------------------------------------------------------------------------------------

    ### AJOUT
def verifier_structure_dvf(dvf):
    """Contrôles légers avant feature engineering."""
    colonnes_min = {
        "id_mutation", "date_mutation", "id_parcelle", "valeur_fonciere",
        "type_local", "surface_reelle_bati", "latitude", "longitude",
    }
    manquantes = colonnes_min - set(dvf.columns)
    if manquantes:
        raise ValueError("Colonnes indispensables manquantes : "f"{sorted(manquantes)}")

    if dvf["id_mutation"].duplicated().any():
        raise ValueError(
            "Le dataset contient encore plusieurs lignes par id_mutation. "
            "Appliquez d'abord la consolidation du preprocessing DVF."
        )

    return dvf.copy()



#--------------------------------------------------------------------------------------------------------------------------



def creer_features_enrichi(dvf):
    """Reprend les features initiales et ajoute les variables structurelles."""
    dvf = verifier_structure_dvf(dvf)

    # 1. Variables temporelles
    dvf["date_mutation"] = pd.to_datetime(dvf["date_mutation"])
    dvf["annee_mutation"] = dvf["date_mutation"].dt.year
    dvf["mois_mutation"] = dvf["date_mutation"].dt.month


#--------------------------------------------------------------------------------------------------------------------------

    ### AJOUT
    dvf["trimestre_mutation"] = dvf["date_mutation"].dt.quarter
    dvf["mois_sin"] = np.sin(2 * np.pi * dvf["mois_mutation"] / 12)
    dvf["mois_cos"] = np.cos(2 * np.pi * dvf["mois_mutation"] / 12)


#--------------------------------------------------------------------------------------------------------------------------


    # 2. Prix au m²
    dvf["prix_m2"] = (dvf["valeur_fonciere"] / dvf["surface_reelle_bati"])

    # 3. Surface par pièce
    # dvf.loc[dvf["nombre_pieces_principales"] <= 0, "nombre_pieces_principales"] = np.nan
    dvf["surface_par_piece"] = (dvf["surface_reelle_bati"] / dvf["nombre_pieces_principales"])
    dvf["log_surface"] = np.log1p(dvf["surface_reelle_bati"])



#--------------------------------------------------------------------------------------------------------------------------
     ### AJOUT
    dvf["log_surface_terrain"] = np.where(dvf["surface_terrain"] > 0,np.log1p(dvf["surface_terrain"]),np.nan)
     ### AJOUT
    dvf["terrain_bati_ratio"] = (dvf["surface_terrain"] / dvf["surface_reelle_bati"])

    ### AJOUT
    dvf["pieces_par_m2"] = (dvf["nombre_pieces_principales"] / dvf["surface_reelle_bati"])
    ### AJOUT
    dvf["has_terrain"] = (dvf["surface_terrain"].fillna(0) > 0).astype(int)

    ### AJOUT
    dvf["has_lot"] = (dvf["nombre_lots"].fillna(0) > 0).astype(int)
    
    ### AJOUT
    carrez_cols = [c for c in ["lot1_surface_carrez", "lot2_surface_carrez","lot3_surface_carrez", "lot4_surface_carrez","lot5_surface_carrez",]if c in dvf.columns]
    if carrez_cols:
        dvf["surface_carrez_totale"] = dvf[carrez_cols].sum(axis=1, min_count=1)
        dvf["has_carrez"] = (dvf["surface_carrez_totale"].fillna(0) > 0).astype(int)
        dvf["carrez_bati_ratio"] = dvf["surface_carrez_totale"] / dvf["surface_reelle_bati"]
        dvf["ecart_carrez_bati"] = dvf["surface_carrez_totale"] - dvf["surface_reelle_bati"]
    else:
        dvf["surface_carrez_totale"] = np.nan
        dvf["has_carrez"] = 0
        dvf["carrez_bati_ratio"] = np.nan
        dvf["ecart_carrez_bati"] = np.nan


    ### AJOUT
    if "nature_culture" in dvf.columns:
        dvf["has_nature_culture"] = dvf["nature_culture"].notna().astype(int)
        nature = dvf["nature_culture"].fillna("").str.lower()
        dvf["nature_jardin"] = nature.eq("jardins").astype(int)
        dvf["nature_terrain_agrement"] = nature.eq("terrains d'agrément").astype(int)

    ### AJOUT
    if "adresse_code_voie" in dvf.columns:
        dvf["has_code_voie"] = dvf["adresse_code_voie"].notna().astype(int)


# ---------------------------------------------------------------


    # 4. Distance par rapport au centre de Lyon
    # https://www.countrycoordinate.com/city-lyon-france/
    lyon = (45.7640, 4.8357)
    dvf["distance_lyon_km"] = dvf.apply(
        lambda ligne: geodesic(
            (ligne["latitude"], ligne["longitude"]), lyon
        ).km
        if pd.notna(ligne["latitude"])
        and pd.notna(ligne["longitude"])
        else np.nan,
        axis=1,
    )

    # 5. Remplacer les valeurs infinies par des valeurs manquantes
    dvf = dvf.replace([np.inf, -np.inf], np.nan)

    return dvf


#--------------------------------------------------------------------------------------------------------------------------


### AJOUT
def ajouter_features_equipements_osm(dvf, osm):
    """Ajoute des variables décrivant les équipements présents à proximité de chaque transaction DVF."""
    dvf = dvf.copy()
    osm = osm.copy()

    # Vérifier que les coordonnées existent
    if "latitude" not in dvf.columns or "longitude" not in dvf.columns:
        raise ValueError("Les coordonnées latitude/longitude sont nécessaires dans DVF.")

    if "latitude" not in osm.columns or "longitude" not in osm.columns:
        raise ValueError("Les coordonnées latitude/longitude sont nécessaires dans OSM.")

    # Garder uniquement les équipements ayant des coordonnées
    osm = osm[osm["latitude"].notna() & osm["longitude"].notna()].copy()

    # Coordonnées OSM en radians
    coordonnees_osm = np.radians(osm[["latitude", "longitude"]].values)

    # Coordonnées DVF en radians
    coordonnees_dvf = np.radians(dvf[["latitude", "longitude"]].fillna(0).values)

    # Rayons de recherche
    # rayon_100m = 100 / 6_371_000
    rayon_500m = 500 / 6_371_000
    # rayon_1000m = 1000 / 6_371_000

    #---------------------------------------------------------
    ## Crèches et petite enfance
    #---------------------------------------------------------

    creches = osm["amenity"].isin([
        "childcare",
        "kindergarten"
    ])

    if creches.sum() > 0:
        arbre_creches = BallTree(coordonnees_osm[creches.values], metric="haversine")

        # dvf["creches_100m"] = arbre_creches.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_100m,
        #     count_only=True
        # )

        dvf["creches_500m"] = arbre_creches.query_radius(
            coordonnees_dvf,
            r=rayon_500m,
            count_only=True
        )

        # dvf["creches_1000m"] = arbre_creches.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_1000m,
        #     count_only=True
        # )

        distances_creches, _ = arbre_creches.query(coordonnees_dvf, k=1)

        # Conversion des distances angulaires en distance mètres => multiplication par rayon moyen de la Terre en mètres (6_371_000)
        dvf["distance_creches_min"] = (distances_creches[:, 0] * 6_371_000)

    else:
        # dvf["creches_100m"] = 0
        dvf["creches_500m"] = 0
        # dvf["creches_1000m"] = 0
        dvf["distance_creches_min"] = np.nan

    #---------------------------------------------------------
    ## Écoles et collèges
    #---------------------------------------------------------

    ecoles = osm["amenity"].isin([
        "school",
        "college"
    ])

    if ecoles.sum() > 0:
        arbre_ecoles = BallTree(coordonnees_osm[ecoles.values], metric="haversine")

        # dvf["ecoles_100m"] = arbre_ecoles.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_100m,
        #     count_only=True
        # )

        dvf["ecoles_500m"] = arbre_ecoles.query_radius(
            coordonnees_dvf,
            r=rayon_500m,
            count_only=True
        )

        # dvf["ecoles_1000m"] = arbre_ecoles.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_1000m,
        #     count_only=True
        # )

        distances_ecoles, _ = arbre_ecoles.query(coordonnees_dvf, k=1)

        # Conversion des distances angulaires en distance mètres => multiplication par rayon moyen de la Terre en mètres (6_371_000)
        dvf["distance_ecoles_min"] = (distances_ecoles[:, 0] * 6_371_000)

    else:
        # dvf["ecoles_100m"] = 0
        dvf["ecoles_500m"] = 0
        # dvf["ecoles_1000m"] = 0
        dvf["distance_ecoles_min"] = np.nan

    #---------------------------------------------------------
    ## Commerces essentiels
    #---------------------------------------------------------

    commerces_essentiels = osm["shop"].isin([
        "supermarket",
        "convenience",
        "grocery",
        "bakery",
        "butcher",
        "greengrocer",
        "deli",
        "department_store",
        "mall"
    ])

    if commerces_essentiels.sum() > 0:
        arbre_commerces_essentiels = BallTree(coordonnees_osm[commerces_essentiels.values], metric="haversine")

        # dvf["commerces_essentiels_100m"] = (
        #     arbre_commerces_essentiels.query_radius(
        #         coordonnees_dvf,
        #         r=rayon_100m,
        #         count_only=True
        #     )
        # )

        dvf["commerces_essentiels_500m"] = (
            arbre_commerces_essentiels.query_radius(
                coordonnees_dvf,
                r=rayon_500m,
                count_only=True
            )
        )

        # dvf["commerces_essentiels_1000m"] = (
        #     arbre_commerces_essentiels.query_radius(
        #         coordonnees_dvf,
        #         r=rayon_1000m,
        #         count_only=True
        #     )
        # )

        distances_commerces, _ = arbre_commerces_essentiels.query(coordonnees_dvf, k=1)

        # Conversion des distances angulaires en distance mètres => multiplication par rayon moyen de la Terre en mètres (6_371_000)
        dvf["distance_commerces_essentiels_min"] = (distances_commerces[:, 0] * 6_371_000)

    else:
        # dvf["commerces_essentiels_100m"] = 0
        dvf["commerces_essentiels_500m"] = 0
        # dvf["commerces_essentiels_1000m"] = 0
        dvf["distance_commerces_essentiels_min"] = np.nan

    #---------------------------------------------------------
    ## Restaurants et cafés
    #---------------------------------------------------------

    restaurants = osm["amenity"].isin([
        "restaurant",
        "cafe",
        "fast_food",
        "pub",
        "bar"
    ])

    if restaurants.sum() > 0:
        arbre_restaurants = BallTree(coordonnees_osm[restaurants.values], metric="haversine")

        # dvf["restaurants_100m"] = arbre_restaurants.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_100m,
        #     count_only=True
        # )

        dvf["restaurants_500m"] = arbre_restaurants.query_radius(
            coordonnees_dvf,
            r=rayon_500m,
            count_only=True
        )

        # dvf["restaurants_1000m"] = arbre_restaurants.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_1000m,
        #     count_only=True
        # )

    else:
        # dvf["restaurants_100m"] = 0
        dvf["restaurants_500m"] = 0
        # dvf["restaurants_1000m"] = 0

    #---------------------------------------------------------
    ## Santé
    #---------------------------------------------------------

    sante = (
        osm["healthcare"].notna()
        | osm["amenity"].isin([
            "pharmacy",
            "doctors",
            "dentist",
            "clinic",
            "hospital"
        ])
    )

    if sante.sum() > 0:
        arbre_sante = BallTree(coordonnees_osm[sante.values], metric="haversine")

        # dvf["sante_100m"] = arbre_sante.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_100m,
        #     count_only=True
        # )

        dvf["sante_500m"] = arbre_sante.query_radius(
            coordonnees_dvf,
            r=rayon_500m,
            count_only=True
        )

        # dvf["sante_1000m"] = arbre_sante.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_1000m,
        #     count_only=True
        # )

    else:
        # dvf["sante_100m"] = 0
        dvf["sante_500m"] = 0
        # dvf["sante_1000m"] = 0

    #---------------------------------------------------------
    ## Transports
    #---------------------------------------------------------

    transports = (
        osm["public_transport"].notna()
        | osm["railway"].notna()
        | osm["highway"].eq("bus_stop")
    )

    if transports.sum() > 0:
        arbre_transports = BallTree(coordonnees_osm[transports.values], metric="haversine")

        # dvf["transports_100m"] = arbre_transports.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_100m,
        #     count_only=True
        # )

        dvf["transports_500m"] = arbre_transports.query_radius(
            coordonnees_dvf,
            r=rayon_500m,
            count_only=True
        )

        # dvf["transports_1000m"] = arbre_transports.query_radius(
        #     coordonnees_dvf,
        #     r=rayon_1000m,
        #     count_only=True
        # )

        distances_transports, _ = arbre_transports.query(
            coordonnees_dvf,
            k=1
        )
        
        # Conversion des distances angulaires en distance mètres => multiplication par rayon moyen de la Terre en mètres (6_371_000)
        dvf["distance_transports_min"] = (distances_transports[:, 0] * 6_371_000)

    else:
        # dvf["transports_100m"] = 0
        dvf["transports_500m"] = 0
        # dvf["transports_1000m"] = 0
        dvf["distance_transports_min"] = np.nan

    #---------------------------------------------------------
    ## Sports et loisirs
    #----------------------------------------------------------

    sports_loisirs = (
        osm["leisure"].isin([
            "sports_centre",
            "sports_hall",
            "fitness_centre",
            "stadium",
            "pitch",
            "fitness_station"
        ])
        | osm["sport"].notna()
    )

    if sports_loisirs.sum() > 0:
        arbre_sports_loisirs = BallTree(coordonnees_osm[sports_loisirs.values], metric="haversine")

        # dvf["sports_loisirs_100m"] = (
        #     arbre_sports_loisirs.query_radius(
        #         coordonnees_dvf,
        #         r=rayon_100m,
        #         count_only=True
        #     )
        # )

        dvf["sports_loisirs_500m"] = (
            arbre_sports_loisirs.query_radius(
                coordonnees_dvf,
                r=rayon_500m,
                count_only=True
            )
        )

        # dvf["sports_loisirs_1000m"] = (
        #     arbre_sports_loisirs.query_radius(
        #         coordonnees_dvf,
        #         r=rayon_1000m,
        #         count_only=True
        #     )
        # )

    else:
        # dvf["sports_loisirs_100m"] = 0
        dvf["sports_loisirs_500m"] = 0
        # dvf["sports_loisirs_1000m"] = 0

    # ---------------------------------------------------------
    ## Espaces verts
    # ---------------------------------------------------------

    espaces_verts = (
        osm["leisure"].isin([
            "park",
            "garden",
            "playground",
            "dog_park"
        ])
        | osm["landuse"].isin([
            "forest",
            "grass",
            "recreation_ground"
        ])
    )

    if espaces_verts.sum() > 0:
        arbre_espaces_verts = BallTree(coordonnees_osm[espaces_verts.values], metric="haversine")

        # dvf["espaces_verts_100m"] = (
        #     arbre_espaces_verts.query_radius(
        #         coordonnees_dvf,
        #         r=rayon_100m,
        #         count_only=True
        #     )
        # )

        dvf["espaces_verts_500m"] = (
            arbre_espaces_verts.query_radius(
                coordonnees_dvf,
                r=rayon_500m,
                count_only=True
            )
        )

        # dvf["espaces_verts_1000m"] = (
        #     arbre_espaces_verts.query_radius(
        #         coordonnees_dvf,
        #         r=rayon_1000m,
        #         count_only=True
        #     )
        # )

        distances_espaces_verts, _ = arbre_espaces_verts.query(coordonnees_dvf, k=1)

        # Conversion des distances angulaires en distance mètres => multiplication par rayon moyen de la Terre en mètres (6_371_000)
        dvf["distance_espaces_verts_min"] = (distances_espaces_verts[:, 0] * 6_371_000)

    else:
        # dvf["espaces_verts_100m"] = 0
        dvf["espaces_verts_500m"] = 0
        # dvf["espaces_verts_1000m"] = 0
        dvf["distance_espaces_verts_min"] = np.nan

    # Les transactions sans coordonnées ne peuvent pas être enrichies
    coordonnees_absentes = dvf["latitude"].isna() | dvf["longitude"].isna()
    
    colonnes_equipements = [
        # "creches_100m",
        "creches_500m",
        # "creches_1000m",
        # "ecoles_100m",
        "ecoles_500m",
        # "ecoles_1000m",
        # "commerces_essentiels_100m",
        "commerces_essentiels_500m",
        # "commerces_essentiels_1000m",
        # "restaurants_100m",
        "restaurants_500m",
        # "restaurants_1000m",
        # "sante_100m",
        "sante_500m",
        # "sante_1000m",
        # "transports_100m",
        "transports_500m",
        # "transports_1000m",
        # "sports_loisirs_100m",
        "sports_loisirs_500m",
        # "sports_loisirs_1000m",
        # "espaces_verts_100m",
        "espaces_verts_500m",
        # "espaces_verts_1000m",
        "distance_creches_min",
        "distance_ecoles_min",
        "distance_transports_min",
        "distance_commerces_essentiels_min",
        "distance_espaces_verts_min"
    ]

    dvf.loc[coordonnees_absentes, colonnes_equipements] = np.nan

    return dvf


#--------------------------------------------------------------------------------------------------------------------------


def detecter_prix_extremes(dvf):

    """Tukey sur log1p(prix_m2), séparément pour Maison/Appartement."""

    # Détecte les prix au m² extrêmes avec la règle de Tukey
    # appliquée au logarithme du prix
    # (La règle de Tukey => méthode statistique utilisée pour détecter les valeurs extrêmes à partir des quartiles.)

    # Le calcul est réalisé séparément pour les maisons
    # et les appartements.

    # Retourne :
    # - le DataFrame filtré ;
    # - les observations extrêmes ;
    # - le tableau des seuils.

    dvf = dvf.copy()

    # 1. Vérifier que prix_m2 existe
    if "prix_m2" not in dvf.columns:
        raise ValueError("Exécutez d'abord creer_features_enrichi().")

    # 2. Conserver uniquement les prix valides
    prix_valides = dvf["prix_m2"].notna() & np.isfinite(dvf["prix_m2"]) & (dvf["prix_m2"] > 0)
    dvf = dvf.loc[prix_valides].copy()

    # 3. Transformer le prix au m² en logarithme
    dvf["log_prix_m2"] = np.log1p(dvf["prix_m2"])

    # 4. Initialiser les résultats
    masque_valide = pd.Series(False, index=dvf.index)
    seuils = []

    # 5. Calculer les seuils séparément pour chaque type de logement
    for type_bien, groupe in dvf.groupby("type_local"):

        q1 = groupe["log_prix_m2"].quantile(0.25)
        q3 = groupe["log_prix_m2"].quantile(0.75)
        iqr = q3 - q1

        borne_basse_log = q1 - 1.5 * iqr
        borne_haute_log = q3 + 1.5 * iqr

        borne_basse = np.expm1(borne_basse_log)
        borne_haute = np.expm1(borne_haute_log)

        masque_type = (dvf["type_local"] == type_bien) & dvf["prix_m2"].between(borne_basse, borne_haute)
        masque_valide |= masque_type

        nombre_extremes = (~masque_type.loc[groupe.index]).sum()

        seuils.append({
            "Type": type_bien,
            "Borne basse (€/m²)": borne_basse,
            "Borne haute (€/m²)": borne_haute,
            "Nombre total": len(groupe),
            "Valeurs extrêmes": nombre_extremes
        })

    # 6. Séparer les données conservées et les valeurs extremes
    dvf_filtre = dvf.loc[masque_valide].copy()
    valeurs_extremes = dvf.loc[~masque_valide].copy()

    # 7. Supprimer la variable temporaire
    dvf_filtre = dvf_filtre.drop(columns="log_prix_m2")
    valeurs_extremes = valeurs_extremes.drop(columns="log_prix_m2")

    # 8. Créer le tableau récapitulatif des seuils
    tableau_seuils = pd.DataFrame(seuils)

    return dvf_filtre, valeurs_extremes, tableau_seuils


#--------------------------------------------------------------------------------------------------------------------------


### AJOUT

def ajouter_historique_parcelle(dvf):
    """Ajoute des informations sur les ventes précédentes de la parcelle."""
    dvf = dvf.copy()
    colonnes = {"id_parcelle", "date_mutation", "prix_m2"}
    manquantes = colonnes - set(dvf.columns)
    if manquantes:
        raise ValueError(f"Historique parcelle impossible : {sorted(manquantes)}")

    # On remet les ventes dans l'ordre chronologique pour chaque parcelle.
    dvf["_ordre_original"] = range(len(dvf))
    dvf = dvf.sort_values(["id_parcelle", "date_mutation", "_ordre_original"])

    groupe = dvf.groupby("id_parcelle", sort=False)

    # Prix et date de la vente précédente.
    dvf["previous_price_m2"] = groupe["prix_m2"].shift(1)
    dvf["previous_sale_date"] = groupe["date_mutation"].shift(1)

    # Une vente le même jour ne compte pas comme une vente passée.
    meme_jour = dvf["previous_sale_date"].dt.normalize()== dvf["date_mutation"].dt.normalize()
    dvf.loc[meme_jour, ["previous_price_m2", "previous_sale_date"]] = [np.nan, pd.NaT]

    # Temps écoulé depuis la vente précédente.
    dvf["years_since_previous_sale"] = ( dvf["date_mutation"] - dvf["previous_sale_date"]).dt.days / 365.25

    cumcount = groupe.cumcount()
    same_day_rank = dvf.groupby(["id_parcelle", dvf["date_mutation"].dt.normalize()]).cumcount()

    # Nombre de ventes connues avant la vente actuelle.
    dvf["parcel_transaction_count"] = cumcount - same_day_rank

    # Evolution entre les deux dernières ventes connues.
    # On n'utilise pas le prix de la vente actuelle pour éviter une fuite de cible.
    price_before_previous = groupe["prix_m2"].shift(2)
    dvf["parcel_price_growth"] = (dvf["previous_price_m2"] / price_before_previous - 1).replace([np.inf, -np.inf], np.nan)

    return dvf.sort_values("_ordre_original").drop(columns="_ordre_original")


### AJOUT

def ajouter_historique_voie(dvf):
    """Ajoute le niveau de prix historique observé sur la même voie."""
    dvf = dvf.copy()
    colonnes = {"adresse_code_voie", "date_mutation", "prix_m2"}

    if not colonnes.issubset(dvf.columns):
        return dvf

    # On regroupe par jour pour ne pas utiliser les ventes du même jour.
    dvf["_jour"] = dvf["date_mutation"].dt.normalize()
    jours = (
        dvf.groupby(["adresse_code_voie", "_jour"], dropna=False)["prix_m2"]
        .agg(prix_median="median", nombre_ventes="count")
        .reset_index()
        .sort_values(["adresse_code_voie", "_jour"])
    )

    groupe = jours.groupby("adresse_code_voie", sort=False)
    jours["street_previous_median_price_m2"] = groupe["prix_median"].transform(lambda serie: serie.shift(1).expanding().median())
    jours["street_previous_transaction_count"] = groupe["nombre_ventes"].transform(lambda serie: serie.shift(1).cumsum()).fillna(0).astype(int)

    dvf = dvf.merge(jours[["adresse_code_voie", "_jour", "street_previous_median_price_m2", "street_previous_transaction_count"]], on=["adresse_code_voie", "_jour"], how="left")

    return dvf.drop(columns="_jour")


### AJOUT

def ajouter_marche_local_historique(dvf):
    """Ajoute le niveau de prix historique observé dans la même commune."""
    dvf = dvf.copy()
    colonnes = {"code_commune", "date_mutation", "prix_m2"}
    manquantes = colonnes - set(dvf.columns)
    if manquantes:
        raise ValueError(f"Marché local impossible : {sorted(manquantes)}")

    dvf["_jour"] = dvf["date_mutation"].dt.normalize()
    jours = (
        dvf.groupby(["code_commune", "_jour"], dropna=False)["prix_m2"]
        .agg(prix_median="median", nombre_ventes="count")
        .reset_index()
        .sort_values(["code_commune", "_jour"])
    )

    groupe = jours.groupby("code_commune", sort=False)
    jours["local_previous_median_price_m2"] = groupe["prix_median"].transform(lambda serie: serie.shift(1).expanding().median())
    jours["local_previous_transaction_count"] = groupe["nombre_ventes"].transform(lambda serie: serie.shift(1).cumsum()).fillna(0).astype(int)

    dvf = dvf.merge(jours[["code_commune","_jour", "local_previous_median_price_m2", "local_previous_transaction_count",]], on=["code_commune", "_jour"], how="left")

    return dvf.drop(columns="_jour")


### AJOUT

def ajouter_comparables(dvf):
    """Ajoute une estimation basée sur des biens de caractéristiques proches."""
    dvf = dvf.copy()
    colonnes = {
        "code_commune",
        "date_mutation",
        "prix_m2",
        "type_local",
        "surface_reelle_bati",
    }

    manquantes = colonnes - set(dvf.columns)
    
    if manquantes:
        raise ValueError(f"Comparables impossibles : {sorted(manquantes)}")

    # Une classe de surface simple permet de comparer des biens de taille proche.
    dvf["_classe_surface"] = (dvf["surface_reelle_bati"] / 20).round()
    dvf["_jour"] = dvf["date_mutation"].dt.normalize()

    # Regroupement des transcations par commune,type de bien,classe de surface et calcul de mediane m2/jour, nb de vente/jour
    jours = (
        dvf.groupby(["code_commune", "type_local", "_classe_surface", "_jour"], dropna=False,)["prix_m2"]
        .agg(prix_median="median", nombre_ventes="count")
        .reset_index()
        .sort_values(["code_commune", "type_local", "_classe_surface", "_jour"])
    )

    groupe = jours.groupby(["code_commune", "type_local", "_classe_surface"],sort=False)

    jours["comparable_median_price_m2"] = groupe["prix_median"].transform(lambda serie: serie.shift(1).expanding().median())
    jours["comparable_count"] = groupe["nombre_ventes"].transform(lambda serie: serie.shift(1).cumsum()).fillna(0).astype(int)

    dvf = dvf.merge(jours[[
        "code_commune", 
        "type_local", 
        "_classe_surface", 
        "_jour", 
        "comparable_median_price_m2", 
        "comparable_count",
        ]],
        on=["code_commune", "type_local", "_classe_surface", "_jour"],
        how="left"
    )

    return dvf.drop(columns=["_jour", "_classe_surface"])



### AJOUT

def finaliser_experience(df, chemin=None):

    """Supprime les colonnes techniques avant l'entraînement."""

    colonnes_a_supprimer = [
        "nature_culture", "nom_commune", "code_postal", "id_mutation",
        "id_parcelle", "adresse_nom_voie", "adresse_code_voie",
        "code_departement", "adresse_numero", "lot1_numero", "lot2_numero",
        "lot3_numero", "lot4_numero", "lot5_numero",
        "code_nature_culture", "code_nature_culture_speciale",
        "nature_culture_speciale", "numero_volume", "ancien_code_commune",
        "ancien_nom_commune", "ancien_id_parcelle", "previous_sale_date",
    ]

    df_final = df.drop(columns=colonnes_a_supprimer, errors="ignore").copy()

    assert df_final["prix_m2"].notna().all(), "NA dans prix_m2."
    assert df_final["valeur_fonciere"].notna().all(), "NA dans valeur_fonciere."
    assert df_final["surface_reelle_bati"].notna().all(), "NA dans surface_reelle_bati."
    assert (df_final["surface_reelle_bati"] > 0).all(), "surface_reelle_bati <= 0."

    if chemin is not None:
        df_final.to_csv(chemin, index=False)
        print("Export expérience :", chemin)
        print("Dimensions :", df_final.shape)

    return df_final




def finaliser_datasets(dvf_initial, dvf_nettoye):

    """Supprime les identifiants uniquement après construction des historiques."""

    colonnes_a_supprimer = [
        "nature_culture", "nom_commune", "code_postal", "id_mutation",
        "id_parcelle", "adresse_nom_voie", "adresse_code_voie",
        # "date_mutation", 
        "code_departement", "adresse_numero",
        "lot1_numero", "lot2_numero", "lot3_numero", "lot4_numero",
        "lot5_numero", "code_nature_culture", "code_nature_culture_speciale",
        "nature_culture_speciale", "numero_volume", "ancien_code_commune",
        "ancien_nom_commune", "ancien_id_parcelle", "previous_sale_date",
    ]
    datasets = {"dvf_initial_features_enrichies": dvf_initial.copy(),"dvf_nettoye_features_enrichies": dvf_nettoye.copy()}
    resultats = {}

    for nom, df in datasets.items():

        # Suppression des colonnes inutiles
        df = df.drop(columns=colonnes_a_supprimer, errors="ignore")

        # Vérification des lignes identiques
        # nb_lignes_identiques = df.duplicated().sum()

        # Contrôles finaux
        assert df["prix_m2"].notna().all(), f"{nom} : NA dans prix_m2."
        assert df["valeur_fonciere"].notna().all(), f"{nom} : NA dans valeur_fonciere."
        assert df["surface_reelle_bati"].notna().all(), f"{nom} : NA dans surface_reelle_bati."
        assert (df["surface_reelle_bati"] > 0).all(), f"{nom} : surface_reelle_bati <= 0."

        # Export
        chemin = f"../data/processed/{nom}.csv"
        df.to_csv(chemin, index=False)

        # Bilan
        print(f"\n--- {nom} ---")
        print("Dimensions :", df.shape)
        # print("Lignes identiques :", nb_lignes_identiques)
        print("Fichier :", chemin)


        resultats[nom] = df

    return resultats["dvf_initial_features_enrichies"], resultats["dvf_nettoye_features_enrichies"]

