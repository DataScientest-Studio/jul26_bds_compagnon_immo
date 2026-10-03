import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal

##############################
#figures de la source INSEE 
##############################
def evolution_prix_territoire(df_dvf, df_insee, variable_insee, annee_debut=2021, annee_fin=2025):

    dvf = df_dvf.copy()
    insee = df_insee.copy()

    # -----------------------------
    # Création variables INSEE
    # -----------------------------

    # Densité de population
    if "densite_population" not in insee.columns: 
        insee["densite_population"] = insee["P19_POP"] / insee["SUPERF"]

    # Évolution de l'emploi
    if "evolution_emploi" not in insee.columns and "P13_EMPLT" in insee.columns and "P19_EMPLT" in insee.columns: 
        insee["evolution_emploi"] = (insee["P19_EMPLT"] - insee["P13_EMPLT"]) / insee["P13_EMPLT"] * 100

    # Taux de chômage
    if "taux_chomage" not in insee.columns and "P19_CHOMEUR1564" in insee.columns and "P19_ACT1564" in insee.columns: 
        insee["taux_chomage"] = insee["P19_CHOMEUR1564"] / insee["P19_ACT1564"] * 100

    #verifier variable_insee
    if variable_insee not in insee.columns:
        raise ValueError(
            f"La variable '{variable_insee}' n'existe pas dans df_insee."
        )
    # -----------------------------
    # Codes communes
    # -----------------------------

    dvf["code_commune"] = (dvf["code_commune"].astype(str).str.zfill(5))
    insee["INSEE_COM"] = (insee["INSEE_COM"].astype(str).str.zfill(5))

    # -----------------------------
    # Préparation DVF
    # -----------------------------

    dvf["date_mutation"] = pd.to_datetime(dvf["date_mutation"])
    dvf["annee"] = dvf["date_mutation"].dt.year

    dvf = dvf[ (dvf["nature_mutation"] == "Vente") & (dvf["type_local"].isin(["Maison", "Appartement"]))
        & (dvf["surface_reelle_bati"] > 0) & (dvf["valeur_fonciere"] > 0)]
    
    # -----------------------------
    # Prix au m²
    # -----------------------------

    dvf["prix_m2"] = ( dvf["valeur_fonciere"] / dvf["surface_reelle_bati"])

    # -----------------------------
    # Prix médian commune / année
    # -----------------------------

    prix_commune = (dvf.groupby(["code_commune", "annee"], as_index=False)["prix_m2"].median())

    # -----------------------------
    # Evolution prix
    # -----------------------------

    evolution = (prix_commune[prix_commune["annee"].isin([annee_debut, annee_fin])]
                    .pivot(index="code_commune", columns="annee", values="prix_m2").dropna().reset_index())

    evolution["evolution_prix_m2"] = ((evolution[annee_fin] - evolution[annee_debut]) / evolution[annee_debut] * 100)

    # -----------------------------
    # Jointure DVF × INSEE
    # -----------------------------

    df_final = evolution.merge(insee, left_on="code_commune", right_on="INSEE_COM", how="inner")

    # -----------------------------
    # Graphique
    # -----------------------------

    colonnes = [ variable_insee, "evolution_prix_m2"]

    if "P19_POP" in df_final.columns:
        colonnes.append("P19_POP")

    data = df_final[colonnes].dropna()
    plt.figure(figsize=(5, 3))
    if "P19_POP" in data.columns:
        sns.scatterplot( data=data, x=variable_insee, y="evolution_prix_m2", size="P19_POP", alpha=0.7, legend=False)

    else:
        sns.scatterplot(data=data, x=variable_insee, y="evolution_prix_m2", alpha=0.7)

    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")
    plt.xlabel(variable_insee)

    plt.ylabel(f"Évolution du prix médian au m² ({annee_debut}-{annee_fin}) (%)")
    plt.title(f"Évolution des prix immobiliers selon {variable_insee}")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Validation statistique
    # -----------------------------

    corr, pvalue = spearmanr(data[variable_insee], data["evolution_prix_m2"])
    print("Corrélation de Spearman :", round(corr, 3))
    print("p-value :", round(pvalue, 4))

    if pvalue < 0.05:
        print("=> Relation statistiquement significative.")
    else:
        print("=> Relation non statistiquement significative.")

    return df_final

def dynamique_territoire(df, afficher_validation=False):
    # Calcul du taux d'évolution de la population entre 2013 et 2019
    df["evolution_population"] = ((df["P19_POP"] - df["P13_POP"]) / df["P13_POP"] * 100 )
    df["evolution_emploi"] = ((df["P19_EMPLT"] - df["P13_EMPLT"]) / df["P13_EMPLT"]) * 100

    plt.figure(figsize=(5, 3))
    sns.scatterplot(df, x="evolution_population", y="evolution_emploi")

    plt.axhline(0, linestyle="--")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Évolution population (%)")
    plt.ylabel("Évolution emploi (%)")
    plt.title("Dynamique démographique et économique des territoires")
    plt.show()

    if afficher_validation:
        corr, pvalue = spearmanr( df["evolution_population"], df["evolution_emploi"])
        print("Corrélation de Spearman :", round(corr, 3))
        print("p-value :", round(pvalue, 4))

        if pvalue < 0.05:
            print("La relation est statistiquement significative.")
        else:
            print("La relation n'est pas statistiquement significative.")

##################################
#figures de la sources DVF
##################################
def boxplot_prix_m2_type_bien(dvf, afficher_validation=False):

    # Filtrage des données
    data = dvf[(dvf["nature_mutation"] == "Vente") & (dvf["type_local"].isin(["Maison", "Appartement"])) &
                (dvf["surface_reelle_bati"] > 0) & (dvf["valeur_fonciere"] > 0) ]
        
    # Calcul du prix au m²
    data["prix_m2"] = (data["valeur_fonciere"] / data["surface_reelle_bati"])

    # Graphique
    # plt.figure(figsize=(8, 5))
    plt.figure(figsize=(5, 3))
    sns.boxplot(data=data, x="type_local", y="prix_m2", showfliers=False)

    plt.title("Distribution du prix au m² selon le type de bien")
    plt.xlabel("Type de bien")
    plt.ylabel("Prix au m² (€)")
    plt.grid(alpha=0.3)
    plt.show()

    # Validation statistique
    if afficher_validation:

        appartement = data[data["type_local"] == "Appartement"]["prix_m2"]
        maison = data[data["type_local"] == "Maison"]["prix_m2"]
        stat, p_value = mannwhitneyu(appartement, maison)

        print("Médiane appartements :", round(appartement.median(), 2), "€/m²")
        print("Médiane maisons :", round(maison.median(), 2), "€/m²")
        print("p-value :", p_value)

        if p_value < 0.05:
            print("La différence est statistiquement significative.")
        else:
            print("La différence n'est pas statistiquement significative.")


def barh_prix_m2_communes(dvf, afficher_validation=False):

    # Garder les ventes de maisons et appartements avec une surface valide
    data = dvf[(dvf["nature_mutation"] == "Vente") & (dvf["type_local"].isin(["Maison", "Appartement"])) &
            (dvf["surface_reelle_bati"] > 0) & (dvf["valeur_fonciere"] > 0) ]
    
    # Calcul du prix au m²
    data["prix_m2"] = (data["valeur_fonciere"] / data["surface_reelle_bati"])

    # Prix médian au m² par commune
    prix_commune = (data.groupby("nom_commune").agg(prix_median_m2=("prix_m2", "median"), nombre_ventes=("prix_m2", "count")))

    # Communes avec au moins 100 ventes
    prix_commune = prix_commune[prix_commune["nombre_ventes"] >= 100]

    # Top 15 communes
    top_communes = (prix_commune.sort_values("prix_median_m2", ascending=False).head(15).sort_values("prix_median_m2"))

    # Graphique
    # plt.figure(figsize=(10, 7))
    plt.figure(figsize=(5, 3))
    top_communes["prix_median_m2"].plot(kind="barh")
    plt.title("Communes aux prix médians au m² les plus élevés")
    plt.xlabel("Prix médian au m² (€)")
    plt.ylabel("Commune")
    plt.grid(alpha=0.3)
    plt.show()

    # Validation
    if afficher_validation:
        # Garder les communes avec au moins 100 ventes pour obtenir des résultats plus représentatifs
        print(prix_commune.sort_values("prix_median_m2", ascending=False).head(15))

def lineplot_evolution_prix_m2(dvf, afficher_validation=False):

    # Garder les ventes de maisons et appartements avec une surface valide
    data = dvf[(dvf["nature_mutation"] == "Vente") & (dvf["type_local"].isin(["Maison", "Appartement"])) &
            (dvf["surface_reelle_bati"] > 0) & (dvf["valeur_fonciere"] > 0) ]

    # Conversion de la date
    data["date_mutation_conver"] = pd.to_datetime(data["date_mutation"])

    # Calcul du prix au m² et de l'année
    data["prix_m2"] = data["valeur_fonciere"] / data["surface_reelle_bati"]
    data["annee"] = data["date_mutation_conver"].dt.year

    # Prix médian au m² par année
    prix_annee = data.groupby("annee")["prix_m2"].median()

    # Graphique
    # plt.figure(figsize=(8, 5))
    plt.figure(figsize=(5, 3))
    prix_annee.plot(kind= "line", marker="o")
    plt.title("Évolution du prix médian au m²")
    plt.xlabel("Année")
    plt.ylabel("Prix médian au m² (€)")
    plt.grid(alpha=0.3)
    plt.show()

    # Validation
    if afficher_validation:
        print("Prix médian au m² par année :", prix_annee)
        evolution = prix_annee.pct_change() * 100 #taux de variation entre une valeur et la valeur précédente
        print("Évolution annuelle en % :", evolution)

def barplot_prix_m2_par_pieces(dvf, afficher_validation=False):
    data = dvf[(dvf["nature_mutation"] == "Vente") & (dvf["type_local"].isin(["Maison", "Appartement"])) &
        (dvf["nombre_pieces_principales"].between(1, 10)) & (dvf["surface_reelle_bati"] > 0) & (dvf["valeur_fonciere"] > 0)]

    # Calcul du prix au m²
    data["prix_m2"] = (data["valeur_fonciere"] / data["surface_reelle_bati"])

    prix_par_pieces = ( data.groupby("nombre_pieces_principales")["prix_m2"].median())

    # Graphique
    # plt.figure(figsize=(10, 5))
    plt.figure(figsize=(5, 3))
    prix_par_pieces.plot(kind="bar")
    plt.title("prix m2 médiane selon le nombre de pièces")
    plt.xlabel("Nombre de pièces principales")
    plt.ylabel("Prix m2 médiane (€)")
    plt.xticks(rotation=0)
    plt.show()

    # Validation statistique
    if afficher_validation:
        # Manipulation de données
        validation = ( data.groupby("nombre_pieces_principales")["prix_m2"].agg(["count", "median"]).round(2))

        print("Prix au m² par nombre de pièces :")
        print(validation)

        # Test de tendance
        correlation, p_value = spearmanr(data["nombre_pieces_principales"],data["prix_m2"])
        print("\nCorrélation de Spearman :", round(correlation, 3))
        print("p-value :", p_value)
        if p_value < 0.05:
            print("La relation est statistiquement significative.")
        else:
            print("La relation n'est pas statistiquement significative.")

def scatter_surface_val_fonciere(dvf, afficher_validation=False):

    # Garder uniquement les ventes de maisons et appartements
    data = dvf[(dvf["nature_mutation"] == "Vente") & (dvf["type_local"].isin(["Maison", "Appartement"]))
            & (dvf["surface_reelle_bati"] > 0) & (dvf["valeur_fonciere"] > 0)].copy()

    # Calcul du prix au m²
    data["prix_m2"] = (data["valeur_fonciere"] / data["surface_reelle_bati"])

    # Deux graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Surface / valeur foncière
    sns.scatterplot(data=data, x="surface_reelle_bati", y="valeur_fonciere", alpha=0.3, ax=axes[0])
    axes[0].set_title("Surface bâtie et valeur foncière")
    axes[0].set_xlabel("Surface bâtie (m²)")
    axes[0].set_ylabel("Valeur foncière (€)")
    axes[0].grid(alpha=0.3)

    # Surface / prix au m²
    sns.scatterplot(data=data, x="surface_reelle_bati", y="prix_m2", alpha=0.3, ax=axes[1])
    axes[1].set_title("Surface bâtie et prix au m²")
    axes[1].set_xlabel("Surface bâtie (m²)")
    axes[1].set_ylabel("Prix au m² (€)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Validation statistique
    if afficher_validation:
        corr_valeur, p_valeur = spearmanr(data["surface_reelle_bati"], data["valeur_fonciere"])
        corr_prix_m2, p_prix_m2 = spearmanr(data["surface_reelle_bati"], data["prix_m2"])

        print("Surface / valeur foncière :")
        print("Corrélation de Spearman :", round(corr_valeur, 3))
        print("p-value :", round(p_valeur, 4))

        print("\nSurface / prix au m² :")
        print("Corrélation de Spearman :", round(corr_prix_m2, 3))
        print("p-value :", round(p_prix_m2, 4))

############################################
#figures de la sources annonces_69
############################################
# def boxplot_prix(df, colonne="prix_m2_vente", afficher_quantiles=False):
#     plt.figure(figsize=(10, 4))
#     plt.boxplot(df[colonne].dropna(), vert=False)
#     plt.title("Distribution des prix des annonces")
#     plt.xlabel("Prix (€)") 
#     plt.grid(alpha=0.3)
#     plt.show() 
 
#     if afficher_quantiles: 
#         print(df[colonne].quantile([0.25, 0.50, 0.75, 0.95, 0.99]))


# def barplot_top_villes(df, colonne, top=20, afficher_validation=False):
    
#     top_villes = df[colonne].value_counts().head(top)

#     plt.figure(figsize=(10, 7))
#     top_villes.sort_values().plot(kind="barh")

#     plt.title(f"Top {top} des {colonne} les plus représentés")
#     plt.xlabel("Nombre d'annonces")
#     plt.ylabel("Villes")
#     plt.grid(axis="x", alpha=0.3)
#     plt.show()

#     if afficher_validation:
#         print(df[colonne].value_counts())
#         print(f"\nNombre de {colonne} différentes : {df[colonne].nunique()}")


# def prix_median_ville(df, top=20, afficher_validation=False):

#     prix_ville = (df.groupby("INSEE_COM")["prix_m2_vente"].median().sort_values(ascending=False).head(top))

#     plt.figure(figsize=(10, 7))
#     prix_ville.sort_values().plot(kind="barh")
#     plt.title(f"{top} villes ayant le prix médian le plus élevé")
#     plt.xlabel("Prix médian (€)")
#     plt.ylabel("Ville")
#     plt.grid(axis="x", alpha=0.3)
#     plt.show()

#     if afficher_validation:
#         print(prix_ville)

#         stats_ville = (df.groupby("INSEE_COM").agg(prix_median=("prix_m2_vente", "median"),nombre_annonces=("prix_m2_vente", "size"))
#             .sort_values("prix_median", ascending=False)
#         )
#         print(stats_ville[stats_ville["nombre_annonces"] >= 10].head(top))


# def prix_m2(df, afficher_validation=False):

#     df["prix_m2"] = df["prix_m2_vente"] / df["surface"]
#     prix_m2 = df.loc[df["surface"] > 0, "prix_m2_vente"].dropna()

#     plt.figure(figsize=(10, 6))
#     plt.hist(prix_m2, bins=40)
#     plt.title("Distribution du prix au m²")
#     plt.xlabel("Prix au m² (€/m²)")
#     plt.ylabel("Nombre d'annonces")
#     plt.grid(alpha=0.3)
#     plt.show()
 
#     if afficher_validation:
#         print(prix_m2.describe())
#         print(prix_m2.quantile([0.25, 0.50, 0.75, 0.95, 0.99]))

# # def correlation(df, afficher_validation=False):

# #     variables = ["price", "livingArea", "rooms", "bedrooms", "prix_m2"]
# #     correlation = df[variables].corr()
# #     plt.figure(figsize=(8, 6))
# #     sns.heatmap(correlation, annot=True, fmt=".2f", cmap="Blues" )
# #     plt.title("Corrélation entre les principales variables numériques")
# #     plt.show()

# #     if afficher_validation:
# #         print(correlation)
# #         print(correlation.abs())

# def relation_surface_prix(df, afficher_validation=False):

#     plt.figure(figsize=(8, 6))
#     plt.scatter(df["surface"],df["prix_m2_vente"], alpha=0.5)
#     plt.title("Relation entre surface habitable et prix")
#     plt.xlabel("Surface (m²)")
#     plt.ylabel("Prix (€)")
#     plt.grid(alpha=0.3) 
#     plt.show()
     
#     if afficher_validation:        
#         # test statistique qui mesure si deux variables ont tendance à évoluer ensemble (façon monotone)
#         correlation, p_value = spearmanr(df["surface"], df["prix_m2_vente"])
#         print("Corrélation de Spearman :", round(correlation, 3))
#         print("p-value :", p_value)
        

# #figure de la sources INSEE
# # def heatmap_correlation(df, variables):
# #     correlation = df[variables].corr()

# #     plt.figure(figsize=(12, 8))
# #     sns.heatmap(correlation, annot=True, fmt=".2f")

# #     plt.title("Corrélations entre les variables INSEE")
# #     plt.show()