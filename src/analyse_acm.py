import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prince
from scipy.stats import chi2_contingency
from utils import *

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def test_independence(df, cat_vars):
    """
    Teste l'indépendance entre les variables qualitatives (Test du Chi2)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    cat_vars : list
        Liste des variables catégorielles
    """
    print("\n" + "=" * 80)
    print("🔬 TESTS D'INDÉPENDANCE (CHI-2)")
    print("=" * 80)
    
    results = []
    for i in range(len(cat_vars)):
        for j in range(i+1, len(cat_vars)):
            var1, var2 = cat_vars[i], cat_vars[j]
            contingency_table = pd.crosstab(df[var1], df[var2])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            results.append({
                'Variable 1': var1,
                'Variable 2': var2,
                'Chi2': round(chi2, 2),
                'p-value': f"{p_value:.4e}",
                'ddl': dof,
                'Liaison': 'Forte' if p_value < 0.001 else ('Modérée' if p_value < 0.05 else 'Faible')
            })
    
    results_df = pd.DataFrame(results)
    save_table(results_df, "tests_independence_chi2.csv", "RÉSULTATS DES TESTS D'INDÉPENDANCE")

def perform_mca(df, cat_vars, n_components=5):
    """
    Effectue l'ACM (Analyse des Correspondances Multiples)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    cat_vars : list
        Variables qualitatives à analyser
    n_components : int
        Nombre de composantes à extraire
        
    Returns:
    --------
    mca : prince.MCA
        Modèle ACM ajusté
    """
    print("\n" + "=" * 80)
    print("🎯 ANALYSE DES CORRESPONDANCES MULTIPLES (ACM)")
    print("=" * 80)
    
    # Initialisation du modèle ACM
    mca = prince.MCA(
        n_components=n_components,
        n_iter=10,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )
    
    # Ajustement du modèle
    mca = mca.fit(df[cat_vars])
    
    print(f"\n✅ ACM ajustée avec {n_components} composantes")
    
    return mca

def analyze_eigenvalues(mca, save_dir="results/figures"):
    """
    Analyse les valeurs propres et l'inertie
    
    Parameters:
    -----------
    mca : prince.MCA
        Modèle ACM ajusté
    save_dir : str
        Répertoire de sauvegarde
    """
    print("\n" + "=" * 80)
    print("📊 VALEURS PROPRES ET INERTIE")
    print("=" * 80)
    
    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    cumulative_inertia = np.cumsum(explained_inertia)
    
    # Création du tableau récapitulatif
    summary = pd.DataFrame({
        'Dimension': [f'Dim {i+1}' for i in range(len(eigenvalues))],
        'Valeur propre (λ)': eigenvalues.round(4),
        'Inertie (%)': (explained_inertia * 100).round(2),
        'Inertie cumulée (%)': (cumulative_inertia * 100).round(2)
    })
    
    save_table(summary, "valeurs_propres_inertie.csv", "VALEURS PROPRES ET INERTIE")
    
    # Graphique des valeurs propres (Scree plot)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scree plot
    axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 
                 marker='o', linewidth=2, markersize=8, color='darkblue')
    axes[0].axhline(y=1/len(eigenvalues), color='red', linestyle='--', 
                    label=f'Seuil moyen (1/p = {1/len(eigenvalues):.3f})')
    axes[0].set_xlabel('Dimension', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Valeur propre', fontsize=12, fontweight='bold')
    axes[0].set_title('Graphique d\'éboulis (Scree plot)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Inertie cumulée
    axes[1].bar(range(1, len(explained_inertia) + 1), explained_inertia * 100, 
                alpha=0.7, color='steelblue', label='Inertie par axe')
    axes[1].plot(range(1, len(cumulative_inertia) + 1), cumulative_inertia * 100, 
                 marker='o', color='red', linewidth=2, label='Inertie cumulée')
    axes[1].axhline(y=80, color='green', linestyle='--', label='Seuil 80%')
    axes[1].set_xlabel('Dimension', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Inertie (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Inertie expliquée par dimension', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/scree_plot_inertie.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique sauvegardé : scree_plot_inertie.png")
    
    # Nombre d'axes à retenir
    n_axes_80 = np.argmax(cumulative_inertia >= 0.80) + 1
    n_axes_criterion = np.sum(eigenvalues > 1/len(eigenvalues))
    
    print(f"\n🎯 Recommandations :")
    print(f"   • Nombre d'axes pour 80% d'inertie : {n_axes_80}")
    print(f"   • Nombre d'axes selon critère de Kaiser (λ > 1/p) : {n_axes_criterion}")

def analyze_individuals(mca, df, save_dir="results/figures"):
    """
    Analyse des individus (lignes)
    
    Parameters:
    -----------
    mca : prince.MCA
        Modèle ACM ajusté
    df : pd.DataFrame
        Dataset original
    save_dir : str
        Répertoire de sauvegarde
    """
    print("\n" + "=" * 80)
    print("👥 ANALYSE DES INDIVIDUS")
    print("=" * 80)
    
    # Coordonnées des individus
    row_coords = mca.row_coordinates(df)
    
    # Contributions des individus
    row_contrib = mca.row_contributions_
    
    # Cos2 des individus
    row_cos2 = mca.row_cosine_similarities(row_coords)

    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    
    # Affichage des 10 individus les plus contributifs sur Dim1
    print("\n🔝 Top 10 individus contribuant le plus à la Dimension 1 :")
    top_contrib_dim1 = row_contrib.iloc[:, 0].nlargest(10)
    print(top_contrib_dim1.to_string())
    
    # Graphique : Individus sur le plan factoriel (1-2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plan 1-2 coloré par cos2
    scatter1 = axes[0].scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1],
                               c=row_cos2.sum(axis=1), cmap='viridis', 
                               alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', 
                       fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', 
                       fontsize=12, fontweight='bold')
    axes[0].set_title('Individus - Plan factoriel (1-2) - Qualité (cos²)', 
                      fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Cos² total')
    axes[0].grid(alpha=0.3)
    
    # Plan 1-2 coloré par contribution
    scatter2 = axes[1].scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1],
                               c=row_contrib.iloc[:, 0], cmap='Reds', 
                               alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', 
                       fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', 
                       fontsize=12, fontweight='bold')
    axes[1].set_title('Individus - Plan factoriel (1-2) - Contribution (Dim1)', 
                      fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Contribution Dim1 (%)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/individus_plan_factoriel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique sauvegardé : individus_plan_factoriel.png")
    
    # Sauvegarder les coordonnées, contributions et cos2
    summary_individuals = pd.DataFrame({
        'Individu': range(1, len(row_coords) + 1),
        'Dim1_coord': row_coords.iloc[:, 0].values,
        'Dim2_coord': row_coords.iloc[:, 1].values,
        'Dim1_contrib': row_contrib.iloc[:, 0].values,
        'Dim2_contrib': row_contrib.iloc[:, 1].values,
        'Cos2_total': row_cos2.sum(axis=1).values
    })
    summary_individuals.to_csv("results/tables/individus_parametres.csv", index=False)
    print(f"✅ Tableau sauvegardé : individus_parametres.csv")

def analyze_modalities(mca, df, cat_vars, save_dir="results/figures"):
    """
    Analyse des modalités (colonnes)
    
    Parameters:
    -----------
    mca : prince.MCA
        Modèle ACM ajusté
    df : pd.DataFrame
        Dataset
    cat_vars : list
        Variables qualitatives
    save_dir : str
        Répertoire de sauvegarde
    """
    print("\n" + "=" * 80)
    print("🏷️  ANALYSE DES MODALITÉS")
    print("=" * 80)
    
    # Coordonnées des modalités
    col_coords = mca.column_coordinates(df[cat_vars])
    
    # Contributions des modalités
    col_contrib = mca.column_contributions_
    
    # Cos2 des modalités
    col_cos2 = mca.column_cosine_similarities(df[cat_vars])

    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    
    # Top modalités contribuant à Dim1
    print("\n🔝 Top 10 modalités contribuant le plus à la Dimension 1 :")
    top_contrib = col_contrib.iloc[:, 0].nlargest(10)
    print(top_contrib.to_string())
    
    # Graphique : Modalités sur le plan factoriel
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plan 1-2 avec cos2
    scatter1 = axes[0].scatter(col_coords.iloc[:, 0], col_coords.iloc[:, 1],
                               c=col_cos2.sum(axis=1), cmap='plasma', 
                               alpha=0.7, s=100, edgecolors='black', linewidth=1)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    # Annoter les modalités
    for idx, label in enumerate(col_coords.index):
        axes[0].annotate(label, 
                        (col_coords.iloc[idx, 0], col_coords.iloc[idx, 1]),
                        fontsize=8, ha='right', alpha=0.8)
    
    axes[0].set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', 
                       fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', 
                       fontsize=12, fontweight='bold')
    axes[0].set_title('Modalités - Plan factoriel (1-2) - Qualité (cos²)', 
                      fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Cos² total')
    axes[0].grid(alpha=0.3)
    
    # Plan 1-2 avec contributions
    scatter2 = axes[1].scatter(col_coords.iloc[:, 0], col_coords.iloc[:, 1],
                               c=col_contrib.iloc[:, 0], cmap='Oranges', 
                               alpha=0.7, s=100, edgecolors='black', linewidth=1)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    for idx, label in enumerate(col_coords.index):
        axes[1].annotate(label, 
                        (col_coords.iloc[idx, 0], col_coords.iloc[idx, 1]),
                        fontsize=8, ha='right', alpha=0.8)
    
    axes[1].set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', 
                       fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', 
                       fontsize=12, fontweight='bold')
    axes[1].set_title('Modalités - Plan factoriel (1-2) - Contribution (Dim1)', 
                      fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Contribution Dim1 (%)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/modalites_plan_factoriel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique sauvegardé : modalites_plan_factoriel.png")
    
    # Sauvegarder
    summary_modalities = pd.DataFrame({
        'Modalite': col_coords.index,
        'Dim1_coord': col_coords.iloc[:, 0].values,
        'Dim2_coord': col_coords.iloc[:, 1].values,
        'Dim1_contrib': col_contrib.iloc[:, 0].values,
        'Dim2_contrib': col_contrib.iloc[:, 1].values,
        'Cos2_total': col_cos2.sum(axis=1).values
    })
    summary_modalities.to_csv("results/tables/modalites_parametres.csv", index=False)
    print(f"✅ Tableau sauvegardé : modalites_parametres.csv")

def analyze_variables(mca, df, cat_vars, save_dir="results/figures"):
    """
    Analyse des variables qualitatives (rapport de corrélation η²)
    
    Parameters:
    -----------
    mca : prince.MCA
        Modèle ACM
    df : pd.DataFrame
        Dataset
    cat_vars : list
        Variables qualitatives
    save_dir : str
        Répertoire de sauvegarde
    """
    print("\n" + "=" * 80)
    print("📊 RAPPORT DE CORRÉLATION η² (Variables)")
    print("=" * 80)
    
    # Calcul du η² pour chaque variable sur chaque dimension
    row_coords = mca.row_coordinates(df)
    
    eta2_results = {}
    for var in cat_vars:
        eta2_by_dim = []
        for dim in range(mca.n_components):
            # Calcul de la variance inter-classes / variance totale
            groups = [row_coords.loc[df[var] == val, row_coords.columns[dim]].values
                for val in df[var].unique()]


            overall_mean = row_coords.iloc[:, dim].mean()
            
            # Variance totale
            total_var = np.var(row_coords.iloc[:, dim])
            
            # Variance inter-groupes
            between_var = sum([len(g) * (np.mean(g) - overall_mean)**2 for g in groups]) / len(df)
            
            # η²
            eta2 = between_var / total_var if total_var > 0 else 0
            eta2_by_dim.append(eta2)
        
        eta2_results[var] = eta2_by_dim
    
    # Création du tableau
    eta2_df = pd.DataFrame(eta2_results, 
                           index=[f'Dim {i+1}' for i in range(mca.n_components)])
    eta2_df = eta2_df.T
    
    save_table(eta2_df.round(4), "eta_squared_variables.csv", 
               "RAPPORT DE CORRÉLATION η² PAR VARIABLE")
    
    # Graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    eta2_df.plot(kind='bar', ax=ax, colormap='Set2', width=0.8, edgecolor='black')
    ax.set_xlabel('Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('η²', fontsize=12, fontweight='bold')
    ax.set_title('Rapport de corrélation η² par variable et dimension', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(eta2_df.index, rotation=45, ha='right')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Seuil 0.5')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eta_squared_variables.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique sauvegardé : eta_squared_variables.png")

def add_supplementary_quantitative(mca, df, cat_vars, quant_var='height', 
                                    save_dir="results/figures"):
    """
    Ajoute la variable quantitative supplémentaire
    
    Parameters:
    -----------
    mca : prince.MCA
        Modèle ACM
    df : pd.DataFrame
        Dataset
    cat_vars : list
        Variables qualitatives actives
    quant_var : str
        Nom de la variable quantitative
    save_dir : str
        Répertoire de sauvegarde
    """
    print("\n" + "=" * 80)
    print(f"📏 PROJECTION DE LA VARIABLE QUANTITATIVE SUPPLÉMENTAIRE : {quant_var}")
    print("=" * 80)
    
    # Coordonnées des individus
    row_coords = mca.row_coordinates(df[cat_vars])

    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    
    # Calcul de la corrélation entre height et les dimensions
    correlations = []
    for dim in range(mca.n_components):
        corr = np.corrcoef(df[quant_var], row_coords.iloc[:, dim])[0, 1]
        correlations.append(corr)
    
    corr_df = pd.DataFrame({
        'Dimension': [f'Dim {i+1}' for i in range(mca.n_components)],
        'Corrélation': correlations,
        'R²': [c**2 for c in correlations]
    })
    
    save_table(corr_df.round(4), "correlation_height_dimensions.csv",
               f"CORRÉLATION ENTRE {quant_var.upper()} ET LES DIMENSIONS")
    
    # Graphique : Biplot individus + height
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1],
                        c=df[quant_var], cmap='coolwarm', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Ajouter une flèche pour représenter la direction de height
    scale = 3
    ax.arrow(0, 0, correlations[0]*scale, correlations[1]*scale,
            head_width=0.2, head_length=0.2, fc='darkgreen', ec='darkgreen',
            linewidth=3, label=f'{quant_var} (corrélation)')
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Biplot : Individus colorés par {quant_var}', 
                 fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label=quant_var)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/biplot_individus_height.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique sauvegardé : biplot_individus_height.png")
    
    # Graphique supplémentaire : Boxplot height par modalité
    create_height_boxplots(df, cat_vars, quant_var, save_dir)

def create_height_boxplots(df, cat_vars, quant_var, save_dir="results/figures"):
    """
    Crée des boxplots de height par modalité de chaque variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    cat_vars : list
        Variables qualitatives
    quant_var : str
        Variable quantitative
    save_dir : str
        Répertoire de sauvegarde
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, var in enumerate(cat_vars):
        df.boxplot(column=quant_var, by=var, ax=axes[idx], 
                   patch_artist=True, grid=False)
        axes[idx].set_xlabel(var, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(quant_var, fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{quant_var} par {var}', fontsize=13, fontweight='bold')
        axes[idx].get_figure().suptitle('')  # Supprime le titre automatique
        
    plt.tight_layout()
    plt.savefig(f"{save_dir}/boxplots_height_par_modalite.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique sauvegardé : boxplots_height_par_modalite.png")

def create_biplot_combined(mca, df, cat_vars, save_dir="results/figures"):
    """
    Crée un biplot combiné (individus + modalités)
    
    Parameters:
    -----------
    mca : prince.MCA
        Modèle ACM
    df : pd.DataFrame
        Dataset
    cat_vars : list
        Variables qualitatives
    save_dir : str
        Répertoire de sauvegarde
    """
    print("\n" + "=" * 80)
    print("🎨 CRÉATION DU BIPLOT COMBINÉ")
    print("=" * 80)
    
    row_coords = mca.row_coordinates(df[cat_vars])
    col_coords = mca.column_coordinates(df[cat_vars])
    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Individus en gris (arrière-plan)
    ax.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1],
              c='lightgray', alpha=0.3, s=20, label='Individus')
    
    # Modalités en couleur
    colors = plt.cm.Set3(np.linspace(0, 1, len(col_coords)))
    for idx, (label, row) in enumerate(col_coords.iterrows()):
        ax.scatter(row[0], row[1], c=[colors[idx]], s=200, 
                  edgecolors='black', linewidth=2, alpha=0.8)
        ax.annotate(label, (row[0], row[1]), 
                   fontsize=10, fontweight='bold', ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx], alpha=0.7))
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', 
                  fontsize=13, fontweight='bold')
    ax.set_title('Biplot : Individus et Modalités', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/biplot_combine.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique sauvegardé : biplot_combine.png")

def main():
    """
    Fonction principale pour exécuter l'analyse ACM complète
    """
    print("\n" + "=" * 80)
    print("🚀 DÉMARRAGE DE L'ANALYSE ACM - DATASET BAMBOO")
    print("=" * 80)
    
    # 1. Création des répertoires
    create_output_dirs()
    
    # 2. Chargement et exploration
    df = load_and_explore_data("data/jayaraman.bamboo.csv")
    
    # Suppression de la colonne 'rownames' si elle existe
    if 'rownames' in df.columns:
        df = df.drop('rownames', axis=1)
    
    # 3. Variables
    cat_vars = ['loc', 'block', 'tree', 'family']
    quant_var = 'height'
    
    # 4. Visualisations exploratoires
    print("\n" + "=" * 80)
    print("📊 VISUALISATIONS EXPLORATOIRES")
    print("=" * 80)
    plot_categorical_distributions(df, cat_vars)
    plot_height_distribution(df)
    
    # 5. Tests d'indépendance
    test_independence(df, cat_vars)
    
    # 6. ACM
    mca = perform_mca(df, cat_vars, n_components=5)
    
    # 7. Analyse des valeurs propres
    analyze_eigenvalues(mca)
    
    # 8. Analyse des individus
    analyze_individuals(mca, df[cat_vars])
    
    # 9. Analyse des modalités
    analyze_modalities(mca, df, cat_vars)
    
    # 10. Analyse des variables (η²)
    analyze_variables(mca, df, cat_vars)

    # 11. Variable quantitative supplémentaire
    add_supplementary_quantitative(mca, df, cat_vars, quant_var)

    print("\n" + "=" * 80)
    print("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
    print("=" * 80)
    print("\n📁 Résultats disponibles dans :")
    print("   • results/figures/ (graphiques)") 
    print("   • results/tables/ (tableaux CSV)")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()