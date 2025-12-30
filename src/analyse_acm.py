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
    """Teste l'indépendance entre les variables qualitatives (Test du Chi2)"""
    print("\n" + "=" * 80)
    print("TESTS D'INDÉPENDANCE (CHI-2)")
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
    
    print("\nINTERPRÉTATION :")
    print("• p-value < 0.001 → Liaison FORTE entre les variables")
    print("• 0.001 ≤ p-value < 0.05 → Liaison MODÉRÉE")
    print("• p-value ≥ 0.05 → Liaison FAIBLE (variables indépendantes)")

def perform_mca(df, cat_vars, n_components=4):
    """Effectue l'ACM (Analyse des Correspondances Multiples)"""
    print("\n" + "=" * 80)
    print("ANALYSE DES CORRESPONDANCES MULTIPLES (ACM)")
    print("=" * 80)
    
    mca = prince.MCA(n_components=n_components, n_iter=10, copy=True, check_input=True, engine='sklearn', random_state=42)
    mca = mca.fit(df[cat_vars])
    
    print(f"\nACM ajustée avec {n_components} composantes")
    print(f"Nombre total de modalités : {len(mca.column_coordinates(df[cat_vars]))}")
    
    return mca

def analyze_eigenvalues(mca, save_dir="results/figures"):
    """Analyse les valeurs propres et l'inertie"""
    print("\n" + "=" * 80)
    print("VALEURS PROPRES ET INERTIE")
    print("=" * 80)
    
    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    cumulative_inertia = np.cumsum(explained_inertia)
    
    summary = pd.DataFrame({
        'Dimension': [f'Dim {i+1}' for i in range(len(eigenvalues))],
        'Valeur propre (λ)': eigenvalues.round(4),
        'Inertie (%)': (explained_inertia * 100).round(2),
        'Inertie cumulée (%)': (cumulative_inertia * 100).round(2)
    })
    
    save_table(summary, "valeurs_propres_inertie.csv", "VALEURS PROPRES ET INERTIE")
    
    flag, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linewidth=2, markersize=10, color='darkblue')
    axes[0].axhline(y=1/len(eigenvalues), color='red', linestyle='--', linewidth=2, label=f'Seuil moyen (1/p = {1/len(eigenvalues):.3f})')
    axes[0].set_xlabel('Dimension', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Valeur propre', fontsize=12, fontweight='bold')
    axes[0].set_title('Graphique d\'éboulis (Scree plot)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)
    
    axes[1].bar(range(1, len(explained_inertia) + 1), explained_inertia * 100, alpha=0.7, color='steelblue', label='Inertie par axe', edgecolor='black')
    axes[1].plot(range(1, len(cumulative_inertia) + 1), cumulative_inertia * 100, marker='o', color='red', linewidth=2, markersize=8, label='Inertie cumulée')
    axes[1].axhline(y=80, color='green', linestyle='--', linewidth=2, label='Seuil 80%')
    axes[1].set_xlabel('Dimension', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Inertie (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Inertie expliquée par dimension', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/scree_plot_inertie.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : scree_plot_inertie.png")
    
    n_axes_80 = np.argmax(cumulative_inertia >= 0.80) + 1 if any(cumulative_inertia >= 0.80) else len(eigenvalues)
    n_axes_criterion = np.sum(eigenvalues > 1/len(eigenvalues))
    
    print(f"\nRECOMMANDATIONS :")
    print(f"   • Nombre d'axes pour 80% d'inertie : {n_axes_80}")
    print(f"   • Nombre d'axes selon critère de Kaiser (λ > 1/p) : {n_axes_criterion}")
    print(f"   • Inertie expliquée par les 2 premiers axes : {cumulative_inertia[1]*100:.2f}%")
    
    print(f"\nINTERPRÉTATION :")
    print(f"   • Les {n_axes_criterion} premières dimensions capturent l'essentiel de l'information")
    print(f"   • Focus sur le plan factoriel 1-2 pour l'interprétation")

def analyze_individuals(mca, df, save_dir="results/figures"):
    """Analyse des individus (lignes)"""
    print("\n" + "=" * 80)
    print("ANALYSE DES INDIVIDUS")
    print("=" * 80)
    
    row_coords = mca.row_coordinates(df)
    row_contrib = mca.row_contributions_
    row_cos2 = mca.row_cosine_similarities(row_coords)
    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    
    print("\n🔝 Top 10 individus contribuant le plus à la Dimension 1 :")
    top_contrib_dim1 = row_contrib.iloc[:, 0].nlargest(10)
    print(top_contrib_dim1.to_string())
    
    print(f"\nINTERPRÉTATION :")
    print(f"   • Contribution moyenne attendue : {100/len(row_coords):.3f}%")
    print(f"   • Les individus avec contribution > moyenne sont influents sur l'axe 1")
    
    flag, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    scatter1 = axes[0].scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1], c=row_cos2.sum(axis=1), cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Individus - Plan factoriel (1-2) - Qualité (cos²)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Cos² total')
    axes[0].grid(alpha=0.3)
    
    scatter2 = axes[1].scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1], c=row_contrib.iloc[:, 0], cmap='Reds', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Individus - Plan factoriel (1-2) - Contribution (Dim1)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Contribution Dim1 (%)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/individus_plan_factoriel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : individus_plan_factoriel.png")
    
    summary_individuals = pd.DataFrame({
        'Individu': range(1, len(row_coords) + 1),
        'Dim1_coord': row_coords.iloc[:, 0].values,
        'Dim2_coord': row_coords.iloc[:, 1].values,
        'Dim1_contrib': row_contrib.iloc[:, 0].values,
        'Dim2_contrib': row_contrib.iloc[:, 1].values,
        'Cos2_total': row_cos2.sum(axis=1).values
    })
    summary_individuals.to_csv("results/tables/individus_parametres.csv", index=False)
    print(f"Tableau sauvegardé : individus_parametres.csv")

def analyze_modalities(mca, df, cat_vars, save_dir="results/figures"):
    """Analyse des modalités (colonnes)"""
    print("\n" + "=" * 80)
    print("ANALYSE DES MODALITÉS")
    print("=" * 80)
    
    col_coords = mca.column_coordinates(df[cat_vars])
    col_contrib = mca.column_contributions_
    col_cos2 = mca.column_cosine_similarities(df[cat_vars])
    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    
    print("\nTop 10 modalités contribuant le plus à la Dimension 1 :")
    top_contrib = col_contrib.iloc[:, 0].nlargest(10)
    print(top_contrib.to_string())
    
    print(f"\nINTERPRÉTATION :")
    print(f"   • Contribution moyenne attendue : {100/len(col_coords):.3f}%")
    print(f"   • Les modalités avec contribution > moyenne définissent l'axe 1")
    
    flag, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    scatter1 = axes[0].scatter(col_coords.iloc[:, 0], col_coords.iloc[:, 1], c=col_cos2.sum(axis=1), cmap='plasma', alpha=0.7, s=150, edgecolors='black', linewidth=1.5)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    for idx, label in enumerate(col_coords.index):
        axes[0].annotate(label, (col_coords.iloc[idx, 0], col_coords.iloc[idx, 1]), fontsize=9, ha='right', alpha=0.9, fontweight='bold')
    
    axes[0].set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Modalités - Plan factoriel (1-2) - Qualité (cos²)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Cos² total')
    axes[0].grid(alpha=0.3)
    
    scatter2 = axes[1].scatter(col_coords.iloc[:, 0], col_coords.iloc[:, 1], c=col_contrib.iloc[:, 0], cmap='Oranges', alpha=0.7, s=150, edgecolors='black', linewidth=1.5)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    for idx, label in enumerate(col_coords.index):
        axes[1].annotate(label, (col_coords.iloc[idx, 0], col_coords.iloc[idx, 1]), fontsize=9, ha='right', alpha=0.9, fontweight='bold')
    
    axes[1].set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Modalités - Plan factoriel (1-2) - Contribution (Dim1)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Contribution Dim1 (%)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/modalites_plan_factoriel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : modalites_plan_factoriel.png")
    
    summary_modalities = pd.DataFrame({
        'Modalite': col_coords.index,
        'Dim1_coord': col_coords.iloc[:, 0].values,
        'Dim2_coord': col_coords.iloc[:, 1].values,
        'Dim1_contrib': col_contrib.iloc[:, 0].values,
        'Dim2_contrib': col_contrib.iloc[:, 1].values,
        'Cos2_total': col_cos2.sum(axis=1).values
    })
    summary_modalities.to_csv("results/tables/modalites_parametres.csv", index=False)
    print(f"Tableau sauvegardé : modalites_parametres.csv")

def analyze_variables(mca, df, cat_vars, save_dir="results/figures"):
    """Analyse des variables qualitatives (rapport de corrélation η²)"""
    print("\n" + "=" * 80)
    print("RAPPORT DE CORRÉLATION η² (Variables)")
    print("=" * 80)
    
    row_coords = mca.row_coordinates(df)
    
    eta2_results = {}
    for var in cat_vars:
        eta2_by_dim = []
        for dim in range(mca.n_components):
            groups = [row_coords.loc[df[var] == val, row_coords.columns[dim]].values for val in df[var].unique()]
            overall_mean = row_coords.iloc[:, dim].mean()
            total_var = np.var(row_coords.iloc[:, dim])
            between_var = sum([len(g) * (np.mean(g) - overall_mean)**2 for g in groups]) / len(df)
            eta2 = between_var / total_var if total_var > 0 else 0
            eta2_by_dim.append(eta2)
        
        eta2_results[var] = eta2_by_dim
    
    eta2_df = pd.DataFrame(eta2_results, index=[f'Dim {i+1}' for i in range(mca.n_components)])
    eta2_df = eta2_df.T
    
    save_table(eta2_df.round(4), "eta_squared_variables.csv", "RAPPORT DE CORRÉLATION η² PAR VARIABLE")
    
    print(f"\nINTERPRÉTATION :")
    print(f"   • η² proche de 1 : variable fortement associée à la dimension")
    print(f"   • η² proche de 0 : variable faiblement associée à la dimension")
    print(f"   • Seuil : η² > 0.5 indique une association forte")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    eta2_df.plot(kind='bar', ax=ax, colormap='Set2', width=0.8, edgecolor='black')
    ax.set_xlabel('Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('η²', fontsize=12, fontweight='bold')
    ax.set_title('Rapport de corrélation η² par variable et dimension', fontsize=14, fontweight='bold')
    ax.legend(title='Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xticklabels(eta2_df.index, rotation=45, ha='right')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Seuil 0.5')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/eta_squared_variables.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : eta_squared_variables.png")

def add_supplementary_quantitative(mca, df, cat_vars, quant_vars, save_dir="results/figures"):
    """Ajoute les variables quantitatives supplémentaires"""
    print("\n" + "=" * 80)
    print(f"PROJECTION DES VARIABLES QUANTITATIVES SUPPLÉMENTAIRES")
    print("=" * 80)
    
    row_coords = mca.row_coordinates(df[cat_vars])
    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    
    flg, axes = plt.subplots(1, len(quant_vars), figsize=(8*len(quant_vars), 7))
    
    if len(quant_vars) == 1:
        axes = [axes]
    
    all_correlations = {}
    
    for idx, quant_var in enumerate(quant_vars):
        correlations = []
        for dim in range(mca.n_components):
            valid_idx = df[quant_var].notna()
            corr = np.corrcoef(df.loc[valid_idx, quant_var], row_coords.loc[valid_idx, row_coords.columns[dim]])[0, 1]
            correlations.append(corr)
        
        all_correlations[quant_var] = correlations
        
        scatter = axes[idx].scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1], c=df[quant_var], cmap='coolwarm', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        scale = 3
        axes[idx].arrow(0, 0, correlations[0]*scale, correlations[1]*scale, head_width=0.2, head_length=0.2, fc='darkgreen', ec='darkgreen', linewidth=3, label=f'{quant_var} (r={correlations[0]:.2f}, {correlations[1]:.2f})')
        
        axes[idx].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[idx].axvline(0, color='gray', linestyle='--', linewidth=0.8)
        axes[idx].set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'Biplot : Individus colorés par {quant_var}', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=axes[idx], label=quant_var)
        axes[idx].legend(loc='upper right')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/biplot_individus_variables_quant.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : biplot_individus_variables_quant.png")
    
    corr_df = pd.DataFrame(all_correlations, index=[f'Dim {i+1}' for i in range(mca.n_components)])
    corr_df = corr_df.T
    for col in corr_df.columns:
        corr_df[f'R²_{col}'] = corr_df[col]**2
    
    save_table(corr_df.round(4), "correlation_variables_quant_dimensions.csv", "CORRÉLATION ENTRE VARIABLES QUANTITATIVES ET DIMENSIONS")
    
    print(f"\nINTERPRÉTATION :")
    print(f"   • Corrélation positive : variable augmente dans la direction de l'axe")
    print(f"   • Corrélation négative : variable diminue dans la direction de l'axe")
    print(f"   • |r| > 0.5 : corrélation forte avec la dimension")
    
    create_quant_boxplots(df, cat_vars, quant_vars, save_dir)

def create_quant_boxplots(df, cat_vars, quant_vars, save_dir="results/figures"):
    """Crée des boxplots des variables quantitatives par modalité de chaque variable qualitative"""
    for quant_var in quant_vars:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, cat_var in enumerate(cat_vars):
            df.boxplot(column=quant_var, by=cat_var, ax=axes[idx], patch_artist=True, grid=False)
            axes[idx].set_xlabel(cat_var, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(quant_var, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{quant_var} par {cat_var}', fontsize=13, fontweight='bold')
            axes[idx].get_figure().suptitle('')
            
        plt.tight_layout()
        plt.savefig(f"{save_dir}/boxplots_{quant_var}_par_modalite.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphique sauvegardé : boxplots_{quant_var}_par_modalite.png")

def create_biplot_combined(mca, df, cat_vars, save_dir="results/figures"):
    """Crée un biplot combiné (individus + modalités)"""
    print("\n" + "=" * 80)
    print("CRÉATION DU BIPLOT COMBINÉ")
    print("=" * 80)
    
    row_coords = mca.row_coordinates(df[cat_vars])
    col_coords = mca.column_coordinates(df[cat_vars])
    eigenvalues = mca.eigenvalues_
    explained_inertia = eigenvalues / np.sum(eigenvalues)
    
    flg, ax = plt.subplots(figsize=(14, 10))
    
    ax.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1], c='lightgray', alpha=0.3, s=20, label='Individus')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(col_coords)))
    for idx, (label, row) in enumerate(col_coords.iterrows()):
        ax.scatter(row[0], row[1], c=[colors[idx]], s=200, edgecolors='black', linewidth=2, alpha=0.8)
        ax.annotate(label, (row[0], row[1]), fontsize=10, fontweight='bold', ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx], alpha=0.7))
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(f'Dimension 1 ({explained_inertia[0]*100:.1f}%)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Dimension 2 ({explained_inertia[1]*100:.1f}%)', fontsize=13, fontweight='bold')
    ax.set_title('Biplot : Individus et Modalités', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/biplot_combine.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : biplot_combine.png")

    print(f"\nINTERPRÉTATION DU BIPLOT :")
    print(f"   • Les individus proches partagent des profils similaires")
    print(f"   • Les modalités proches sont choisies par les mêmes individus")
    print(f"   • Distance modalité-origine : représentativité de la modalité")

def main():
    """Fonction principale pour exécuter l'analyse ACM complète"""
    print("\n" + "=" * 80)
    print("DÉMARRAGE DE L'ANALYSE ACM - DATASET LOSSAVERSION")
    print("=" * 80)
    
    create_output_dirs()
    
    df = load_and_explore_data("data/LossAversion.csv")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    cat_vars = ['gender', 'treatment', 'grade', 'arrangement']
    quant_vars = ['invest', 'age']
    
    print("\n" + "=" * 80)
    print("VISUALISATIONS EXPLORATOIRES")
    print("=" * 80)
    plot_categorical_distributions(df, cat_vars)
    plot_quantitative_distributions(df, quant_vars)
    
    test_independence(df, cat_vars)
    
    mca = perform_mca(df, cat_vars, n_components=4)
    
    analyze_eigenvalues(mca)
    
    analyze_individuals(mca, df[cat_vars])
    
    analyze_modalities(mca, df, cat_vars)
    
    analyze_variables(mca, df, cat_vars)
    
    add_supplementary_quantitative(mca, df, cat_vars, quant_vars)
    
    create_biplot_combined(mca, df, cat_vars)
    
    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE AVEC SUCCÈS")
    print("=" * 80)
    print("\nRésultats disponibles dans :")
    print("   • results/figures/ (11 graphiques)")
    print("   • results/tables/ (6 tableaux CSV)")
    print("\nGraphiques générés :")
    print("   1. distributions_variables_qualitatives.png")
    print("   2. distributions_variables_quantitatives.png")
    print("   3. scree_plot_inertie.png")
    print("   4. individus_plan_factoriel.png")
    print("   5. modalites_plan_factoriel.png")
    print("   6. eta_squared_variables.png")
    print("   7. biplot_individus_variables_quant.png")
    print("   8-9. boxplots_[variable]_par_modalite.png")
    print("   10. biplot_combine.png")

if __name__ == "__main__":
    main()