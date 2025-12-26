import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_output_dirs():
    """Crée les répertoires de sortie s'ils n'existent pas"""
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)

def load_and_explore_data(filepath):
    """
    Charge et explore le dataset
    
    Parameters:
    -----------
    filepath : str
        Chemin vers le fichier CSV
        
    Returns:
    --------
    df : pd.DataFrame
        Dataset chargé
    """
    df = pd.read_csv(filepath)
    
    print("=" * 80)
    print("📊 EXPLORATION DU DATASET")
    print("=" * 80)
    print(f"\n🔢 Dimensions : {df.shape[0]} individus × {df.shape[1]} variables\n")
    
    print("📋 Aperçu des données :")
    print(df.head(10))
    
    print("\n📊 Types de variables :")
    print(df.dtypes)
    
    print("\n📈 Statistiques descriptives :")
    print(df.describe(include='all'))
    
    print("\n❓ Valeurs manquantes :")
    print(df.isnull().sum())
    
    return df

def plot_categorical_distributions(df, cat_vars, save_dir="results/figures"):
    """
    Visualise la distribution des variables catégorielles
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    cat_vars : list
        Liste des variables catégorielles
    save_dir : str
        Répertoire de sauvegarde
    """
    n_vars = len(cat_vars)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, var in enumerate(cat_vars):
        counts = df[var].value_counts()
        axes[idx].bar(range(len(counts)), counts.values, color='steelblue', alpha=0.7)
        axes[idx].set_xlabel(var, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Fréquence', fontsize=12)
        axes[idx].set_title(f'Distribution de {var}', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(range(len(counts)))
        axes[idx].set_xticklabels(counts.index, rotation=45, ha='right')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Ajouter les effectifs
        for i, v in enumerate(counts.values):
            axes[idx].text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/distributions_variables_qualitatives.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique sauvegardé : distributions_variables_qualitatives.png")

def plot_height_distribution(df, save_dir="results/figures"):
    """
    Visualise la distribution de la variable height
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    save_dir : str
        Répertoire de sauvegarde
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogramme
    axes[0].hist(df['height'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Hauteur (height)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Fréquence', fontsize=12)
    axes[0].set_title('Distribution de la hauteur', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Boxplot
    axes[1].boxplot(df['height'], vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Hauteur (height)', fontsize=12, fontweight='bold')
    axes[1].set_title('Boxplot de la hauteur', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/distribution_height.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique sauvegardé : distribution_height.png")

def save_table(df, filename, title):
    """
    Sauvegarde un DataFrame en CSV avec affichage
    
    Parameters:
    -----------
    df : pd.DataFrame
        Tableau à sauvegarder
    filename : str
        Nom du fichier
    title : str
        Titre à afficher
    """
    filepath = f"results/tables/{filename}"
    df.to_csv(filepath)
    print(f"\n{'=' * 80}")
    print(f"📊 {title}")
    print(f"{'=' * 80}")
    print(df.to_string())
    print(f"\n✅ Tableau sauvegardé : {filename}")