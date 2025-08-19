"""
Population Distribution Visualization
Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup directories
os.makedirs('../data', exist_ok=True)
os.makedirs('../output', exist_ok=True)

def load_population_data():
    """Load population dataset"""
    url = "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%201/StudentsPerformance.csv"
    df = pd.read_csv(url)
    
    # Add age column (since original dataset doesn't have age, we'll simulate it)
    import numpy as np
    np.random.seed(42)
    df['age'] = np.random.randint(15, 22, size=len(df))
    
    return df

def plot_population_distribution(df, column, plot_type, **kwargs):
    """Create population distribution plot"""
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'bar':
        # Categorical distribution
        ax = sns.countplot(data=df, x=column, order=df[column].value_counts().index,
                          color='#3498db')
        
        # Add counts on bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}', 
                       (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha='center', va='center', 
                       xytext=(0, 5), 
                       textcoords='offset points')
        
        plt.ylabel('Population Count', labelpad=10)
        
    elif plot_type == 'hist':
        # Numerical distribution
        sns.histplot(data=df, x=column, bins=kwargs.get('bins', 15),
                    color='#e74c3c', kde=True)
        
        plt.ylabel('Population Density', labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.title(f'Population Distribution by {column.title()}', pad=20, fontsize=14)
    plt.xlabel(column.title(), labelpad=10)
    
    # Save plot
    filename = f"{column.lower().replace(' ', '_')}_distribution.png"
    plt.savefig(f'../output/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    
    return filename

def main():
    # Set visual style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    
    # Load and prepare data
    population_df = load_population_data()
    
    # Generate visualizations
    print("Creating population distribution visualizations...")
    
    # 1. Gender Distribution (Categorical)
    plot_population_distribution(population_df, 'gender', 'bar')
    
    # 2. Age Distribution (Numerical)
    plot_population_distribution(population_df, 'age', 'hist', bins=15)
    
    # 3. Race Distribution (Categorical)
    plot_population_distribution(population_df, 'race/ethnicity', 'bar')
    
    print("All visualizations saved to /output folder")

if __name__ == "__main__":
    main()
