# src/association_rules.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
from IPython.display import display
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CLUSTER_NAMES_MAP = {
    0: "The Plant-Based Household",
    1: "The Family Budget Optimizer",
    2: "The Promotion-Driven Pet Parent",
    3: "The Demanding Premium Tech Consumer",
    4: "The Wellness Customer",
    5: "The Affluent Health-Conscious Buyer"
}

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_and_prepare_data(basket_path='customer_basket (1).csv', clusters_path='dataset_clusters.csv'):
    """Load and merge basket and cluster data"""
    print("Loading data files...")
    customer_basket = pd.read_csv(basket_path, index_col='invoice_id')
    customer_clusters = pd.read_csv(clusters_path)
    
    if customer_clusters.index.name == 'customer_id':
        customer_clusters = customer_clusters.reset_index()
    elif 'customer_id' not in customer_clusters.columns and customer_clusters.columns[0] == 'Unnamed: 0':
        customer_clusters = customer_clusters.rename(columns={'Unnamed: 0': 'customer_id'})
    elif 'customer_id' not in customer_clusters.columns:
        customer_clusters = customer_clusters.reset_index().rename(columns={'index': 'customer_id'})

    for col in ['cluster', 'Cluster']:
        if col in customer_clusters.columns:
            customer_clusters[col] = customer_clusters[col].map(CLUSTER_NAMES_MAP)
            customer_clusters = customer_clusters.rename(columns={col: 'cluster'})
    
    merged_df = customer_clusters.merge(customer_basket, how='inner', on='customer_id')
    return merged_df


def convert_string_to_list(string):
    """Convert string representation of list to actual list"""
    return json.loads(string.replace("'", '"'))

# ============================================================================
# ASSOCIATION RULES ANALYSIS
# ============================================================================

def run_association_rules(basket_data, min_support=0.05, min_confidence=0.2, split_train=True):
    """
    Run apriori and association rules algorithms
    
    Parameters:
    -----------
    basket_data : list
        List of transactions (each transaction is a list of products)
    min_support : float
        Minimum support threshold
    min_confidence : float
        Minimum confidence threshold
    split_train : bool
        Whether to use 80% train/20% test split
    
    Returns:
    --------
    frequent_itemsets : DataFrame
        Frequent itemsets
    rules : DataFrame
        Association rules sorted by lift
    """
    if split_train:
        train_size = int(len(basket_data) * 0.8)
        data_to_model = basket_data[:train_size]
    else:
        data_to_model = basket_data
        
    te = TransactionEncoder()
    te_fit = te.fit(data_to_model).transform(data_to_model)
    transactions_items = pd.DataFrame(te_fit, columns=te.columns_)
    
    frequent_itemsets = apriori(transactions_items, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
    return frequent_itemsets, rules

# ============================================================================
# DATA CLEANING & FORMATTING
# ============================================================================

def clean_rules_df(df_rules):
    """Format rules DataFrame for display"""
    if df_rules.empty:
        return df_rules
    df_cool = df_rules.copy()
    df_cool['antecedents'] = df_cool['antecedents'].apply(lambda x: f"[{', '.join(list(x))}]")
    df_cool['consequents'] = df_cool['consequents'].apply(lambda x: f"[{', '.join(list(x))}]")
    metrics = ['support', 'confidence', 'lift', 'leverage', 'conviction', 'zhangs_metric']
    for m in metrics:
        if m in df_cool.columns:
            df_cool[m] = df_cool[m].round(4)
    return df_cool


def clean_items_df(df_items):
    """Format itemsets DataFrame for display"""
    if df_items.empty:
        return df_items
    df_cool = df_items.copy()
    df_cool['itemsets'] = df_cool['itemsets'].apply(lambda x: f"[{', '.join(list(x))}]")
    df_cool['support'] = df_cool['support'].round(4)
    return df_cool


def calculate_metric_range(rules_df, metric='lift'):
    """Calculate and display metric range"""
    if rules_df.empty:
        return f"{metric} Range: N/A (No rules generated)"
    min_val = rules_df[metric].min()
    max_val = rules_df[metric].max()
    return f"{metric} Range: {max_val - min_val:.4f} (Min: {min_val:.4f}, Max: {max_val:.4f})"

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_top_products(basket_list, title_suffix=""):
    """Plot top 10 most frequent products"""
    te = TransactionEncoder()
    te_ary = te.fit(basket_list).transform(basket_list)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    top_items = df_trans.sum().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 5))
    top_items.plot(kind='bar', color='#5c6bc0', edgecolor='black')
    plt.title(f'Top 10 Most Frequent Products - {title_suffix}', fontweight='bold')
    plt.ylabel('Absolute Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"top_products_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()


def plot_association_results(rules_df, title_suffix=""):
    """Plot association rules scatter plot"""
    if rules_df.empty:
        return
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(rules_df['support'], rules_df['confidence'], 
                          c=rules_df['lift'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Lift')
    plt.title(f'Rules Scatter Plot (Support vs Confidence) - {title_suffix}', fontweight='bold')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"rules_scatter_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()


# ============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS
# ============================================================================

def perform_global_analysis(df_basket_clusters, min_support=0.02, min_confidence=0.15):
    """
    Run and display association rules for the entire dataset.
    """
    print("================================================================================")
    print("=== PART 1: GLOBAL ASSOCIATION RULES (ALL CUSTOMERS COMBINED) ===")
    print("================================================================================")
    
    all_baskets = df_basket_clusters['list_of_goods_parsed'].tolist()
    plot_top_products(all_baskets, "Global Dataset")
    
    items_all, rules_all = run_association_rules(
        all_baskets, 
        min_support=min_support, 
        min_confidence=min_confidence, 
        split_train=True
    )
    
    print(f"\n✓ Frequent Itemsets: {len(items_all)}")
    print(f"✓ Association Rules: {len(rules_all)}")
    
    if not rules_all.empty:
        display(clean_rules_df(rules_all[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15)))
        print(f"\n{calculate_metric_range(rules_all, 'lift')}")
        plot_association_results(rules_all, "Global Dataset")


def perform_cluster_analysis(df_basket_clusters, min_support=0.02, min_confidence=0.15):
    """
    Run and display association rules for each individual cluster.
    """
    print("\n================================================================================")
    print("=== PART 2: STANDARDIZED ASSOCIATION RULES ANALYSIS BY CLUSTER ===")
    print("================================================================================")
    
    for cluster_id, cluster_name in CLUSTER_NAMES_MAP.items():
        print(f"\n------------------------------------------------------------")
        print(f"ANALYSIS FOR CLUSTER {cluster_id}: {cluster_name}")
        print("------------------------------------------------------------")
        
        df_cluster = df_basket_clusters[df_basket_clusters['cluster'] == cluster_name]
        cluster_baskets = df_cluster['list_of_goods_parsed'].tolist()
        
        print(f"Total transactions within this segment: {len(cluster_baskets)}")
        
        if len(cluster_baskets) > 0:
            plot_top_products(cluster_baskets, cluster_name)
            
            items_cl, rules_cl = run_association_rules(
                cluster_baskets, 
                min_support=min_support, 
                min_confidence=min_confidence, 
                split_train=True
            )
            
            print(f"\n   -> Frequent Itemsets Found: {len(items_cl)}")
            if not items_cl.empty:
                display(clean_items_df(items_cl.head(10)))
            
            print(f"\n   -> Association Rules Found: {len(rules_cl)}")
            if not rules_cl.empty:
                display(clean_rules_df(rules_cl[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)))
                print(f"\n      {calculate_metric_range(rules_cl, 'lift')}")
                plot_association_results(rules_cl, cluster_name)