import pandas as pd
import json
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

class BasketAnalyzer:
    def __init__(self, basket_path: str, clusters_path: str):
        # Carregamento e preparação idênticos
        self.df = self._prepare_data(basket_path, clusters_path)

    def _prepare_data(self, basket_path: str, clusters_path: str) -> pd.DataFrame:
        basket = pd.read_csv(basket_path)
        clusters = pd.read_csv(clusters_path)
        
        # Mapeamento de clusters (conforme o seu CLUSTER_NAMES_MAP)
        cluster_map = {
            0: "The Plant-Based Household", 1: "The Family Budget Optimizer",
            2: "The Promotion-Driven Pet Parent", 3: "The Demanding Premium Tech Consumer",
            4: "The Wellness Customer", 5: "The Affluent Health-Conscious Buyer"
        }
        clusters['cluster'] = clusters['cluster'].map(cluster_map)
        
        df = clusters.merge(basket, on='customer_id')
        df['list_of_goods_parsed'] = df['list_of_goods'].apply(lambda x: json.loads(x.replace("'", '"')))
        return df

    def _clean_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df_c = df.copy()
        # Formatação idêntica: lista com strings dentro de colchetes
        df_c['antecedents'] = df_c['antecedents'].apply(lambda x: f"[{', '.join(list(x))}]")
        df_c['consequents'] = df_c['consequents'].apply(lambda x: f"[{', '.join(list(x))}]")
        return df_c.round(4)

    def _clean_items(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df_c = df.copy()
        df_c['itemsets'] = df_c['itemsets'].apply(lambda x: f"[{', '.join(list(x))}]")
        return df_c.round(4)

    def get_analysis(self, cluster_name: str = None, min_sup: float = 0.02, min_conf: float = 0.15):
        """Retorna o dataset filtrado, itemsets e regras limpas."""
        target_df = self.df if cluster_name is None else self.df[self.df['cluster'] == cluster_name]
        data = target_df['list_of_goods_parsed'].tolist()
        
        te = TransactionEncoder()
        df_trans = pd.DataFrame(te.fit(data).transform(data), columns=te.columns_)
        
        itemsets = apriori(df_trans, min_support=min_sup, use_colnames=True)
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_conf)
        
        return self._clean_items(itemsets.head(10)), self._clean_rules(rules.sort_values('lift', ascending=False))

# --- Execução Equivalente ---
if __name__ == "__main__":
    analyzer = BasketAnalyzer('customer_basket (1).csv', 'dataset_clusters.csv')
    
    # 1. Global
    items, rules = analyzer.get_analysis()
    print("=== GLOBAL ASSOCIATION RULES ===")
    display(items)
    display(rules.head(15))
    
    # 2. Por Cluster
    for cluster in analyzer.df['cluster'].dropna().unique():
        print(f"\n--- Cluster: {cluster} ---")
        items, rules = analyzer.get_analysis(cluster_name=cluster)
        display(items.head(10))
        display(rules.head(10))