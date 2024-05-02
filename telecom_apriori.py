from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

# Sample transactions
transactions = {
    'transaction_id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'products': [['cable', 'internet', 'sim', 'landline'],
                 ['internet', 'sim', 'landline'],
                 ['cable', 'internet', 'landline'],
                 ['cable', 'internet', 'sim'],
                 ['cable', 'internet', 'sim', 'landline'],
                 ['internet', 'sim', 'landline'],
                 ['cable', 'internet', 'landline'],
                 ['cable', 'internet', 'sim'],
                 ['cable', 'internet', 'sim', 'landline'],
                 ['internet', 'sim', 'landline']]
}

# Convert transactions to DataFrame
df = pd.DataFrame(transactions)

# Convert list of products to a string
df['products'] = df['products'].apply(lambda x: ','.join(x))

# Apply one-hot encoding to the transactions
one_hot_encoded = df['products'].str.get_dummies(',')

# Apply Apriori algorithm
frequent_itemsets = apriori(one_hot_encoded, min_support=0.3, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Print frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Print association rules
print("\nAssociation Rules:")
print(rules)

