from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

# Sample dataset containing symptom-disease relationships
data = {
    'Symptoms': [['fever', 'cough', 'fatigue'],
                 ['fever', 'headache'],
                 ['cough', 'runny nose'],
                 ['fever', 'cough', 'sore throat'],
                 ['fatigue', 'body ache'],
                 ['headache', 'fatigue'],
                 ['fever', 'cough', 'fatigue', 'headache']],
    'Disease': ['Flu', 'Migraine', 'Common Cold', 'Strep Throat', 'Flu', 'Migraine', 'Flu']
}

df = pd.DataFrame(data)

# Transform the data into one-hot encoded format
def encode_units(x):
    if x:
        return 1
    else:
        return 0

symptom_disease = (df
                   .explode('Symptoms')
                   .groupby(['Disease', 'Symptoms'])['Symptoms']
                   .count().unstack().reset_index().fillna(0)
                   .set_index('Disease'))

basket_sets = symptom_disease.applymap(encode_units)

# Generate frequent itemsets using Apriori algorithm
frequent_itemsets = apriori(basket_sets, min_support=0.3, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Print the frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociated Rules:")
print(rules)
