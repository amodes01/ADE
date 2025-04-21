import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Path to the JSON file
json_file_path = './detailed_evaluation_results.json'

# Read the JSON file
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# Variable containing the JSON data
data = json_data

categories = ['entity', 'topical']
sectors = ['pair_matching', 'attitude_prediction']

precision = []
recall = []
f1 = []

for cat in categories:
    precision_list = []
    recall_list = []
    f1_list = []
    for sample in data['samples'].values():
        for sec in sectors:
            if sec in sample['metrics'][cat]:  
                precision_list.append(sample['metrics'][cat][sec]['precision'])
                recall_list.append(sample['metrics'][cat][sec]['recall'])
                f1_list.append(sample['metrics'][cat][sec]['f1'])
    # print("PRECISION: ", precision_list)
    # print("="*50)
    # print("RECALL: ", recall_list)
    # print("="*50)
    # print("F1: ", f1_list)
    # print("="*50)
    precision.append(precision_list)
    recall.append(recall_list)
    f1.append(f1_list)

out_data_entity = {
    'model': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for i in range(len(precision[0])):  
    out_data_entity['model'].append('entity')
    out_data_entity['precision'].append(precision[0][i])
    out_data_entity['recall'].append(recall[0][i])
    out_data_entity['f1'].append(f1[0][i])

df = pd.DataFrame(out_data_entity)

# Melt the DataFrame to long format
df_melted = df.melt(id_vars='model', var_name='metric', value_name='score')

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='metric', y='score', hue='model', data=df_melted)
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.legend(title='Model')
plt.tight_layout()
plt.show()

# Save the plot as a PNG file
plt.savefig('entity_model_performance_metrics.png', dpi=300)

# out_data_topic = {
#     'model': [],
#     'precision': [],
#     'recall': [],
#     'f1': []
# }

# for i in range(len(precision[1])):  
#     out_data_topic['model'].append('topical')
#     out_data_topic['precision'].append(precision[1][i])
#     out_data_topic['recall'].append(recall[1][i])
#     out_data_topic['f1'].append(f1[1][i])

# df = pd.DataFrame(out_data_topic)

# # Melt the DataFrame to long format
# df_melted = df.melt(id_vars='model', var_name='metric', value_name='score')

# # Create boxplot
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='metric', y='score', hue='model', data=df_melted)
# plt.title('Model Performance Metrics')
# plt.ylabel('Score')
# plt.xlabel('Metric')
# plt.legend(title='Model')
# plt.tight_layout()
# plt.show()
