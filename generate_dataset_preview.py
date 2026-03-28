"""
Generate Dataset Preview Visualization for Report
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load dataset
df = pd.read_csv('data/raw/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Create dataset preview figure
fig = plt.figure(figsize=(14, 10))

# 1. Sample messages table
ax1 = plt.subplot(2, 2, (1, 2))
ax1.axis('off')

# Select diverse samples
ham_samples = df[df['label'] == 'ham'].sample(4, random_state=42)
spam_samples = df[df['label'] == 'spam'].sample(4, random_state=42)
samples = pd.concat([ham_samples, spam_samples]).reset_index(drop=True)

# Create table data with truncated messages
table_data = []
for idx, row in samples.iterrows():
    msg = row['message']
    if len(msg) > 60:
        msg = msg[:60] + '...'
    table_data.append([idx+1, row['label'].upper(), msg])

# Create table
table = ax1.table(cellText=table_data,
                  colLabels=['#', 'Label', 'Message'],
                  cellLoc='left',
                  loc='center',
                  colWidths=[0.05, 0.08, 0.87])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white')

# Style cells based on label
for i in range(1, 9):
    label_cell = table[(i, 1)]
    if table_data[i-1][1] == 'HAM':
        label_cell.set_facecolor('#d5f4e6')
        label_cell.set_text_props(color='#27ae60', weight='bold')
    else:
        label_cell.set_facecolor('#fadbd8')
        label_cell.set_text_props(color='#e74c3c', weight='bold')

ax1.set_title('Dataset Sample Preview', fontsize=14, fontweight='bold', pad=20)

# 2. Class distribution pie chart
ax2 = plt.subplot(2, 2, 3)
class_counts = df['label'].value_counts()
colors = ['#3498db', '#e74c3c']
explode = (0.05, 0.05)

wedges, texts, autotexts = ax2.pie(class_counts.values,
                                     labels=['Ham (Legitimate)', 'Spam'],
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     colors=colors,
                                     explode=explode,
                                     shadow=True,
                                     textprops={'fontsize': 11, 'weight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_weight('bold')

ax2.set_title('Class Distribution', fontsize=12, fontweight='bold', pad=10)

# 3. Dataset statistics
ax3 = plt.subplot(2, 2, 4)
ax3.axis('off')

stats_data = [
    ['Total Messages', f'{len(df):,}'],
    ['Ham Messages', f'{(df["label"] == "ham").sum():,} ({(df["label"] == "ham").sum()/len(df)*100:.1f}%)'],
    ['Spam Messages', f'{(df["label"] == "spam").sum():,} ({(df["label"] == "spam").sum()/len(df)*100:.1f}%)'],
    ['Avg Message Length', f'{df["message"].str.len().mean():.1f} characters'],
    ['Max Message Length', f'{df["message"].str.len().max()} characters'],
    ['Min Message Length', f'{df["message"].str.len().min()} characters'],
    ['Train Set (80%)', f'{int(len(df) * 0.8):,} messages'],
    ['Test Set (20%)', f'{int(len(df) * 0.2):,} messages'],
]

stats_table = ax3.table(cellText=stats_data,
                        colLabels=['Statistic', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.5, 0.5])

stats_table.auto_set_font_size(False)
stats_table.set_fontsize(10)
stats_table.scale(1, 2.2)

# Style header
for i in range(2):
    cell = stats_table[(0, i)]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(stats_data) + 1):
    for j in range(2):
        cell = stats_table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        if j == 1:  # Value column
            cell.set_text_props(weight='bold', color='#2c3e50')

ax3.set_title('Dataset Statistics', fontsize=12, fontweight='bold', pad=10)

plt.suptitle('SMS Spam Collection Dataset Overview', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/plots/dataset_preview.png', dpi=150, bbox_inches='tight')
print("Dataset preview saved to: results/plots/dataset_preview.png")

# Also create a simple table image with actual message examples
fig2, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Better examples
examples = [
    ['1', 'HAM', 'Go until jurong point, crazy.. Available only in bugis n great world la e buffet...'],
    ['2', 'HAM', 'Ok lar... Joking wif u oni...'],
    ['3', 'HAM', 'U dun say so early hor... U c already then say...'],
    ['4', 'HAM', 'Nah I don\'t think he goes to usf, he lives around here though'],
    ['5', 'SPAM', 'FreeMsg Hey there darling it\'s been 3 week\'s now and no word back! I\'d like...'],
    ['6', 'SPAM', 'WINNER!! As a valued network customer you have been selected to receive...'],
    ['7', 'SPAM', 'Urgent! You have won a 1 week FREE membership in our £100,000 Prize...'],
    ['8', 'SPAM', 'XXXMobileMovieClub: To use your credit, click the WAP link in the next txt...'],
]

table2 = ax.table(cellText=examples,
                  colLabels=['ID', 'Label', 'Message Text (Truncated)'],
                  cellLoc='left',
                  loc='center',
                  colWidths=[0.05, 0.08, 0.87])

table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2.8)

# Style header
for i in range(3):
    cell = table2[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Color code by label
for i in range(1, 9):
    label_cell = table2[(i, 1)]
    row_label = examples[i-1][1]

    if row_label == 'HAM':
        label_cell.set_facecolor('#d5f4e6')
        label_cell.set_text_props(color='#27ae60', weight='bold', fontsize=10)
        # Light green background for entire row
        for j in range(3):
            if j != 1:
                table2[(i, j)].set_facecolor('#f0fdf4')
    else:
        label_cell.set_facecolor('#fee2e2')
        label_cell.set_text_props(color='#dc2626', weight='bold', fontsize=10)
        # Light red background for entire row
        for j in range(3):
            if j != 1:
                table2[(i, j)].set_facecolor('#fef2f2')

plt.title('SMS Spam Collection - Sample Messages', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/plots/dataset_samples.png', dpi=150, bbox_inches='tight')
print("Dataset samples saved to: results/plots/dataset_samples.png")

print("\nDataset preview images generated successfully!")
print("Files created:")
print("  - results/plots/dataset_preview.png (comprehensive overview)")
print("  - results/plots/dataset_samples.png (message examples)")
