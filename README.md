## This is my first Repository
## Customer Satisfaction Prediction (EDA + ML) Project
#  Author- Kartikeswar Acharya


Analyze customer support tickets, generate publication-ready plots, and train baseline ML models to predict customer satisfaction.

## Dataset
- File: `customer_support_tickets.csv`
- Default path in examples: `/content/customer_support_tickets.csv` (Colab). Change to `./data/customer_support_tickets.csv` if local.
- Example shape from EDA: 8,469 × 17

## Setup

```bash
# (Optional) create venv
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -U numpy pandas matplotlib seaborn scikit-learn xgboost
```

Create a `plots/` folder:
```bash
mkdir -p plots
```

## Load Data

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# for modeling
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier  # optional

# Paths
CSV_PATH = "/content/customer_support_tickets.csv"  # change if needed
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load
df = pd.read_csv(CSV_PATH)
pd.set_option('display.max_columns', None)

print(df.shape)
df.head()
```

## EDA: Trends, Segments, Distributions

Set style:
```python
sns.set(style="whitegrid")
```

### 1) Ticket Trends Over Time
```python
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'])
df['Year-Month'] = df['Date of Purchase'].dt.to_period('M')
ticket_trends = df.groupby('Year-Month').size()

plt.figure(figsize=(10, 6))
ticket_trends.plot(kind='line', marker='o', color='g')
plt.title('Customer Support Ticket Trends Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Tickets')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ticket_trends.png", dpi=200)
plt.show()
```

![Ticket Trends](plots/ticket_trends.png)

### 2) Top 10 Common Issues (Ticket Subject)
```python
common_issues = df['Ticket Subject'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=common_issues.values, y=common_issues.index, palette='viridis')
plt.title('Top 10 Common Issues')
plt.xlabel('Count')
plt.ylabel('Ticket Subject')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/common_issues.png", dpi=200)
plt.show()
```

![Common Issues](plots/common_issues.png)

### 3) Satisfaction Distribution
```python
plt.figure(figsize=(10, 6))
sns.histplot(df['Customer Satisfaction Rating'], bins=5, kde=True, color='skyblue')
plt.title('Customer Satisfaction Distribution')
plt.xlabel('Satisfaction Rating')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/satisfaction_distribution.png", dpi=200)
plt.show()
```

![Satisfaction Distribution](plots/satisfaction_distribution.png)

### 4) Ticket Status Distribution
```python
ticket_status_distribution = df['Ticket Status'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(ticket_status_distribution, labels=ticket_status_distribution.index,
        autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=180)
plt.title('Ticket Status Distribution')
plt.axis('equal')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ticket_status_distribution.png", dpi=200)
plt.show()
```

![Ticket Status](plots/ticket_status_distribution.png)

### 5) Age Distribution
```python
plt.figure(figsize=(10, 6))
sns.histplot(df['Customer Age'], bins=20, kde=True, color='salmon')
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/age_distribution.png", dpi=200)
plt.show()
```

![Age Distribution](plots/age_distribution.png)

### 6) Gender Distribution
```python
customer_gender_distribution = df['Customer Gender'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(customer_gender_distribution, labels=customer_gender_distribution.index,
        autopct='%1.1f%%', colors=sns.color_palette('Set2'), startangle=90)
plt.title('Customer Gender Distribution')
plt.axis('equal')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/gender_distribution.png", dpi=200)
plt.show()
```

![Gender Distribution](plots/gender_distribution.png)

### 7) Ticket Channel Distribution
```python
plt.figure(figsize=(15, 7))
ticket_channel_distribution = df['Ticket Channel'].value_counts()
sns.barplot(x=ticket_channel_distribution.index, y=ticket_channel_distribution.values,
            palette='rocket')
plt.title('Ticket Channel Distribution')
plt.xlabel('Ticket Channel')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/channel_distribution.png", dpi=200)
plt.show()
```

![Channel Distribution](plots/channel_distribution.png)

### 8) Average Satisfaction by Gender
```python
average_satisfaction = df.groupby('Customer Gender')['Customer Satisfaction Rating'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(x='Customer Gender', y='Customer Satisfaction Rating', data=average_satisfaction,
            order=['Male', 'Female', 'Other'], palette='dark')
plt.title('Average Customer Satisfaction by Gender')
plt.xlabel('Customer Gender')
plt.ylabel('Average Satisfaction Rating')
plt.ylim(1, 5)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/avg_satisfaction_by_gender.png", dpi=200)
plt.show()
```

![Avg Sat by Gender](plots/avg_satisfaction_by_gender.png)

### 9) Top 10 Products Purchased
```python
plt.figure(figsize=(10, 6))
product_purchased_distribution = df['Product Purchased'].value_counts().head(10)
sns.barplot(y=product_purchased_distribution.index, x=product_purchased_distribution.values, palette='magma')
plt.title('Top 10 Products Purchased')
plt.xlabel('Count')
plt.ylabel('Product')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/top_products.png", dpi=200)
plt.show()
```

![Top Products](plots/top_products.png)

### 10) Top Items Purchased by Gender (3 Subplots)
```python
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
top_items_male = df[df['Customer Gender'] == 'Male']['Product Purchased'].value_counts().head(5)
top_items_male.plot(kind='barh', color='skyblue')
plt.title('Top Items Purchased by Males')
plt.xlabel('Count')
plt.ylabel('Product')

plt.subplot(1, 3, 2)
top_items_female = df[df['Customer Gender'] == 'Female']['Product Purchased'].value_counts().head(5)
top_items_female.plot(kind='barh', color='salmon')
plt.title('Top Items Purchased by Females')
plt.xlabel('Count')
plt.ylabel('Product')

plt.subplot(1, 3, 3)
top_items_other = df[df['Customer Gender'] == 'Other']['Product Purchased'].value_counts().head(5)
top_items_other.plot(kind='barh', color='lightgreen')
plt.title('Top Items Purchased by Other Genders')
plt.xlabel('Count')
plt.ylabel('Product')

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/top_items_by_gender.png", dpi=200)
plt.show()
```

![Top Items by Gender](plots/top_items_by_gender.png)

### 11) Ticket Type Distribution
```python
ticket_type_distribution = df['Ticket Type'].value_counts()

plt.figure(figsize=(8, 6))
ticket_type_distribution.plot(kind='pie', autopct='%1.1f%%',
                              colors=['skyblue', 'salmon', 'lightgreen', 'plum', 'gold'])
plt.title('Ticket Type Distribution')
plt.ylabel('')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ticket_type_distribution.png", dpi=200)
plt.show()
```

![Ticket Type Distribution](plots/ticket_type_distribution.png)

### 12) Priority Level Distribution
```python
priority_distribution = df['Ticket Priority'].value_counts()

plt.figure(figsize=(8, 6))
priority_distribution.plot(kind='pie', autopct='%1.1f%%',
                           colors=['lightblue', 'lightgreen', 'lightsalmon', 'skyblue'])
plt.title('Priority Level Distribution')
plt.ylabel('')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/priority_distribution.png", dpi=200)
plt.show()
```

![Priority Distribution](plots/priority_distribution.png)

### 13) Tickets by Age Group
```python
bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70','71-80', '81-90', '91-100']
df['Age Group'] = pd.cut(df['Customer Age'], bins=bins, labels=labels, right=False)

tickets_by_age_group = df.groupby('Age Group').size()

plt.figure(figsize=(10, 6))
tickets_by_age_group.plot(kind='bar', color='skyblue')
plt.title('Tickets Raised by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Tickets')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/tickets_by_age_group.png", dpi=200)
plt.show()
```

![Tickets by Age Group](plots/tickets_by_age_group.png)

### 14) Age Distribution Faceted by Ticket Type
```python
g = sns.FacetGrid(df, col='Ticket Type', col_wrap=3, height=4, aspect=1.2)
g.map(sns.histplot, 'Customer Age', bins=20, kde=True)
g.set_titles('{col_name}')
g.set_axis_labels('Age', 'Number of Tickets')
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Distribution of Ticket Types by Age')
plt.savefig(f"{PLOTS_DIR}/facet_ticket_type_by_age.png", dpi=200, bbox_inches='tight')
plt.show()
```

![Facet Age by Ticket Type](plots/facet_ticket_type_by_age.png)

## Baseline ML: Predict Satisfaction Rating

This section creates a simple supervised baseline to predict `Customer Satisfaction Rating` using basic categorical features. Rows with missing target are dropped.

```python
# Drop rows with null target
df_ml = df.dropna(subset=['Customer Satisfaction Rating']).copy()

# Define target (treat as classification; cast to int if ratings are 1-5 floats)
y = df_ml['Customer Satisfaction Rating'].astype(int)

# Simple feature set (extend as needed)
feature_cols_cat = ['Customer Gender', 'Product Purchased', 'Ticket Type',
                    'Ticket Status', 'Ticket Priority', 'Ticket Channel']
feature_cols_num = ['Customer Age']

X = df_ml[feature_cols_cat + feature_cols_num]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess: one-hot for categoricals, passthrough numeric
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
preprocess = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, feature_cols_cat),
        ('num', 'passthrough', feature_cols_num),
    ]
)

# Model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# Pipeline
pipe = Pipeline(steps=[('prep', preprocess), ('clf', rf)])

# Train
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", round(acc, 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

### Confusion Matrix Plot
```python
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
plt.title('RandomForest Confusion Matrix (Satisfaction Rating)')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix_rf.png", dpi=200)
plt.show()
```

![Confusion Matrix RF](plots/confusion_matrix_rf.png)

### Optional: XGBoost Baseline
```python
# from xgboost import XGBClassifier
# xgb = XGBClassifier(
#     n_estimators=400,
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     random_state=42,
#     n_jobs=-1,
#     tree_method='hist'
# )
# pipe_xgb = Pipeline(steps=[('prep', preprocess), ('clf', xgb)])
# pipe_xgb.fit(X_train, y_train)
# y_pred_xgb = pipe_xgb.predict(X_test)
# print("XGB Accuracy:", round(accuracy_score(y_test, y_pred_xgb), 4))
```

## Reproducibility Notes
- Many `Customer Satisfaction Rating` values are missing in raw data. We drop them for supervised training here; consider imputation or semi-supervised approaches for production.
- Set `random_state` for deterministic splits.

## Generate All Figures At Once (Optional)
```python
# Simply run the EDA cells in order. Ensure PLOTS_DIR exists.
# All figures are saved under the 'plots/' directory as shown above.
```

## License
Add your chosen license, e.g., MIT, in `LICENSE`.

## Acknowledgments
Built with pandas, seaborn, scikit-learn, and (optional) XGBoost.