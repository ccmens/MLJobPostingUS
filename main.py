# ML Job Postings in the US - Project Based on Google Advanced Data Analytics

#import libraries and packages
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

import matplotlib.pyplot as plt
import seaborn as sns

# This displays all of the columns, preventing Juptyer from redacting them.
pd.set_option('display.max_columns', None)

# This module lets us save our models once we fit them.
import pickle

#import data
file='1000_ml_jobs_us.csv'
df=pd.read_csv(file)

#display first 10 rows of data
df.head()

#drop useless column
df.drop('Unnamed: 0',axis=1)

#check the size of dataset
df.shape

#check the data types in dataset 
df.dtypes

#check the data info
df.info()

#check the descriptive dataset
df.describe()

#check if there is any missing values
df.isna().sum()

#since the missing value of 'job_description_text' is very less, I choose to delete the row that the value is in.
df = df.dropna(subset=['job_description_text','seniority_level']).reset_index(drop=True)
#check the misssing value again to make sure the missing value in job_description_text' was deleted successfully
df.isna().sum()

#check varibles in the columns that have missing values
df['company_address_locality'].unique()
df['company_address_locality'].fillna('Unknown', inplace=True)

df['company_address_region'].unique()
df['company_address_region'].fillna('Unknown', inplace=True)

df['company_website'].fillna('Unknown', inplace=True)
df['company_description'].fillna('Unknown', inplace=True)
df.isna().sum()

#drop duplicates
df.drop_duplicates(inplace=True)

#What kind of job that the market/company needs the most
popular_job = df['job_title'].value_counts().head(10)
plt.figure(figsize=(12,8))
sns.barplot(x=popular_job.index, y=popular_job.values)
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 ML Job Titles")
plt.xlabel("Job Title")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#Analyze top locations for machine learning jobs
popular_locations = df['company_address_locality'].value_counts().head()
popular_region = df['company_address_region'].value_counts().head()

#Job posting trends over time
#convert job post date as datetime
df['job_posted_date'] = pd.to_datetime(df['job_posted_date'], errors='coerce')
df['month'] = df['job_posted_date'].dt.to_period('M')
time_trend=df['month'].value_counts().sort_index().reset_index(drop=False)
time_trend.columns = ['month', 'count']
time_trend['month'] = time_trend['month'].astype(str)
plt.figure(figsize=(10,6))
plt.xticks(rotation=45, ha='right')
sns.lineplot(data=time_trend, x='month', y='count', marker='o')
plt.title('Monthly Job Posting Trend')
plt.xlabel('Month')
plt.ylabel('Number of Postings')
plt.tight_layout()
plt.show()

#top hiring company
df['company_name'].value_counts().head(10).plot(kind='barh', figsize=(10,6), color='skyblue')
plt.title("Top Hiring Companies")
plt.xlabel("Number of Postings")
plt.ylabel("Company")
plt.tight_layout()
plt.show()

#check the seniority_level
df['seniority_level'].value_counts().plot(kind='pie',autopct='%1.1f%%',legend=True,figsize=(8,8))
plt.title("Seniority Level Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# (Optional Next Step) Keyword Extraction from Job Descriptions
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english', max_features=30)
X = vectorizer.fit_transform(df['job_description_text'].fillna(''))

keywords = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Plot keyword frequencies
keywords.sum().sort_values(ascending=False).head(20).plot(kind='barh', figsize=(8,6))
plt.title("Top Keywords in ML Job Descriptions")
plt.tight_layout()
plt.show()

#  Predict Seniority Level
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Filter valid labels
df_model = df.dropna(subset=['seniority_level'])
X_text = df_model['job_description_text'].fillna('')
y = df_model['seniority_level']

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(X_text)

X_text = df['job_description_text']
y = df['seniority_level']

# delete the categorical data with too less value
rare_classes = y.value_counts()[y.value_counts() < 2].index
mask = ~y.isin(rare_classes)
X_text = X_text[mask]
y = y[mask]

#  TF-IDF 
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf.fit_transform(X_text)

#  split the train, test values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Tune randomforest model 
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\ud83c\udf1f Classification Report:")
print(classification_report(y_test, y_pred))

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

#  Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# \u8bbe\u7f6e GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,  
    scoring='f1_weighted',  
    n_jobs=-1, 
    verbose=2
)

# fit the model to the training data.
grid_search.fit(X_train, y_train)

# Examine the best combination of hyperparameters.
print("\ud83d\udd0d Best Parameters:")
print(grid_search.best_params_)

# Examine the best estimator
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Results
print("\ud83d\udcca Tuned Model Classification Report:")
print(classification_report(y_test, y_pred_best))

# Step 2\uff1aXGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8, 1.0]
}

# Step 3\uff1aGrid Search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=3,
    verbose=2,
    n_jobs=-1
)
y_encoded, uniques = pd.factorize(y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
grid_search.fit(X_train, y_train)

print("\u2705 Best parameters found:")
print(grid_search.best_params_)

# Predict on best estimator
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)

# Results Report
print("\ud83d\udcca Classification Report:")
print(classification_report(y_test, y_pred))

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# save result
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = report
