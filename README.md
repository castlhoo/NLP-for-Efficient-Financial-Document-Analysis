# 📊 Topic Modeling and Document Similarity Analysis for Financial Documents

## 🎯 1. Data Selection Reason: Quick Processing of Financial Documents and Clusterization for Enhanced Efficiency
In this project, we focused on financial accounting documents, aiming to quickly process and summarize these documents using NLP techniques. The ultimate goal is to improve workflow efficiency by leveraging clustering and topic modeling techniques. Financial documents often contain vast amounts of structured and unstructured data, and this project aims to automate the processing and analysis of such information.

## 💡 2. Practical Application: Topic Modeling through Natural Language Processing (NLP) and Document Similarity
To demonstrate the power of NLP, we applied Topic Modeling and Document Similarity techniques on the financial documents. Topic modeling identifies the underlying topics within the documents, while document similarity measures how similar two or more documents are to each other. Both techniques help in summarizing, clustering, and understanding large sets of financial data more effectively.

## 🔍 3. Code Explanation: Detailed Breakdown
Now, let's break down the provided code and explain each part in detail. I'll walk you through the entire process, from reading and summarizing the documents to applying topic modeling and document clustering.

### 🔄 Step 1: Loading Google Drive and Installing Required Packages
The first step involves mounting Google Drive and installing necessary packages.

```python
from google.colab import drive
drive.mount('/content/gdrive')

!pip install openai==0.28.1
```

Here, we're mounting Google Drive so we can access the files stored there. We're also installing the OpenAI package for text summarization.

### 📝 Step 2: Summarizing Financial Documents
The core of this project is to summarize each financial document into a brief paragraph for analysis. We use BeautifulSoup to extract text from the HTML files and send this text to the OpenAI API for summarization.

```python
import os
import json
import time
from bs4 import BeautifulSoup
import openai

file_path = "/content/gdrive/MyDrive/Colab Notebooks/Notes"
file_list = os.listdir(file_path)

if os.path.exists("/content/gdrive/MyDrive/Colab Notebooks/f_summary.json"):
    with open("f_summary.json", "r", encoding="utf-8") as f:
        file_tables = json.load(f)
else:
    file_tables = {}

def html_to_summary(file_list):
  for file in file_list:
    if file in file_tables:
        continue
    try:
        with open(os.path.join(file_path, file), 'r', encoding='utf-8') as f:
            html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n')

        prompt = f"""Summarize the following financial information into a short paragraph for analysis:\n{text[:3000]}"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        summary = response['choices'][0]['message']['content']
        file_tables[file] = summary

        with open("/content/gdrive/MyDrive/Colab Notebooks/file_summary.json", "w", encoding="utf-8") as f:
            json.dump(file_tables, f, ensure_ascii=False, indent=2)

        print(f"[{len(file_tables)} / {len(file_list)}] summarized: {file}")
        time.sleep(1)

    except Exception as e:
        print(f"ERROR on {file}: {e}")
        continue

  return file_tables

result = html_to_summary(file_list)
```

In this block of code:

- BeautifulSoup is used to extract the text from each HTML file.
- We then use the OpenAI API to generate a summary of the extracted text.
- The summary is stored in a dictionary, which is saved as a JSON file to avoid recalculating summaries in future runs.

### 📊 Step 3: Transforming the Summaries into a DataFrame
Once the summaries are obtained, we store the file names and their summaries into a DataFrame:

```python
import pandas as pd
audit_df = pd.DataFrame({
    "File_name" : [k.replace('.html','') for k in result.keys()],
    "contents" : result.values()
})
```

This makes it easier to work with the data and apply further analysis.

### 🧩 Step 4: Topic Modeling using Latent Dirichlet Allocation (LDA)
Now, let's move to topic modeling, where we use Latent Dirichlet Allocation (LDA) to extract topics from the financial documents.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import numpy as np

# GridSearch parameters for optimization
param_grid = {
    'c_vect__ngram_range': [(1, 2), (1, 3)],
    'c_vect__max_df': [0.5, 0.75, 1.0],
    'c_vect__min_df': [1, 2, 3],
    'lda__n_components': [5, 7, 10]
}

# Pipeline for Topic Modeling
pipeline = Pipeline([
    ('c_vect', CountVectorizer(stop_words='english')),
    ('lda', LatentDirichletAllocation(random_state=0))
])

# Running GridSearch for optimization
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid_search.fit(audit_topic_df['contents'])

print(f"Best Parameters: {grid_search.best_params_}")
```

This part of the code performs the following:

- GridSearchCV is used to find the optimal parameters for the CountVectorizer and LDA models.
- CountVectorizer transforms the text data into a matrix of token counts, which is then passed into the LDA model for topic extraction.
- The best parameters are printed out once the grid search completes.

### 📋 Step 5: Displaying Topics
Once the model is trained, we display the top words for each topic:

```python
def display_topic_words(model, feature_names, no_top_words):
    for topic_index, topic in enumerate(model.components_):
        print(f'\nTopic #{topic_index}:')
        topic_word_indexes = topic.argsort()[::-1]  # Sort words by descending order of importance
        top_indexes = topic_word_indexes[:no_top_words]

        feature_concat = ' '.join([feature_names[i] for i in top_indexes])  # Join top words
        print(feature_concat)

# Get feature names (words) from the vectorizer
feature_names = best_model.named_steps['c_vect'].get_feature_names_out()

# Display the top 15 words for each topic
display_topic_words(best_model.named_steps['lda'], feature_names, 15)
```

### 🔗 Step 6: Document Similarity using Cosine Similarity
Next, we perform document clustering using KMeans clustering and TF-IDF vectorization. The goal is to find similar documents and group them into clusters.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import string

# Lemmatization function
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# KMeans Clustering with TF-IDF Vectorizer
audit_cluster_df = audit_df.copy()

param_grid = {
    'kmeans__n_clusters': [4, 5, 6, 7, 8, 10]
}

# Pipeline
pipeline = Pipeline([
    ('tfid_vect', TfidfVectorizer(tokenizer = LemNormalize, ngram_range=(1,2), min_df=3, max_df=0.5, stop_words='english')),
    ('kmeans', KMeans(random_state=0, max_iter = 1000))
])

# Run GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid_search.fit(audit_topic_df['contents'])

print(f"Best Parameters: {grid_search.best_params_}")
```

### 📈 Step 7: Displaying the Clusters
Finally, we display the clusters and the top words that define each cluster:

```python
def print_cluster(df):
    num_clusters = len(df['cluster'].unique())
    for i in range(num_clusters):
        print(f"\nCluster {i}:")
        print(df[df['cluster'] == i].sort_values(by='contents'))

# Print the clusters
print_cluster(audit_cluster_df)
```

## 📊 4. Topic Modeling Results / Document Similarity Results
### 🔍 Topic Modeling Results:
Based on the parameters provided, the following topics were extracted from the financial documents:

**Topic #0: Equity, Shares, and Capital** 💰

Top Words: shares, equity, investment, capital, tax, profit, etc.

Interpretation: Documents in this topic focus on capital structures, equity shares, and financial performance relating to investments and profit.

**Topic #1: Loans and Borrowings** 💸

Top Words: loans, liabilities, assets, borrowing, term, income, etc.

Interpretation: This topic focuses on loan structures, including secured and unsecured borrowings, and their impact on financial statements.

**Topic #2: Expenses and Employee Advances** 👨‍💼

Top Words: expenses, increase, employee, advances, previous year, etc.

Interpretation: This topic deals with expenses, particularly employee-related advances and their year-on-year changes.

**Topic #3: Cash Flow and Liabilities** 💵

Top Words: cash, liabilities, march, increase, payables, crores, etc.

Interpretation: This topic covers liabilities, particularly trade payables, and cash flow management, especially near fiscal year-end (March 31).

**Topic #4: Asset Depreciation and Equipment Value** 🏭

Top Words: assets, value, depreciation, equipment, carrying, plant, etc.

Interpretation: This topic focuses on the depreciation of assets, including equipment and plant, and the impact on company financials.

### 🔗 Document Similarity Results:
Using KMeans clustering and TF-IDF, documents were clustered into 10 distinct groups based on their similarities. Here's a detailed breakdown of each cluster:

**Cluster 0: Equity and Investment** 📈

Top Words: shares, equity, investment, capital

Interpretation: This cluster includes documents related to stock issuance and **capital management.

**Cluster 1: Asset Valuation and Depreciation** 📉

Top Words: asset, depreciation, net value, equipment

Interpretation: Focuses on the valuation and depreciation of assets, especially equipment.

**Cluster 2: Loans and Borrowings** 🏦

Top Words: loan, borrowing, advance, bank, unsecured

Interpretation: This cluster centers around loan agreements and borrowings, with a focus on unsecured loans.

**Cluster 3: Financial Figures and Currency** 💱

Top Words: INR, crore, liabilities, cash, 2018

Interpretation: Contains data on INR and crore values, with a focus on liabilities and cash.

**Cluster 4: Employee Benefits and Provisions** 👥

Top Words: provision, employee, benefit, gratuity

Interpretation: Focused on employee benefits, gratuity, and other employee-related provisions.

**Cluster 5: Liabilities and Payables** 💳

Top Words: payable, liability, trade payable, financial liability

Interpretation: Documents detailing liabilities and trade payables.

**Cluster 6: Cash Flow and Receivables** 💰

Top Words: cash, bank, receivables, USD

Interpretation: This cluster discusses cash flow management and outstanding receivables.

**Cluster 7: Financial Figures and Changes** 📊

Top Words: crore, liability, increase, current

Interpretation: Contains documents detailing financial changes in crore and liabilities.

**Cluster 8: Asset Depreciation and Management** ⏳

Top Words: useful life, depreciation, equipment, rate

Interpretation: Focuses on the useful life of assets, their depreciation, and equipment.

**Cluster 9: Tax and Income** 💵

Top Words: income, tax, expense, decrease

Interpretation: This cluster discusses income tax, expenses, and income-related documents.

## 🔄 5. Reflection
This project demonstrates how NLP techniques like Topic Modeling and Document Clustering can be used to effectively process and analyze large volumes of financial data. By categorizing documents into specific topics and clusters, it becomes easier to understand the key areas of focus, such as equity, loans, expenses, and liabilities. These methods can drastically improve the speed and accuracy of financial document analysis, enhancing decision-making processes and workflow efficiency.
