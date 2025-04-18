from google.colab import drive
drive.mount('/content/gdrive')

import os
os.environ['OPENAI_API_KEY'] = "SECRET"  

import openai
openai.api_key = os.environ['OPENAI_API_KEY']

# 1. Crawling for HTML
## 1) getting all HTML file to text
import os
import json
import time
from bs4 import BeautifulSoup
import openai
import random

file_path = "SECRET"
file_list = os.listdir(file_path)

if os.path.exists("SECRET"):
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


      with open("SECRET", "w", encoding="utf-8") as f:
        json.dump(file_tables, f, ensure_ascii=False, indent=2)

      print(f"[{len(file_tables)} / {len(file_list)}] summarized: {file}")
      time.sleep(1)

    except Exception as e:
      print(f"ERROR on {file}: {e}")
      continue

  return file_tables

result = html_to_summary(file_list)
'''
[660 / 702] summarized: 18599651_table_200.html
[661 / 702] summarized: 18661963_table_23.html
[662 / 702] summarized: 18888243_table_91.html
[663 / 702] summarized: 18989758_table_194.html
[664 / 702] summarized: 19005595_table_207.html
[665 / 702] summarized: 18699662_table_40.html
[666 / 702] summarized: 18646845_table_103.html
[667 / 702] summarized: 18612942_table_41.html
[668 / 702] summarized: 18661965_table_43.html
[669 / 702] summarized: 18630222_table_153.html
[670 / 702] summarized: 18600361_table_47.html
[671 / 702] summarized: 18599651_table_165.html
[672 / 702] summarized: 19005595_table_215.html
[673 / 702] summarized: 19005595_table_214.html
[674 / 702] summarized: 18630219_table_44.html
[675 / 702] summarized: 18646837_table_79.html
[676 / 702] summarized: 18699662_table_47.html
[677 / 702] summarized: 18646837_table_85.html
[678 / 702] summarized: 18630222_table_159.html
[679 / 702] summarized: 19073312_table_210.html
[680 / 702] summarized: 18950920_table_127.html
[681 / 702] summarized: 18599651_table_110.html
[682 / 702] summarized: 19073396_table_338.html
[683 / 702] summarized: 18599651_table_111.html
[684 / 702] summarized: 18599651_table_176.html
[685 / 702] summarized: 19070958_table_182.html
[686 / 702] summarized: 18600361_table_43.html
[687 / 702] summarized: 18646837_table_123.html
[688 / 702] summarized: 18699662_table_42.html
[689 / 702] summarized: 18950920_table_176.html
[690 / 702] summarized: 19058354_table_38.html
[691 / 702] summarized: 18630219_table_75.html
[692 / 702] summarized: 18646837_table_109.html
[693 / 702] summarized: 19058355_table_102.html
[694 / 702] summarized: 18630222_table_162.html
[695 / 702] summarized: 18612942_table_13.html
[696 / 702] summarized: 19034589_table_84.html
[697 / 702] summarized: 18630219_table_45.html
[698 / 702] summarized: 18599651_table_151.html
[699 / 702] summarized: 18630219_table_70.html
[700 / 702] summarized: 18646837_table_84.html
[701 / 702] summarized: 19005595_table_242.html
[702 / 702] summarized: 18630222_table_154.html'''

# When colab memory out
import json

file_summary_path = "SECRET"

try:
    with open(file_summary_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    print("File summary successfully loaded.")
except FileNotFoundError:
    print(f"{file_summary_path} not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding the JSON file. It may be corrupted.")

## 2) Transition to DataFrame
#print(result)
import pandas as pd
audit_df = pd.DataFrame({
    "File_name" : [k.replace('.html','') for k in result.keys()],
    "contents" : result.values()
})

# * Topic Modeling
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import numpy as np

# Count Vectorizer
audit_topic_df = audit_df.copy()

# GridSearch
param_grid = {
    'c_vect__ngram_range': [(1, 2), (1, 3)],
    'c_vect__max_df': [0.5, 0.75, 1.0],
    'c_vect__min_df': [1, 2, 3],
    'lda__n_components': [5, 7, 10]
}

# Pipeline
pipeline = Pipeline([
    ('c_vect', CountVectorizer(stop_words='english')),
    ('lda', LatentDirichletAllocation(random_state=0))
])

# Run GridSearch
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid_search.fit(audit_topic_df['contents'])

print(f"Best Parameters: {grid_search.best_params_}")

# Best model
best_model = grid_search.best_estimator_

# Transform the text data into topic distributions
topic_dis = best_model.transform(audit_topic_df['contents'])

# Display top words for each topic
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
'''Best Parameters: {'c_vect__max_df': 0.5, 'c_vect__min_df': 3, 'c_vect__ngram_range': (1, 2), 'lda__n_components': 5}
Best Score: -99913.71542097542

Topic #0:
shares rs tax share million equity profit 000 capital december march previous equity shares 2018 december 31

Topic #1:
term loans income liabilities previous assets previous year long march march 31 long term current increase borrowings decrease

Topic #2:
expenses rs increase previous previous year march decrease 2018 increased compared march 31 decreased overall employee advances

Topic #3:
2018 cash march current investments crores liabilities lacs increased march 31 increase inr decreased trade payables

Topic #4:
assets years value depreciation december 31 december net information equipment carrying financial information provided block plant information provided'''

import nltk

nltk.download('punkt_tab')
nltk.download('wordnet')

# *Document Similarity with Cosine
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
import string

# Lemmatization
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Count Vectorizer
audit_cluster_df = audit_df.copy()

# GridSearch
param_grid = {
    'kmeans__n_clusters': [4, 5, 6, 7, 8, 10]
}

# Pipeline
pipeline = Pipeline([
    ('tfid_vect', TfidfVectorizer(tokenizer = LemNormalize, ngram_range=(1,2), min_df=3, max_df=0.5, stop_words='english')),
    ('kmeans', KMeans(random_state=0, max_iter = 1000))
])

# Run GridSearch
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid_search.fit(audit_topic_df['contents'])

print(f"Best Parameters: {grid_search.best_params_}")

# Best model
best_model = grid_search.best_estimator_

# Cluster Create
cluster_labels = best_model.named_steps['kmeans'].labels_
cluster_centers = best_model.named_steps['kmeans'].cluster_centers_

# Add the cluster labels to the original DataFrame
audit_cluster_df['cluster'] = cluster_labels

# Function to print clusters
def print_cluster(df):
    num_clusters = len(df['cluster'].unique())
    for i in range(num_clusters):
        print(f"\nCluster {i}:")
        print(df[df['cluster'] == i].sort_values(by='contents'))

# Call the function to print the clusters
print_cluster(audit_cluster_df)

def get_cluster_details(cluster_model, cluster_data, feature_names, clusters_num, top_n_features=10):
    cluster_details = {}

    # Get cluster centroids
    centroid_feature_distribs = cluster_model.cluster_centers_

    # Iterate through each cluster
    for cluster_num in range(clusters_num):
        # Get top features for each cluster
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster'] = cluster_num

        # Get top features for each cluster (sorting by feature importance)
        top_feature_indexes = centroid_feature_distribs[cluster_num].argsort()[::-1][:top_n_features]
        top_features = [feature_names[index] for index in top_feature_indexes]
        top_features_values = centroid_feature_distribs[cluster_num][top_feature_indexes].tolist()

        # Add top features to the cluster details
        cluster_details[cluster_num]['top_features'] = {feature: value for feature, value in zip(top_features, top_features_values)}

        # Get filenames for the current cluster
        filenames = cluster_data[cluster_data['cluster'] == cluster_num]['File_name']
        filenames = filenames.tolist()
        cluster_details[cluster_num]['filenames'] = filenames

    return cluster_details

def get_print_cluster_details(cluster_details):
    for cluster_num, cluster_detail in cluster_details.items():
        print(f'Cluster {cluster_num}')
        print('Top features:', cluster_detail['top_features'])
        print('File names:', cluster_detail['filenames'])
        print('============================================')

# Get feature names from the best model's vectorizer
feature_names = best_model.named_steps['tfid_vect'].get_feature_names_out()

# Assuming 'best_model' is the best pipeline you have fitted (it should have KMeans as 'kmeans' step)
cluster_details = get_cluster_details(cluster_model = best_model.named_steps['kmeans'],
                                      cluster_data = audit_cluster_df,
                                      feature_names=feature_names,
                                      clusters_num=10, top_n_features=10)

get_print_cluster_details(cluster_details)

feature_names = best_model.named_steps['tfid_vect'].get_feature_names_out()

cluster_details = get_cluster_details(cluster_model = best_model.named_steps['kmeans'],
                                      cluster_data = audit_cluster_df,
                                      feature_names=feature_names,
                                      clusters_num=10, top_n_features=10)

get_print_cluster_details(cluster_details)
'''Cluster 0
Top features: {'share': 0.23831692455095943, 'equity': 0.10710445524927696, 'investment': 0.08241843655991818, 'equity share': 0.07790966067244069, 'r': 0.05791572905043337, 'reserve': 0.05151763486698436, 'capital': 0.04674460180508211, 'number': 0.043562381132349705, 'issued': 0.04094206254057946, 'value': 0.03441377789626003}
File names: ['18612942_table_19', '18793895_table_123', '19005595_table_205', '19058356_table_80', '18630219_table_58', '19073396_table_339', '19005595_table_127', '19058354_table_70', '19034589_table_64', '18612942_table_33', '18731570_table_48', '18950920_table_191', '18731570_table_72', '18950920_table_192', '18630219_table_36', '18699662_table_32', '18699662_table_34', '18599651_table_126', '19005595_table_168', '19034589_table_48', '19005595_table_206', '19070958_table_99', '18612942_table_15', '18630222_table_152', '18646845_table_74', '18599651_table_125', '18646845_table_111', '18699662_table_35', '18630222_table_89', '18612942_table_67', '18646845_table_76', '19058356_table_72', '18646837_table_134', '18612942_table_20', '18646837_table_97', '18630219_table_35', '18950920_table_148', '19070958_table_180', '18731570_table_49', '18989758_table_80', '18661964_table_40', '18599651_table_188', '19070958_table_188', '19058356_table_74', '18731570_table_55', '18599651_table_123', '18646845_table_77', '19073312_table_202', '19058356_table_93', '18646837_table_81', '19070958_table_117', '18888243_table_135', '19005595_table_110', '18599651_table_168', '18950920_table_163', '18612942_table_18', '18599651_table_112', '18731570_table_73', '18661965_table_40', '18630222_table_195', '18699662_table_41', '19070958_table_193', '18646845_table_128', '18612942_table_34', '18599651_table_124', '18599651_table_137', '19119447_table_78', '18950920_table_110', '19070958_table_107', '19005595_table_109', '18630219_table_77', '18793895_table_214', '18630222_table_153', '19005595_table_215', '18599651_table_176', '19058354_table_38', '18612942_table_13', '18630222_table_154']
============================================
Cluster 1
Top features: {'asset': 0.18129588280636533, 'block': 0.10033748830694604, 'carrying': 0.09861530276926304, 'depreciation': 0.09741382574753475, 'net': 0.08538393683670722, 'value': 0.08224831654991664, 'intangible': 0.0792108157186407, 'intangible asset': 0.07749730872099567, 'gross': 0.07325366318071931, 'equipment': 0.06818574273547252}
File names: ['18779279_table_68', '18793895_table_121', '19058356_table_71', '19070958_table_174', '18793895_table_204', '18646837_table_73', '18731570_table_53', '18950920_table_64', '18630222_table_140', '18612942_table_45', '18888243_table_152', '18612942_table_44', '18630222_table_80', '18612942_table_43', '18599651_table_166', '19083995_table_54', '19073312_table_62', '18630222_table_144', '19083995_table_55', '19070958_table_176', '19073312_table_167', '18989758_table_71', '18646837_table_71', '19073396_table_221', '18630222_table_143', '18599651_table_108', '19070958_table_92', '19034589_table_54', '18731570_table_52', '19058355_table_142', '18950920_table_186', '18630222_table_79', '19083995_table_139', '18630222_table_77', '18888243_table_100', '18630219_table_38', '18630222_table_76', '19034589_table_47', '18646837_table_72', '18630219_table_71', '19034589_table_45', '18646845_table_57', '18630222_table_78', '18630219_table_37', '18950920_table_146', '18779279_table_178', '19034589_table_74', '18853952_table_39', '19070958_table_88', '18630222_table_139', '18661963_table_27', '18646845_table_58', '18989756_table_36', '18699662_table_40', '18599651_table_165', '18599651_table_110', '19073396_table_338', '18599651_table_111', '18630219_table_70']
============================================
Cluster 2
Top features: {'loan': 0.2234012572124791, 'borrowing': 0.08940811544095538, 'advance': 0.07763029337772957, 'bank': 0.07590672648845653, 'loan advance': 0.07350303832992564, 'unsecured': 0.06239922027493942, 'secured': 0.06030342288840699, 'r': 0.059166270391719164, 'march': 0.0570409360242598, 'march 31': 0.056278518199595645}
File names: ['19058354_table_48', '18599651_table_172', '18928156_table_95', '18989758_table_198', '18928156_table_94', '19058354_table_44', '18600361_table_49', '19058356_table_200', '19058354_table_49', '18888243_table_75', '18630219_table_76', '19070958_table_194', '19070958_table_108', '18888243_table_84', '19034589_table_55', '18612942_table_39', '18646837_table_76', '18661963_table_30', '19058356_table_98', '18945373_table_30', '18888243_table_136', '18888243_table_131', '18661965_table_42', '19034589_table_60', '18699662_table_38', '19058355_table_103', '18600361_table_41', '18661965_table_41', '18888243_table_140', '18612942_table_52', '18612942_table_35', '19058355_table_157', '18888243_table_127', '18661963_table_24', '18945375_table_95', '18945375_table_17', '18793895_table_130', '18600361_table_45', '18646845_table_80', '18612942_table_62', '18612942_table_47', '18888243_table_79', '19058354_table_47', '18888243_table_88', '18837500_table_36', '18630222_table_155', '18600361_table_39', '19058356_table_255', '18630222_table_81', '18661964_table_42', '18612942_table_41', '18646837_table_79', '18950920_table_127', '18699662_table_42', '18950920_table_176', '18630219_table_75', '19058355_table_102']
============================================
Cluster 3
Top features: {'inr': 0.3702705609185928, 'lac': 0.3340952940831808, 'lac 2016': 0.09465278012650487, 'lac 2017': 0.07877114179088528, '2016 inr': 0.07244471573746147, 'inr lac': 0.07098606499434906, '2016': 0.06425949879702468, '31 december': 0.056195674307839685, 'december': 0.0496682787620701, 'december 2017': 0.04595029343190162}
File names: ['18646845_table_69', '18646845_table_95', '18646845_table_97', '18646845_table_78', '18646845_table_100', '18646845_table_84', '18646845_table_73', '18731570_table_87', '18646845_table_79', '18646845_table_93', '18646845_table_99', '18646845_table_59', '18646845_table_72', '18646845_table_112', '18612942_table_65', '18646845_table_89', '18646845_table_70', '18646845_table_71', '18646845_table_96', '18793895_table_184', '18646845_table_88', '18646845_table_82', '18646845_table_60', '18646845_table_98', '18646845_table_85', '18646845_table_101', '18646845_table_92', '19070958_table_187', '18646845_table_66', '18646845_table_90', '18646845_table_65']
============================================
Cluster 4
Top features: {'provision': 0.1766687455071461, 'benefit': 0.10255961168799803, 'employee': 0.09527362077903945, 'employee benefit': 0.08746381989990568, 'gratuity': 0.051294011707780356, 'expense': 0.0498260124837707, 'plan': 0.049050324825553264, '2016': 0.04776066531596273, 'increased': 0.04297365196209042, 'provision employee': 0.0418038613006173}
File names: ['18646845_table_63', '18888243_table_94', '18793893_table_61', '19058354_table_50', '18950920_table_184', '18731570_table_51', '18646845_table_87', '18630222_table_96', '19034589_table_62', '18950920_table_79', '18630222_table_165', '18731570_table_57', '18630222_table_183', '18612942_table_42', '18989758_table_125', '18646837_table_90', '18888243_table_134', '18630219_table_49', '18646845_table_83', '18630222_table_82', '18950920_table_102', '19070958_table_197', '18950920_table_173', '18661964_table_41', '18793895_table_349', '18600361_table_44', '18600361_table_60', '18630222_table_150', '18661963_table_26', '18599651_table_170', '18612942_table_58', '18600361_table_40', '18731570_table_102', '18731570_table_101', '18630222_table_97', '18600361_table_56', '18599651_table_193', '18630222_table_120', '18888243_table_147', '19070958_table_118', '18888243_table_130', '19058354_table_65', '18612942_table_38', '18630222_table_87', '19058354_table_46', '18837500_table_41', '18630219_table_73', '18599651_table_136', '18646845_table_64', '18646837_table_108', '18630222_table_158', '18630222_table_148', '19058356_table_205', '18646837_table_110', '19070958_table_112', '18793895_table_350', '18646837_table_109']
============================================
Cluster 5
Top features: {'million': 0.24672885346772672, 'payable': 0.12776118219264623, 'trade': 0.08360419555759113, 'liability': 0.08348147220511273, 'trade payable': 0.07704435486515515, 'million 2016': 0.06080791923318043, '2016': 0.05721955966912491, 'million 2017': 0.0497916581954686, 'financial liability': 0.04187319776534773, 'increase': 0.038063311564641746}
File names: ['18630222_table_86', '18599651_table_153', '19034589_table_61', '18630222_table_156', '19070958_table_111', '18599651_table_113', '19083995_table_156', '18646845_table_81', '18646837_table_77', '18599651_table_187', '18646845_table_126', '19083995_table_73', '18599651_table_179', '18599651_table_148', '18599651_table_133', '18646845_table_86', '19070958_table_195', '18646837_table_86', '18630222_table_92', '19034589_table_46', '18888243_table_129', '18793893_table_50', '18599651_table_174', '18599651_table_201', '18630222_table_141', '18699662_table_39', '18630222_table_95', '18950920_table_177', '18599651_table_167', '18599651_table_191', '18612942_table_40', '18888243_table_77', '19005595_table_226', '18646837_table_127', '18599651_table_152', '18600361_table_42', '18646837_table_92', '18599651_table_105', '19034589_table_49', '18646837_table_128', '18646845_table_122', '18599651_table_146', '19034589_table_72', '18599651_table_178', '18630222_table_99', '18599651_table_181', '19070958_table_196', '18599651_table_173', '19058356_table_100', '18599651_table_147', '18646837_table_82', '18599651_table_177', '18599651_table_139', '18950920_table_95', '19005595_table_135', '19083995_table_69', '18888243_table_80', '19058355_table_158', '18646837_table_129', '18630222_table_149', '19073312_table_210', '18599651_table_151', '18646837_table_84']
============================================
Cluster 6
Top features: {'cash': 0.09468222305442048, '2016': 0.07689019747627437, 'balance': 0.06354238791223231, 'december': 0.05324248186583563, 'bank': 0.05149234565619914, 'december 31': 0.05149225433634301, 'receivables': 0.04779920406146749, 'usd': 0.04407779335553296, 'asset': 0.04400990538843128, 'eur': 0.03818276179951314}
File names: ['18630222_table_146', '19058354_table_57', '18630222_table_145', '18646845_table_67', '18599651_table_142', '18599651_table_127', '18646837_table_120', '18646837_table_75', '18599651_table_192', '18599651_table_114', '19070958_table_225', '19119447_table_74', '18630222_table_83', '18646837_table_125', '18779279_table_186', '18646845_table_119', '18612942_table_50', '18793895_table_210', '18600361_table_64', '19058356_table_85', '18630222_table_94', '19070958_table_134', '18630222_table_121', '18599651_table_143', '18599651_table_144', '18989758_table_86', '18793893_table_64', '18793895_table_240', '18646837_table_132', '18793895_table_147', '18646845_table_120', '18646837_table_83', '18793895_table_241', '18646845_table_121', '18646837_table_78', '18599651_table_122', '18646845_table_118', '18599651_table_140', '18630222_table_192', '18630222_table_84', '18793895_table_148', '18599651_table_120', '18646837_table_130', '18599651_table_121', '18793895_table_209', '18599651_table_185', '19034589_table_53', '18599651_table_183', '18599651_table_204', '19058356_table_84', '18599651_table_180', '18646837_table_119', '18950920_table_161', '18646845_table_123', '18793893_table_72', '18599651_table_184', '18646837_table_94', '18612942_table_51', '18646837_table_87', '18630222_table_147', '19070958_table_186', '18599651_table_141', '18630222_table_85', '18646837_table_131', '18646837_table_74', '19058354_table_56', '18599651_table_171', '18888243_table_139', '18630222_table_157', '18699662_table_43', '18793895_table_270', '18630222_table_138', '18731570_table_85', '19119447_table_73', '18599651_table_182', '18779279_table_79', '18731571_table_64', '19034589_table_58', '18599651_table_145', '18888243_table_133', '18630219_table_41', '19034589_table_51', '18646837_table_111', '18888243_table_87', '18630222_table_187', '18950920_table_160', '18646845_table_124', '18699662_table_33', '18646837_table_124', '18646845_table_75', '18600361_table_48', '18793895_table_242', '18950920_table_81', '18888243_table_86', '18599651_table_194', '18599651_table_109', '18612942_table_68', '18646837_table_126', '18600361_table_47', '18630222_table_159', '18646837_table_123']
============================================
Cluster 7
Top features: {'crore': 0.43511963011912724, 'r': 0.10216279280475202, '2018': 0.06685897631205415, 'crore 2018': 0.06493372670655281, 'crore 2017': 0.061218938938911546, 'liability': 0.05127278679175275, '31 march': 0.05013246361002612, 'current': 0.04488924531629814, 'march': 0.04024900307009527, 'increase': 0.03830861730404811}
File names: ['19005595_table_112', '19005595_table_241', '19005595_table_182', '18989758_table_222', '18989758_table_223', '19005595_table_223', '18989758_table_95', '19005595_table_298', '18989758_table_216', '18989758_table_201', '18989758_table_213', '19073312_table_173', '18989758_table_87', '18989758_table_193', '19005595_table_130', '18989758_table_262', '18731571_table_78', '18989758_table_101', '19058354_table_58', '19005595_table_237', '18989758_table_239', '18989758_table_145', '18989758_table_84', '18989758_table_144', '18989758_table_247', '19073312_table_162', '18989758_table_263', '18989758_table_103', '18989758_table_194', '19005595_table_207', '19005595_table_214', '19005595_table_242']
============================================
Cluster 8
Top features: {'useful life': 0.21861334696323037, 'useful': 0.21861334696323037, 'life': 0.2095173490546324, 'rate': 0.12596384848236916, 'depreciated': 0.10766719620065174, 'equipment': 0.07251438927801507, 'discount rate': 0.06103389791909727, 'discount': 0.055622485840450674, 'management': 0.05546747913155073, 'building': 0.05451614706926469}
File names: ['19070958_table_178', '19058356_table_65', '18731571_table_58', '19070958_table_173', '18779279_table_176', '18793895_table_120', '19058356_table_177', '19070958_table_172', '18950920_table_202', '18646845_table_55', '19073396_table_336', '18646845_table_56', '18950920_table_121', '19005595_table_102', '19073396_table_219', '19073312_table_77', '18793893_table_43', '19070958_table_93', '18599651_table_200']
============================================
Cluster 9
Top features: {'expense': 0.06887431995046514, 'income': 0.06152679168932846, 'tax': 0.04559881487576664, 'lakh': 0.044960819479066085, '2018': 0.044195610173471414, 'wa': 0.0424424338779625, 'march': 0.03901736562225068, 'r': 0.03879279943532428, 'decrease': 0.03740251912467002, 'previous': 0.035949713791513106}
File names: ['19034589_table_63', '18646837_table_93', '18793895_table_137', '18600361_table_59', '18793895_table_272', '18950920_table_188', '18612942_table_57', '18950920_table_38', '18600361_table_57', '18989758_table_127', '19058356_table_207', '19058356_table_88', '19070958_table_128', '18793893_table_92', '19005595_table_138', '18599651_table_118', '18646845_table_94', '18779279_table_148', '18630222_table_163', '18630222_table_116', '18699662_table_45', '19034589_table_50', '19058354_table_54', '19070958_table_130', '18600361_table_58', '19058356_table_190', '19058354_table_68', '18731570_table_86', '18888243_table_137', '18989758_table_85', '18599651_table_150', '18612942_table_54', '18612942_table_36', '19058356_table_97', '18699662_table_44', '18793893_table_57', '19073312_table_58', '18630219_table_46', '18989758_table_245', '18950920_table_83', '18793893_table_66', '18699662_table_50', '18950920_table_101', '19119447_table_72', '18630222_table_88', '19083995_table_158', '19034589_table_68', '18888243_table_128', '19070958_table_115', '18600361_table_63', '19034589_table_69', '18699662_table_46', '18699662_table_48', '18699662_table_27', '18731570_table_50', '18612942_table_64', '18731570_table_71', '19058356_table_110', '18950920_table_187', '18888243_table_145', '18600361_table_52', '18612942_table_46', '18888243_table_76', '18731570_table_83', '18950920_table_82', '19058355_table_155', '18599651_table_107', '18630222_table_151', '19058356_table_106', '19058354_table_62', '18793895_table_182', '18630222_table_161', '18699662_table_49', '19070958_table_98', '19058354_table_61', '18950920_table_78', '18661963_table_31', '18600361_table_50', '18888243_table_95', '19073396_table_266', '18888243_table_92', '19070958_table_199', '19005595_table_236', '19058356_table_109', '18646837_table_95', '18600361_table_51', '18888243_table_149', '18793895_table_212', '18950920_table_170', '18661964_table_43', '18888243_table_138', '18612942_table_60', '18779279_table_263', '18612942_table_48', '18661963_table_28', '18699662_table_37', '18630222_table_160', '19070958_table_200', '19070958_table_183', '18646845_table_62', '19070958_table_95', '19058356_table_191', '19058354_table_60', '19034589_table_52', '19058356_table_79', '18599651_table_128', '19070958_table_146', '18950920_table_180', '18612942_table_59', '18599651_table_115', '18888243_table_96', '18599651_table_134', '19034589_table_77', '19070958_table_221', '19058354_table_66', '18731570_table_88', '18600361_table_55', '19034589_table_57', '18646845_table_61', '18950920_table_183', '19083995_table_162', '19119447_table_70', '18599651_table_119', '18950920_table_94', '19058355_table_161', '18630222_table_93', '18600361_table_53', '18950920_table_104', '18779279_table_218', '18661963_table_35', '18646837_table_96', '18950920_table_181', '18646845_table_117', '18612942_table_55', '18612942_table_63', '18888243_table_141', '18888243_table_85', '19070958_table_116', '18612942_table_53', '19058356_table_111', '19070958_table_97', '18950920_table_109', '19058356_table_214', '18646837_table_88', '18837500_table_45', '18599651_table_149', '18950920_table_172', '18731570_table_59', '19058356_table_203', '19119447_table_75', '18612942_table_61', '18612942_table_56', '19083995_table_57', '19083995_table_62', '18888243_table_148', '18661963_table_25', '19070958_table_102', '19034589_table_59', '19058356_table_78', '19119447_table_76', '19070958_table_114', '18600361_table_46', '18888243_table_143', '18661963_table_32', '18779279_table_118', '18888243_table_93', '19070958_table_132', '18950920_table_103', '18888243_table_81', '18950920_table_185', '18646837_table_89', '18950920_table_162', '18888243_table_98', '18888243_table_144', '19070958_table_135', '18646845_table_91', '19070958_table_177', '18646837_table_80', '19058354_table_67', '18612942_table_49', '18950920_table_100', '19083995_table_142', '19058354_table_59', '18612942_table_37', '19005595_table_211', '18630219_table_72', '18731570_table_70', '18950920_table_182', '18661963_table_29', '19058356_table_189', '18600361_table_54', '18646837_table_91', '18599651_table_169', '18661963_table_23', '18888243_table_91', '18646845_table_103', '18661965_table_43', '18630219_table_44', '18699662_table_47', '18646837_table_85', '19070958_table_182', '18600361_table_43', '18630222_table_162', '19034589_table_84', '18630219_table_45']
============================================'''