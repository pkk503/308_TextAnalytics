import nltk
# nltk.download()

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import re
import glob
from collections import Counter

# Create List of Documents
PATH = "C:\\Users\\Pranav\\Documents\\Northwestern\\Junior\\Winter\\IEMS_308\\Kaza_Text_Analytics\\text_data\\*.txt"
files = glob.glob(PATH)
doc_list = list()
for name in files:
    with open(name, 'r',errors = "ignore",encoding = "utf-8") as test_data:
        data=test_data.read().replace('\n', '')
    doc_list.append(data)

# Explore Data by Determining Context of CEO name usage (separate by document)
# elon = "Elon Musk"
# elon_documents = list()
# for document in doc_list:
#    if elon in document:
#        elon_documents.append(document)

# apu = "Apu Gupta"
# apu_documents = list()
# for document in doc_list:
#    if apu in document:
#        apu_documents.append(document)

# ron = "Ron Johnson"
# ron_documents = list()
# for document in doc_list:
#    if ron in document:
#        ron_documents.append(document)

# Develop training data set
# random.shuffle(doc_list)
# train_data = doc_list[:365]
# test_data = doc_list[365:]
 
# Transform into a String
data_string = ""
for document in doc_list:
    data_string = data_string + " " + document

# Word Tokenize
punc = (",`~{}|:./;'?&-$()[]!+_=-:*^\<>#@&")
transtable = {ord(c): None for c in punc}
data_nopunc = data_string.translate(transtable)
data_words = word_tokenize(data_nopunc)

# Remove Stop Words
stop_words = set(stopwords.words('english'))
data_words_1 = list()
for word in data_words:
    if word not in stop_words:
        data_words_1.append(word)
data_words = data_words_1.copy()
del data_words_1

# Reform Whole String of Data
data_string = ' '.join(data_words)

# Generate Bigrams
data_bigrams = list(nltk.bigrams(data_string.split()))
data_bigrams = [b[0] + " " + b[1] for b in data_bigrams]

# Find all Capitalized Words for Test Set
cap_words = re.findall('[A-Z][A-Za-z]+',data_string)

# Delete Useless and Frequent Words
counts = Counter(cap_words)
commons = counts.most_common(1000)

commons = [ii[0] for ii in commons]
common_dict = {}
for ii in range(len(commons)):
    common_dict[commons[ii]] = 1
final_words = list()
for word in cap_words:
    if word not in common_dict.keys():
        final_words.append(word)

# Generate Bigrams for Test Set
data_string1 = ' '.join(final_words)
final_bigrams = list(nltk.bigrams(data_string1.split()))
final_bigrams = [b[0] + " " + b[1] for b in final_bigrams]

# Create Dictionary of CEOs
PATH = "C:\\Users\\Pranav\\Documents\\Northwestern\\Junior\\Winter\\IEMS_308\\Kaza_Text_Analytics\\ceo.csv"
ceo = pd.read_csv(PATH,encoding = 'latin1',header = None)
del ceo[2]
ceo = ceo.drop_duplicates()
ceo['Name'] = ceo[0] + " " + ceo[1]
del ceo[0]
del ceo[1]
ceo = ceo.reset_index()
del ceo['index']
ceo = ceo['Name'].tolist()
ceo_dict = {}
for ii in range(len(ceo)):
    ceo_dict[ceo[ii]] = 1

# Find all Bigrams in Text that are CEO Names
matching = list()
for bigram in data_bigrams:
    if bigram in ceo_dict.keys():
        matching.append(bigram)

# Create Data Frame so that a Classification Model can be Run
df_ceo = pd.DataFrame(np.array(matching).reshape(len(matching),1),columns = ["ceo_name"])
df_ceo = df_ceo.drop_duplicates()
df_ceo = df_ceo.set_index('ceo_name')
df_ceo['ceo'] = 1

# Read in U.S. Politician Names Data as Negative Samples
PATH = "C:\\Users\\Pranav\\Documents\\Northwestern\\Junior\\Winter\\IEMS_308\\Kaza_Text_Analytics\\politicians.csv"

df_pol = pd.read_csv(PATH,header = None)
df_pol['Name'] = df_pol[0] + " " + df_pol[1]
del df_pol[0]
del df_pol[1]
df_pol = df_pol.reset_index()
del df_pol['index']
df_pol = df_pol['Name'].tolist()
df_pol = [x for x in df_pol if str(x) != 'nan']
pol_dict = {}
for ii in range(len(df_pol)):
    pol_dict[df_pol[ii]] = 1

# Match all Politician Names from Corpus
matching = list()
for bigram in data_bigrams:
    if bigram in pol_dict.keys():
        matching.append(bigram)
df_pol = pd.DataFrame(np.array(matching).reshape(len(matching),1),columns = ["pol_name"])
df_pol = df_pol.drop_duplicates()
df_pol = df_pol.set_index('pol_name')
df_pol['ceo'] = 0

# Collect Random Bigrams that are not CEO names as Additional Negative Samples
samp = random.sample(data_bigrams,len(df_ceo) - len(df_pol) - 100)
negatives = list()
for ii in range(0,len(samp)-1):
    if (samp[ii] not in ceo_dict.keys()) and (samp[ii] not in pol_dict.keys()):
        negatives.append(samp[ii])
negatives = pd.DataFrame(np.array(negatives).reshape(len(negatives),1),columns = ['name'])
negatives = negatives.set_index('name')
negatives['ceo'] = 0

# Combine Positive and Negative Samples into One Dataframe
df_ceo = pd.concat([df_ceo,df_pol,negatives])
df_ceo = df_ceo.reset_index()
df_ceo.columns = [['name','isCEO']]

# Develop Training Feature: Determine if the word "CEO" is nearby
df_ceo['word_nearby'] = 0
for ii in range(0,len(df_ceo)-1):
    if ('CEO' in data_string[re.search(df_ceo['name'][ii],data_string).start()-30:re.search(df_ceo['name'][ii],data_string).end()+30]):
        df_ceo.loc[ii,'word_nearby'] = 1

# Apply Feature to Test Set
ceo_xtest = list()
for ii in range(0,len(final_bigrams)-1):
    if ('CEO' in data_string1[re.search(final_bigrams[ii],data_string1).start()-30:re.search(final_bigrams[ii],data_string1).end()+30]):
        ceo_xtest = ceo_xtest + [1]
    else:
        ceo_xtest = ceo_xtest + [0]

# Fit logistic Regression Model
ceo_xtrain = df_ceo['word_nearby'].values.reshape(-1,1)
ceo_ytrain = df_ceo['isCEO'].values
LogReg = LogisticRegression()
LogReg.fit(ceo_xtrain,ceo_ytrain)
ceo_xtest = pd.DataFrame(np.array(ceo_xtest).reshape(len(ceo_xtest),1),columns = ['word_nearby'])
ceo_logit = sm.Logit(ceo_ytrain,ceo_xtrain)
result = ceo_logit.fit()
print(result.summary())
ceo_ytest = LogReg.predict(ceo_xtest.values.reshape(-1,1))

# Apply Model to Test Set
df_ytest = pd.DataFrame(np.array(ceo_ytest).reshape(len(ceo_ytest),1),columns = ['prediction'])
df_final_bigrams = pd.DataFrame(np.array(final_bigrams).reshape(len(final_bigrams),1),columns = ['bigram'])
ceo_results = pd.concat([df_ytest,df_final_bigrams],axis = 1)
ceo_results = ceo_results[ceo_results['prediction'] == 1]
del ceo_results['prediction']
ceo_results.to_csv('ceo_results.csv')
