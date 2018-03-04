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

# Transform into a String
data_string = ""
for document in doc_list:
    data_string = data_string + " " + document

# Word Tokenize
punc = (",`~{}|:./;'?&-$()[]+_=-:*^\<>#@&")
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

# Generate All Bigrams
data_bigrams = list(nltk.bigrams(data_string.split()))
data_bigrams = [b[0] + " " + b[1] for b in data_bigrams]

# Find all Capitalized Words for Test Set
cap_words = re.findall('[A-Z][A-Za-z]+',data_string)

# Delete Useless and Frequent Words
counts = Counter(cap_words)
commons = counts.most_common(500)

commons = [ii[0] for ii in commons]
common_dict = {}
for ii in range(len(commons)):
    common_dict[commons[ii]] = 1
final_words = list()
for word in cap_words:
    if word not in common_dict.keys():
        final_words.append(word)

# Generate Test Set Bigrams
data_string1 = ' '.join(final_words)
final_bigrams = list(nltk.bigrams(data_string1.split()))
final_bigrams = [b[0] + " " + b[1] for b in final_bigrams]

# Create Dictionary of Companies
PATH = "C:\\Users\\Pranav\\Documents\\Northwestern\\Junior\\Winter\\IEMS_308\\Kaza_Text_Analytics\\companies.csv"
company = pd.read_csv(PATH,encoding = 'latin1',header = None)
company = company.drop_duplicates()
company.columns = ['name']
company = company.reset_index()
del company['index']
company = company['name'].tolist()
company_dict = {}
for ii in range(len(company)):
    company_dict[company[ii]] = 1

# Find all Bigrams in Text that are Company Names
matching = list()
for bigram in data_bigrams:
    if bigram in company_dict.keys():
        matching.append(bigram)

# Create Data Frame so that a Classification Model can be Run
df_company = pd.DataFrame(np.array(matching).reshape(len(matching),1),columns = ["name"])
df_company = df_company.drop_duplicates()
df_company = df_company.set_index('name')
df_company['company'] = 1

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
df_pol['company'] = 0

# Collect Random Bigrams that are not Company names as Additional Negative Samples
samp = random.sample(data_bigrams,len(df_company) - len(df_pol) - 100)
negatives = list()
for ii in range(0,len(samp)-1):
    if (samp[ii] not in company_dict.keys()) and (samp[ii] not in pol_dict.keys()):
        negatives.append(samp[ii])
negatives = pd.DataFrame(np.array(negatives).reshape(len(negatives),1),columns = ['name'])
negatives = negatives.set_index('name')
negatives['company'] = 0

# Combine Positive and Negative Samples into One Dataframe
df_company = pd.concat([df_company,df_pol,negatives])
df_company = df_company.reset_index()
df_company.columns = [['name','isCompany']]

# Develop Training Feature 1: Determine if Relevant Words are Nearby
df_company['word_nearby'] = 0
for ii in range(0,len(df_company)-1):
    if ('Inc' in data_string[re.search(df_company['name'][ii],data_string).start():re.search(df_company['name'][ii],data_string).end()+10]):
        df_company.loc[ii,'word_nearby'] = 1
    elif ('Ltd' in data_string[re.search(df_company['name'][ii],data_string).start():re.search(df_company['name'][ii],data_string).end()+10]):
        df_company.loc[ii,'word_nearby'] = 1
    elif ('Co' in data_string[re.search(df_company['name'][ii],data_string).start():re.search(df_company['name'][ii],data_string).end()+10]):
        df_company.loc[ii,'word_nearby'] = 1
    elif ('Group' in data_string[re.search(df_company['name'][ii],data_string).start():re.search(df_company['name'][ii],data_string).end()+10]):
        df_company.loc[ii,'word_nearby'] = 1
        

company_xtest = list()
for ii in range(0,len(final_words)-1):
    if ('Inc' in data_string1[re.search(final_bigrams[ii],data_string1).start():re.search(final_bigrams[ii],data_string1).end()+10]):
        company_xtest = company_xtest + [1]
    elif ('Ltd' in data_string1[re.search(final_bigrams[ii],data_string1).start():re.search(final_bigrams[ii],data_string1).end()+10]):
        company_xtest = company_xtest + [1]
    elif ('Co' in data_string1[re.search(final_bigrams[ii],data_string1).start():re.search(final_bigrams[ii],data_string1).end()+10]):
        company_xtest = company_xtest + [1]
    elif ('Group' in data_string1[re.search(final_bigrams[ii],data_string1).start():re.search(final_bigrams[ii],data_string1).end()+10]):
        company_xtest = company_xtest + [1]
    else:
            company_xtest = company_xtest + [0]
    
# Fit logistic Regression Model
company_xtrain = df_company['word_nearby'].values.reshape(-1,1)
company_ytrain = df_company['isCompany'].values
LogReg = LogisticRegression()
LogReg.fit(company_xtrain,company_ytrain)
company_xtest = pd.DataFrame(np.array(company_xtest).reshape(len(company_xtest),1),columns = ['word_nearby'])
company_logit = sm.Logit(company_ytrain,company_xtrain)
result = company_logit.fit()
print(result.summary())
company_ytest = LogReg.predict(company_xtest.values.reshape(-1,1))


