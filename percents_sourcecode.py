import pandas as pd
import numpy as np
import re
import glob

# Create List of Documents
PATH = "C:\\Users\\Pranav\\Documents\\Northwestern\\Junior\\Winter\\IEMS_308\\Kaza_Text_Analytics\\text_data\\*.txt"
files = glob.glob(PATH)
doc_list = list()
for name in files:
    with open(name, 'r',errors = "ignore",encoding = "utf-8") as test_data:
        data=test_data.read().replace('\n', '')
    doc_list.append(data)

data_string = ""
for document in doc_list:
    data_string = data_string + " " + document

# Read in Percentage Training Data
PATH = "C:\\Users\\Pranav\\Documents\\Northwestern\\Junior\\Winter\\IEMS_308\\Kaza_Text_Analytics\\percentage.csv"
pct = pd.read_csv(PATH,encoding = 'latin1',header = None)

# Use Regular Expressions to Find Percentages
p1 = re.findall('\s[\)\~]?[\+\-]?[0-9]+\.?[0-9]+%[\s\)]?',data_string)
p2 = re.findall('\s[\)\~]?[\+\-]?[0-9]+\.?[0-9]+\spercent[\s\)]?',data_string)
p3 = re.findall('\s[\)\~]?[\+\-]?[0-9]+\.?[0-9]+\spercentile\spoints?[\s\)]?',data_string)
p4 = re.findall('\s[\)\~]?[\+\-]?[0-9]+\.?[0-9]+\spercentage\spoints?[\s\)]?',data_string)
p5 = re.findall('\s[\)\~]?[A-Za-z]+\spercent[\s\)]?',data_string)
p6 = re.findall('\s[\)\~]?[A-Za-z]+\spercentage\spoints?[\s\)]?',data_string)
p7 = re.findall('\s[\)\~]?[A-Za-z]+\spercentile\spoints?[\s\)]?',data_string)
    
# Combine all Lists and Remove Unnecessary Spaces
percents = p1 + p2 + p3 + p4 + p5 + p6
percents_revised = list()
for pct in percents:
    percents_revised.append(pct[1:-1])

# Write to CSV Files
pct_results = pd.DataFrame(np.array(percents_revised).reshape(len(percents_revised),1),columns = ['percent'])
pct_results.to_csv('pct_results.csv')