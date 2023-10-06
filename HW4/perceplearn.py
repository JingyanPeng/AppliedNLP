# CSCI544 Coding Assignment 4
# Author: Jingyan Peng
# date 10/24/2022
import random
import sys
import re
import json

train_file = sys.argv[1]
train_data = open(train_file, encoding = 'UTF-8')
lines = [line.rstrip('\n') for line in train_data]
train_data.close()

def clean_preprocessing(text_line):
    text_line = text_line.lower()   
    # contraction
    cont_dic = {"can't": "can not", "cannot": "can not", "won't": "will not", "wanna": "want to", "gonna": "going to", "gotta": "got to", "'cause": "because", "let's": "let us", "n't": " not", "'re": " are", "'m": " am", "'ll": " will", "'ve": " have", "'d": " would", "'s been": " has been", "'s": " is"}
    for key in cont_dic:
        text_line = text_line.replace(key, cont_dic[key])
    #stopwords from https://www.textfixer.com/tutorials/common-english-words.txt
    #stopwords = set(['1','2','3','4','5','6','7','8','9','0','a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','between','but','by','can','cannot','could','dear','did','do','does','during','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','once','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','through','to','too','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your','yourself'])
    text_line = text_line.replace('\n', ' ')
    text_line = text_line.replace('\t', ' ')
    text_line = re.sub(' +', ' ', text_line)
    text_line = re.sub('[^a-z]+', ' ', text_line)
    text_feature = {}
    for word in text_line.split(' '):
        if word !='': #and word not in stopwords:
            if word in text_feature:
                text_feature[word] += 1
            else:
                text_feature[word] = 1
    return text_feature

data = []
for i in range(len(lines)):
    item = lines[i].split(' ')
    sample = {}
    sample['id'] = item[0]
    sample['truthfulness'] = item[1]
    sample['positivity'] = item[2]
    #cleaning & preprocessing comment text
    text_line = ' '.join(item[3:])
    sample['text_feature'] = clean_preprocessing(text_line)
    data.append(sample)  

labels = {'True':1, 'Fake':-1, 'Pos':1, 'Neg':-1}
vanilla_truthfulness_weights = {}
vanilla_positivity_weights = {}
averaged_truthfulness_weights = {}
averaged_positivity_weights = {}
u_truthfulness_weights = {}
u_positivity_weights = {}

#initialize bias
vanilla_truthfulness_weights['**bias'] = 0
vanilla_positivity_weights['**bias'] = 0
u_truthfulness_weights['**bias'] = 0
u_positivity_weights['**bias'] = 0

c = 1
epochs = 80
for i in range(epochs):
    random.shuffle(data)
    #traverse the training data
    for j in range(len(data)): 
        feature = data[j]['text_feature']
        #initialize weights
        for word in feature:
            if word not in vanilla_truthfulness_weights:
                vanilla_truthfulness_weights[word] = 0
                vanilla_positivity_weights[word] = 0
                u_truthfulness_weights[word] = 0
                u_positivity_weights[word] = 0
        
        vanilla_truthfulness_activation = vanilla_truthfulness_weights['**bias']
        vanilla_positivity_activation = vanilla_positivity_weights['**bias']
        
        for word in feature:
            vanilla_truthfulness_activation += vanilla_truthfulness_weights[word] * feature[word] 
            vanilla_positivity_activation += vanilla_positivity_weights[word] * feature[word]

        if vanilla_truthfulness_activation * labels[data[j]['truthfulness']] <= 0:
            for word in feature:
                vanilla_truthfulness_weights[word] += labels[data[j]['truthfulness']] * feature[word]
                vanilla_truthfulness_weights['**bias'] += labels[data[j]['truthfulness']]
                u_truthfulness_weights[word] += labels[data[j]['truthfulness']] * c * feature[word]
                u_truthfulness_weights['**bias'] += labels[data[j]['truthfulness']] * c
        
        if vanilla_positivity_activation * labels[data[j]['positivity']] <= 0:
            for word in feature:
                vanilla_positivity_weights[word] += labels[data[j]['positivity']] * feature[word]
                vanilla_positivity_weights['**bias'] += labels[data[j]['positivity']]
                u_positivity_weights[word] += labels[data[j]['positivity']] * c * feature[word]
                u_positivity_weights['**bias'] += labels[data[j]['positivity']] * c
        
        c += 1

for w in vanilla_truthfulness_weights:
    averaged_truthfulness_weights[w] = vanilla_truthfulness_weights[w] - u_truthfulness_weights[w] / (c * 1.0)
    averaged_positivity_weights[w] = vanilla_positivity_weights[w] - u_positivity_weights[w] / (c * 1.0)

vanilla = {'truthfulness': vanilla_truthfulness_weights, 'positivity': vanilla_positivity_weights}
fhandle1 = open('vanillamodel.txt', 'w')
fhandle1.write(json.dumps(vanilla, indent = 2))
fhandle1.close()

averaged = {'truthfulness': averaged_truthfulness_weights, 'positivity': averaged_positivity_weights}
fhandle2 = open('averagedmodel.txt','w')
fhandle2.write(json.dumps(averaged, indent = 2))
fhandle2.close()
