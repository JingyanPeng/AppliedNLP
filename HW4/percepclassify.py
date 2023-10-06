# CSCI544 Coding Assignment 4
# Author: Jingyan Peng
# date 10/24/2022
import sys
import re
import json

percepmodel = open(sys.argv[1])
model = json.loads(percepmodel.read())
percepmodel.close()

test_file = sys.argv[2]
test_data = open(test_file, encoding = 'UTF-8')
lines = [line.rstrip('\n') for line in test_data]
test_data.close()

def clean_preprocessing(text_line):
    text_line = text_line.lower()   
    #contraction
    cont_dic = {"can't": "can not", "cannot": "can not", "won't": "will not", "wanna": "want to", "gonna": "going to", "gotta": "got to", "'cause": "because","let's": "let us", "n't": " not", "'re": " are", "'m": " am", "'ll": " will", "'ve": " have", "'d": " would", "'s been": " has been", "'s": " is"}
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
        if word !='':# and word not in stopwords:
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
    #cleaning & preprocessing comment text
    text_line = ' '.join(item[1:])
    sample['text_feature'] = clean_preprocessing(text_line)
    data.append(sample)  


fhandle = open('percepoutput.txt', 'w')
for i in range(len(data)):
    feature = data[i]['text_feature']
    truthfulness_activation = model['truthfulness']['**bias']
    positivity_activation = model['positivity']['**bias']

    for word in feature:
        if word in model['truthfulness']:
            truthfulness_activation += model['truthfulness'][word] * feature[word]
            positivity_activation += model['positivity'][word] * feature[word]

    output_line = data[i]['id']
    if truthfulness_activation >= 0:
        output_line += ' True'
    else:
        output_line += ' Fake'
    if positivity_activation >= 0:
        output_line += ' Pos'
    else:
        output_line += ' Neg'
    if i != len(data) - 1:
        output_line += '\n'
    fhandle.write(output_line)
fhandle.close()
    