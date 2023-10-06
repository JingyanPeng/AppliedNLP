# CSCI544 Coding Assignment 3
# author: Jingyan Peng
# date: 10/11/2022

import sys
import json

# read the train file
train_file = sys.argv[1]
train_data = open(train_file, encoding = 'UTF-8')
lines = [line.rstrip('\n') for line in train_data]
words_list_of_line = []
for i in range(len(lines)):
    words_list_of_line.append(lines[i].split(' '))

# count the number of tag & tag->word & each word to build vocabulary
count_tag = {}
count_tag_word = {}
vocabulary = {}
count_tag['start_state'] = len(words_list_of_line)
count_tag['end_state'] = len(words_list_of_line)

for line in words_list_of_line:
    for word_tag in line:
        item = word_tag.split('/')
        tag = item[-1]
        # careful with slash character
        word = item[0:-1]  
        word = '/'.join(word)
        # count the number of each word
        if word not in vocabulary:
            vocabulary[word] = 1
        else:
            vocabulary[word] +=1
        # count the number of each tag
        if tag not in count_tag:
            count_tag[tag] = 1
        else:
            count_tag[tag] += 1
        # count the number of each tag->word
        if tag not in count_tag_word:
            count_tag_word[tag] = {}
        if word not in count_tag_word[tag]:
            count_tag_word[tag][word] = 1
        else:
            count_tag_word[tag][word] += 1

# build the vocabulary with training data (to handle the unknown words)
threshold = 2
delete_words = []
count_unkown = 0
for word,count in vocabulary.items():
    if count < threshold:
        count_unkown += count
        delete_words.append(word)
for word in delete_words:
    del vocabulary[word]
vocabulary['<unknown>'] = count_unkown

count_tag_known_word = {} # add a word '<unknown>'
for tag in count_tag_word:
    count_tag_known_word[tag] = {}
    for word in count_tag_word[tag]:
        count_tag_known_word[tag][word] = count_tag_word[tag][word]
        if word not in vocabulary:
            if '<unknown>' not in count_tag_known_word[tag]:
                count_tag_known_word[tag]['<unknown>'] = 1
            else:
                count_tag_known_word[tag]['<unknown>'] += 1

# calculate the transition matrix （with smoothing）
transition_matrix = {}
count_tag_tag = {} #current_tag -> next_tag
for tag in count_tag:
    if tag == 'end_state':
        continue
    else:
        count_tag_tag[tag] = {}
        transition_matrix[tag] = {}

for line in words_list_of_line:
    for i in range(0, len(line)+1):
        if i < len(line):
            current_state = line[i].split('/')[-1]
        if i < len(line)-1:
            next_state = line[i+1].split('/')[-1]
        if i == 0:
            if current_state not in count_tag_tag['start_state']:
                count_tag_tag['start_state'][current_state] = 1
            else:
                count_tag_tag['start_state'][current_state] += 1
        elif i == len(line):
            if 'end_state' not in count_tag_tag[current_state]:
                count_tag_tag[current_state]['end_state'] = 1
            else:
                count_tag_tag[current_state]['end_state'] += 1
        else:
            if next_state not in count_tag_tag[current_state]:
                count_tag_tag[current_state][next_state] = 1
            else:
                count_tag_tag[current_state][next_state] += 1


for current_tag in count_tag_tag:
    for next_tag in count_tag:
        if next_tag == 'start_state' or current_tag == 'end_state':
            continue
        elif next_tag not in count_tag_tag[current_tag]: 
            # there are no this kind of transition in training data ----smoothing
            transition_matrix[current_tag][next_tag] = 1/(count_tag[current_tag] + 4 * len(count_tag))
        else:
            # (without smoothing) transition_matrix[current_tag][next_tag] = count_tag_tag[current_tag][next_tag] / count_tag[current_tag]
            transition_matrix[current_tag][next_tag] = (count_tag_tag[current_tag][next_tag] + 1) / (count_tag[current_tag] + 4 * len(count_tag))

# calculate the emission matrix (without smoothing)
emission_matrix = {}
for tag in count_tag_known_word:
    for word in count_tag_known_word[tag]:
        if word not in emission_matrix:
                emission_matrix[word] = {}
        emission_matrix[word][tag] = count_tag_known_word[tag][word] / count_tag[tag]

# write model in hmmmodel.txt 
# hmmmodel : vocabulary + transition_matrix + emission_matrix
hmmmodel = {}
hmmmodel['vocabulary'] = vocabulary
hmmmodel['emission_matrix'] = emission_matrix
hmmmodel['transition_matrix'] = transition_matrix
f = open('hmmmodel.txt', 'w')
f.write(json.dumps(hmmmodel, indent = 2))
f.close
