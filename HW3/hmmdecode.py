# CSCI544 Coding Assignment 3
# author: Jingyan Peng
# date: 10/11/2022

import sys
import json

#get the model parameters as json
hmmmodel = open('hmmmodel.txt', 'r', encoding = 'UTF-8')
model = json.loads(hmmmodel.read())
hmmmodel.close()
transition_matrix = model['transition_matrix']
emission_matrix = model['emission_matrix']

dev_raw_file = sys.argv[1]
test_data = open(dev_raw_file, encoding = 'UTF-8').read()
sentences_list = test_data.split('\n')
if sentences_list[-1] == '':
    sentences_list = sentences_list[0:-1]

f = open('hmmoutput.txt', 'w', encoding = 'UTF-8')

# Viterbi Decoding Algorithm
for sentence in sentences_list:
    words_of_sentence = sentence.split(' ')
    viterbi_matrix = [{}]

    # the first word of a sentence
    if words_of_sentence[0] in emission_matrix:
        states = emission_matrix[words_of_sentence[0]]
    else:
        states = emission_matrix['<unknown>']
    for tag in states:
        # get e
        if words_of_sentence[0] in emission_matrix:
            e = emission_matrix[words_of_sentence[0]][tag] 
        else:
            e = emission_matrix['<unknown>'][tag]  # solve the unknown word problem
        viterbi_matrix[0][tag] = {}
        viterbi_matrix[0][tag]['tag'] = 'start_state' # from which tag to current_tag
        # calculate  e * t
        viterbi_matrix[0][tag]['prob'] = e * transition_matrix['start_state'][tag]

    ## not the first word of a sentence
    for i in range(1, len(words_of_sentence) + 1):
        ### not the last word of a sentence
        if i != len(words_of_sentence):
            viterbi_matrix.append({})
            if words_of_sentence[i] in emission_matrix:
                states = emission_matrix[words_of_sentence[i]]
            else:
                states = emission_matrix['<unknown>']
            for tag in states:
                # get e
                if words_of_sentence[i] in emission_matrix:
                    e = emission_matrix[words_of_sentence[i]][tag]
                else:
                    e = emission_matrix['<unknown>'][tag]
                viterbi_matrix[i][tag] = {}
                viterbi_matrix[i][tag]['prob'] = 0
                viterbi_matrix[i][tag]['tag'] = ''
                for previous_tag in viterbi_matrix[i-1]:
                # calculate every e * t and store the max with its tag
                    pre = viterbi_matrix[i-1][previous_tag]['prob'] * transition_matrix[previous_tag][tag] * e
                    if (pre > viterbi_matrix[i][tag]['prob']):
                        viterbi_matrix[i][tag]['prob'] = pre
                        viterbi_matrix[i][tag]['tag'] = previous_tag
        ### the last word of a sentence
        else:
            lastword = viterbi_matrix[-1]
            states = lastword.keys()
            viterbi_matrix.append({})
            viterbi_matrix[-1]['end_state'] = {}
            viterbi_matrix[-1]['end_state']['prob'] = 0
            viterbi_matrix[-1]['end_state']['tag'] = ''
            for tag in states:
            # calculate  e * t and store the max with its tag
                pre = viterbi_matrix[-2][tag]['prob'] * transition_matrix[tag]['end_state']
                if (pre > viterbi_matrix[-1]['end_state']['prob']):
                    viterbi_matrix[-1]['end_state']['prob'] = pre
                    viterbi_matrix[-1]['end_state']['tag'] = tag
 
    # reach the end of the sentence
    # then backword to get the tagging sequence
    current_idx = len(words_of_sentence)
    current_state = 'end_state'
    seq = ""
    x = current_idx - 1
    while x >= 0:
        seq = words_of_sentence[x] + "/" + viterbi_matrix[current_idx][current_state]['tag'] + ' ' + seq
        current_state = viterbi_matrix[current_idx][current_state]['tag']
        current_idx -= 1
        x -= 1
    #print(right_seq)
    if sentence !='' and sentence != sentences_list[-1]:
        f.write(seq + '\n')
    elif sentence != '':
        f.write(seq)
    
f.close()