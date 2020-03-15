import numpy as np
import codecs
import copy
class Dataloader(object):
    def __init__(self, config):
        self.file_pathtype = config.file_pathtype
        self.smoothing_type = config.smoothing_type
        # self.smoothing_d = smoothing_d
    #'TBD' means the word we can not observe in training set.
    
    # Used in extraction.py
    def Load(self, smoothing_d ,mode = 'train', ):
        self.smoothing_d = smoothing_d
        if mode == 'train':
            f = codecs.open(self.file_pathtype + '_train.txt','r','utf8')
            text = f.readlines()
            f.close()
        else:
            f = codecs.open(self.file_pathtype + '_test.txt', 'r', 'utf8')
            text = f.readlines()
            f.close()
        
        word_states = {}
        states = {}
        transitions = {}
        emissions = {}
        
        for word in text:
            if word.strip().split():
                word_type = word.strip().split()
                word_states[word_type[1]] = word_states.get(word_type[1], 0) + 1
                # Emsission matrix
                if (word_type[1] in emissions):
                    emissions[word_type[1]][word_type[0]] = emissions[word_type[1]].get(word_type[0], 0) + 1
                else:
                    emissions[word_type[1]] = {}
                    emissions[word_type[1]][word_type[0]] = 1
        fea = ['START']
        for word in text:
            if word.strip().split():
            # Transition matrix
                fea.append(word.strip().split()[1])
            else:
                fea.append('END')
                for i in range(len(fea) - 1):
                    if (fea[i] in transitions):
                        transitions[fea[i]][fea[i+1]] = transitions[fea[i]].get(fea[i+1], 0) + 1
                    else:
                        transitions[fea[i]] = {}
                        transitions[fea[i]][fea[i+1]] = 1
                fea = ['START']

                    

        states = word_states.keys()
        
        if self.smoothing_type == 'add-d smoothing':
            word_states_prob, emissions_prob, transitions_prob = self.add_d_smoothing(word_states, emissions, transitions)
        
        return states, word_states_prob, emissions_prob, transitions_prob  
    # Smoothing in extraction.py
    def add_d_smoothing(self, word_states, emissions, transitions):
        # Smoothing parameter
        d = self.smoothing_d
        # Unigram
        word_states_prob = {}
        emissions_prob = copy.deepcopy(emissions)
        transitions_prob = copy.deepcopy(transitions)
        length = len(word_states)
        total_value = sum(word_states.values())
        for key in word_states:
            word_states_prob[key] = (word_states[key] + d) / (total_value + d * length)
        word_states_prob['TBD'] = d / (total_value + d * length)
        # Bigram
        for key in emissions:
            total_value = sum(emissions[key].values())
            for value in emissions[key]:
                emissions_prob[key][value] =  (emissions[key][value] + d) / (total_value + d * len(emissions[key]))
            emissions_prob[key]['TBD'] = d / (total_value + d * len(emissions[key]))
        # add "START" and "END" at the start and end of the sentence
        for key in transitions:
            total_value = sum(transitions[key].values())
            for value in transitions[key]:
                transitions_prob[key][value] =  (transitions[key][value] + d) / (total_value + d * (length))
            transitions_prob[key]['TBD'] = d / (total_value + d * (length))
            
            if 'END' in transitions[key]:
                transitions_prob[key]['END'] =  (transitions[key]['END'] + d) / (total_value + d * (length+1))
            else:
                transitions_prob[key]['END'] = d / (total_value + d * (length+1))
            
        return word_states_prob, emissions_prob,  transitions_prob         
    # Used in trigram_extraction.py
    def LoadforTrigram(self, smoothing_d, mode = 'train'):
        self.smoothing_d = smoothing_d
        if mode == 'train':
            f = codecs.open(self.file_pathtype + '_train.txt','r','utf8')
            text = f.readlines()
            f.close()

        word_states = {}
        states = {}
        transitions = {}
        emissions = {}
        
        #  Emission matrix & word matrix
        for word in text:
            if word.strip().split():
                # Unigram
                word_type = word.strip().split()
                word_states[word_type[1]] = word_states.get(word_type[1], 0) + 1
                # Emsission matrix
                if (word_type[1] in emissions):
                    emissions[word_type[1]][word_type[0]] = emissions[word_type[1]].get(word_type[0], 0) + 1
                else:
                    emissions[word_type[1]] = {}
                    emissions[word_type[1]][word_type[0]] = 1
        
            
        # Bigram count
        transitions_Big = {}
        fea = ['START']
        for word in text:
            if word.strip().split():
            # Transition matrix
                fea.append(word.strip().split()[1])
            else:
                fea.append('END')
                for i in range(len(fea) - 1):
                    if (fea[i] in transitions_Big):
                        transitions_Big[fea[i]][fea[i+1]] = transitions_Big[fea[i]].get(fea[i+1], 0) + 1
                    else:
                        transitions_Big[fea[i]] = {}
                        transitions_Big[fea[i]][fea[i+1]] = 1
                fea = ['START']
        
        # Trigram count
        fea = ['START']
        transitions_tri = {}
        for word in text:
            if word.strip().split():
                fea.append(word.strip().split()[1])
            else:
                fea.append('END')
                for i in range(2,len(fea)):
                    if (' '.join(fea[i-2:i]) in transitions_tri):
                        transitions_tri[' '.join(fea[i-2:i])][fea[i]] = transitions_tri[' '.join(fea[i-2:i])].get(fea[i], 0) + 1
                    else:
                        transitions_tri[' '.join(fea[i-2:i])] = {}
                        transitions_tri[' '.join(fea[i-2:i])][fea[i]] = 1
                fea = ['START']
        
        states = word_states.keys()
        states = list(states)
        word_states_prob, transitions_triprob, transitions_bigprob, emissions_prob = self.smoothing_trigram(emissions, word_states, transitions_Big, transitions_tri)
        return states, word_states_prob, transitions_triprob, transitions_bigprob, emissions_prob
    # Smoothing in trigram_extraction.py
    def smoothing_trigram(self, emissions, word_states, transitions_Big, transitions_tri):
        # smoothing patameter
        d = self.smoothing_d
        # word_states
        total = sum(word_states.values())
        for key in word_states:
            word_states[key] /= total
        word_states['TBD'] = min(word_states.values()) / total
        word_states['END'] = min(word_states.values()) / total
        length = len(word_states)
        # Emission matrix
        emissions_prob = {}
        for key in emissions:
            emissions_prob[key] = {}
            total_value = sum(emissions[key].values())
            for value in emissions[key]:
                emissions_prob[key][value] =  (emissions[key][value] + d) / (total_value + d * len(emissions[key]))
            emissions_prob[key]['TBD'] = d / (total_value + d * len(emissions[key]))
        
        # Biggram
        transitions_Bigprob = {}
        for key in transitions_Big:
            total_value = sum(transitions_Big[key].values())
            length = len(transitions_Big[key])
            transitions_Bigprob[key] = {}
            for value in transitions_Big[key]:
                transitions_Bigprob[key][value] =  (transitions_Big[key][value] + d) / (total_value + d * (length))
            transitions_Bigprob[key]['TBD'] = d / (total_value + d * (length+1))
            
            if 'END' in transitions_Big[key]:
                transitions_Bigprob[key]['END'] =  (transitions_Big[key]['END'] + d) / (total_value + d * (length+1))
            else:
                transitions_Bigprob[key]['END'] = d / (total_value + d * (length+1))
        
        # Trigram
        for key in transitions_tri:
            total_value = sum(transitions_tri[key].values())
            for value in transitions_tri[key]:
                transitions_tri[key][value] =  (transitions_tri[key][value] + d) / (total_value + d * (length + 1))
            transitions_tri[key]['TBD'] =  (transitions_tri[key][value] + d) / (total_value + d * (length + 1))
        # Weighted average of different language models
        transitions_prob = {}
        l1 = 0.1
        l2 = 0.1
        l3 = 0.8
        for key in transitions_tri:
            transitions_prob[key] = {}
            for value in transitions_tri[key]:
                p2 = key.split()[0]
                p1 = key.split()[1]
                if value in transitions_Big[p1]:
                    transitions_prob[key][value] = l1 * transitions_tri[key][value] + l2 * transitions_Bigprob[p1][value] + l3 * word_states[value]
                else:
                    transitions_prob[key][value] = l1 * transitions_tri[key][value] + l2 * transitions_Bigprob[p1]['TBD'] + l3 * word_states[value]
            transitions_prob[key]['TBD'] = 10 ** (-10)
        return word_states, transitions_prob, transitions_Bigprob, emissions_prob

                     

                
                    
                
        
        


        
                    
        
                
        
    