from DataLoader import Dataloader
import numpy as np
import codecs
import argparse
class HMM(object):
    def __init__(self, config):
        self.dataLoader = Dataloader(config)
        
        self.file_pathtype = config.file_pathtype
        self.smoothing_d = config.smoothing_d
        
    def viterbi(self, states, word_states_prob, emissions_prob, transitions_prob, sentence):
        probability = [{}]
        path = {} 
        # The first state
        for state in states:
            if (sentence[0] in emissions_prob[state]) and (state in transitions_prob['START']):
                probability[0][state] = np.log(transitions_prob['START'][state]) + np.log(emissions_prob[state][sentence[0]])
            elif (state in transitions_prob['START']):
                probability[0][state] = np.log(transitions_prob['START'][state]) + np.log(emissions_prob[state]['TBD'])
            else:
                probability[0][state] = np.log(transitions_prob['START']['TBD']) + np.log(emissions_prob[state]['TBD'])
                
            path[state] = [state]
        #other words in sentense
        for index in range(1, len(sentence)):
            update_path = {}
            probability.append({})
            
            for state in states:
                trans_prob = {}
                for pri_state in states:
                    if state in transitions_prob[pri_state] and (sentence[index] in emissions_prob[state]):
                        trans_prob[pri_state] = probability[index-1][pri_state] + np.log(transitions_prob[pri_state][state]) + np.log(emissions_prob[state][sentence[index]])
                    elif state in transitions_prob[pri_state]:
                        trans_prob[pri_state] = probability[index-1][pri_state] + np.log(transitions_prob[pri_state][state]) + np.log(emissions_prob[state]['TBD'])                                            
                    elif (sentence[index] in emissions_prob[state]):
                        trans_prob[pri_state] = probability[index-1][pri_state] + np.log(transitions_prob[pri_state]['TBD']) + np.log(emissions_prob[state][sentence[index]])
                    else:
                        trans_prob[pri_state] = probability[index-1][pri_state] + np.log(transitions_prob[pri_state]['TBD']) + np.log(emissions_prob[state]['TBD'])
                # find the best path in state t-1
                max_key = max(trans_prob, key = trans_prob.get)
                max_prob = trans_prob[max_key]
                probability[index][state]  = max_prob
                update_path[state] = path[max_key] + [state]
            # update the path
            path = update_path
        
        probability.append({})
        # consider for the last word in sentence
        for state in states:
            probability[len(sentence)][state] = probability[len(sentence)-1][state] + np.log(transitions_prob[state]['END'])
        
        
        key = max(probability[-1], key = probability[-1].get)
        
        return path[key]


    def evalution(self):
        f = codecs.open(self.file_pathtype + '_test.txt', 'r', 'utf8')
        text = f.readlines()
        f.close()
        len_text = len(text)
        states, word_states_prob, emissions_prob, transitions_prob  = self.dataLoader.Load(self.smoothing_d, mode = 'train', )
        sentence = []
        features = []
        perdict = []
        all_sentence = []
        count = 0
        for words in text:
            if words.strip():
                word = words.strip().split()
                all_sentence.append(word[0])
                sentence.append(word[0])
                features.append(word[1])
                count += 1
            else:
                # process the sentence
                perdict += self.viterbi(states, word_states_prob, emissions_prob, transitions_prob, sentence)
                sentence = []
                
        f = codecs.open(self.file_pathtype + '_result.txt', 'w', 'utf8')
        for i in range(len(perdict)):
            wri = all_sentence[i] + '\t' +  features[i] + '\t' + perdict[i] + '\n'
            f.write(wri)
        f.close()
        # Evaluation
        TP, FP, TN, FN, type_correct, sum = 0, 0, 0, 0, 0, 0
        sumlen = len(perdict)
        for i in range(sumlen):
            if features[i] != 'O' and perdict[i] != 'O':
                TP += 1
                if features[i] == perdict[i]:
                    type_correct += 1
            if features[i] != 'O' and perdict[i] == 'O':
                FN += 1
            if features[i] == 'O' and perdict[i] != 'O':
                FP += 1
            if features[i] == 'O' and perdict[i] == 'O':
                TN += 1

        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        accuracy = (TP+TN)/sumlen
        F1 = 2 * precision * recall/(precision+recall)

        print('====='+  self.file_pathtype +' labeling result=====')
        print("type_correct: ", round(type_correct/TP, 4))
        print("accuracy: ",round(accuracy, 4))
        print("precision: ",round(precision, 4))
        print("recall: ",round(recall,4))
        print("F1: ",round(F1,4))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    #mode
    parser.add_argument('--file_pathtype', type=str, default='./argument')
    # If test with trigger, change the file_type as './trigger'
    # else: './argument'
    parser.add_argument('--smoothing_type', type=str, default='add-d smoothing')
    parser.add_argument('--smoothing_d', type=int, default=1 * 10**(-8))

    config = parser.parse_args() #return a name space
    hmm = HMM(config)
    hmm.evalution()
      


                        