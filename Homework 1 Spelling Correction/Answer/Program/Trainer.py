import os
import Util
from nltk.corpus import reuters
from math import log,exp
import nltk

class Solver(object):
    def __init__(self,config):
        self.modelsavepath = config.model_path
        self.data_root = config.data_root
        self.smoothingMethod = config.smoothingMethod
        
        self.Unigrampath = os.path.join(self.data_root, 'Unigram.txt')
        self.Bigrampath = os.path.join(self.data_root, 'Bigram.txt')
        self.missionpath = os.path.join(self.data_root, 'testdata.txt')
        self.vocabpath = os.path.join(self.data_root, 'vocab.txt')
        self.changepath = os.path.join(self.data_root, 'change.txt')
        self.Followpath = os.path.join(self.data_root, 'Followpath.txt')
        self.punctions = [',','.',"'s","n't","'ve", "'ll"]  #
        if not os.path.exists(self.data_root):
            os.mkdir(self.data_root)
        self.vocab = Util.gettxt(self.vocabpath)       
        self.maxchangecount = config.maxchangecount
        self.weight = config.weight.split()
        
        #self.anspath = './elements'
        
    def modeling(self):
        '''
        Create the Unigram and Bigram language model by the Reuters Corpus 
        '''
        if not os.path.exists(self.data_root):
            os.mkdir(self.data_root)
        words = Util.reuter() # To get all the words & conbination from Reuter
        length = len(words)
        Unigram = {}
        Bigram = {}
        for j in range(length):
            print(j,'/',length)
            for i in range(len(words[j])):
                if words[j][i] in self.vocab:
                    Unigram[words[j][i]] = Unigram.get(words[j][i], 0) + 1
                    if ((i+1) < len(words[j])):
                        if words[j][i+1] in self.vocab:
                            connect = words[j][i]+ ' ' + words[j][i+1]
                            Bigram[connect] = Bigram.get(connect, 0) + 1
        
        Follow = {}
        for dict_key in Bigram.keys():
            key = dict_key.split()
            for k in key:
                Follow[k] = Follow.get(k, 0) + 1
                
        
        Util.write(Unigram, self.Unigrampath)
        Util.write(Bigram, self.Bigrampath)
        Util.write(Follow, self.Followpath)
        #Save the model   
        

    
    def punctionflow(self, word_pun):
        '''
        Extract the suffix of the word so that it can be retrieved in the dictionary
        '''
        for punction in self.punctions:
            if (punction in word_pun) and (word_pun not in self.vocab):
                punction_re = punction
                word = word_pun.rstrip(punction)
                return punction_re,word
        return  '',word_pun

    
    def languageModel(self, config, Unigram, Unigramcount, Bigram, Bigramlen, replaceSentence, smoothing = 'add-k smoothing', follow = None):
        UnigramProb = 0
        BigramProb = 0
        #Language Model Probability
        for w in range(len(replaceSentence)-1):
            UnigramProb += log(Unigram.get(replaceSentence[w],1) / Unigramcount)
            precursor = replaceSentence[w+1]
            precursorConbin = replaceSentence[w] + ' ' + replaceSentence[w+1]
            
            if smoothing == 'add-k smoothing':
                k = config.k_in_addk
                BigramProb += log((Bigram.get(precursorConbin, 0) + k) / (Unigram.get(precursor, 0)  + k*Bigramlen))
            
            elif smoothing == 'Absolute Discounting Interpolation':

                BigramProb += log(max(Bigram.get(precursorConbin,0)-0.75, 1) / Unigram.get(precursor, 1) +  follow.get(precursor,0) * Unigram.get(replaceSentence[w],0) / Unigramcount)
            
            else:
                print( 'Not in the smoothing method list!')
        return UnigramProb, BigramProb
            
    def test(self,config):
        # Load the model and the test data
        Unigramcount, Unigram = Util.Uniloader(self.Unigrampath)
        Unigramcount = int(Unigramcount)
        Bigramlen, Bigram = Util.Bigloader(self.Bigrampath)
        Bigramlen = int(Bigramlen)
        Change = Util.loaderchange(self.changepath)
        changecount = sum(Change.values())
        missions = Util.gettxt(self.missionpath)
        Followlen, Follow = Util.Uniloader(self.Followpath)
        
        '''
        #For vaildation
        ansfile = open(self.anspath,'r')
        anslines = ansfile.readlines()
        ansfile.close()
        '''




        #realword = [5, 19, 47, 54, 64, 118, 123, 138, 153, 157, 170, 177, 265, 267, 294, 356, 391, 410, 435, 471, 493, 534, 538, 553, 619, 620, 635, 681, 684, 768, 783, 801, 813, 819, 843, 860, 869, 875, 901, 911, 931, 932, 945, 954, 955, 968, 979, 987]
        
        for count in  range(len(missions)):
        
            
            mission = missions[count]
            # Using slides to quickly locate the place that can be incorrect
            mission = mission.split('\t')
            No = mission[0]
            Enumber = int(mission[1])
            print(No , '/' , '1000')
            finalpun = mission[2][-1]
            sentence = mission[2][:-1:].split()
            #Exact the information of the processing row
            nonwordloc = []
            nonwordnum = 0
            
            
            
            ###################################    Non-word Error Processing #####################         
            for word_puct in sentence:                
                punction, word = self.punctionflow(word_puct)
                if word not in self.vocab:
                    index = sentence.index(word_puct)
                    nonwordnum += 1
                    origin = word
                    
                    replace = {}   # Alternative words and their probabilities
                    #count = 0 
                    
                    for reword in self.vocab:
                        #count += 1
                        #print(count,'/', len(self.vocab))
                        #reword = self.vocab[1990]  ####
                        if (abs(len(reword) - len(origin)) <= 3):
                            # First judge the length to reduce the operation of the edit distance module
                            editdistance, editpath = Util.levenshtein(origin, reword)
                            if (editdistance <= 2):
                                replace[reword] = 0
                                replaceProb = 0
                                
                                #Edit distance Probability
                                for i in range(editdistance):
                                    replaceProb += log(Change.get(editpath[i],1) / changecount)
                                
                                replaceSentence = sentence
                                replaceSentence[index] = reword + punction
                                UnigramProb, BigramProb = self.languageModel(config, Unigram, Unigramcount, Bigram, Bigramlen, replaceSentence, smoothing = self.smoothingMethod, follow = Follow)
                                
                                # Add the probability by weight
                                replace[reword]  = float(self.weight[0]) * replaceProb +  float(self.weight[1]) * UnigramProb + float(self.weight[2]) * BigramProb
                            
                    replace = sorted(replace.items(),key=lambda item:item[1],reverse=True)
                    replace = replace[:5]
                    # Choose the maximun likelihood word
                    print(origin, replace)
                    sentence[index] = replace[0][0] + punction
                    
            ############################## Real-word Error ##########################################################
            if nonwordnum < Enumber:           
                tailnumber = Enumber - nonwordnum  # The number of real word error 
                replace = {}
                for index in range(len(sentence)):
                    if index not in nonwordloc:
                        
                        #print(index, '/' , len(sentence) )
                        origin_pun = sentence[index]
                        punction,origin = self.punctionflow(origin_pun)
                        
                        for word in self.vocab:
                            if (abs(len(word) - len(origin)) <= 3):
                                editdistance, editpath = Util.levenshtein(origin, word)
                                if (editdistance <= 2):
                                    word = word + punction
                                    inword = str(index) + ' ' + word
                                    replaceProb = 0
                                    
                                    if editdistance == 0:
                                        replaceProb += log(self.maxchangecount / changecount)
                                    # If the alternative word is the word itself, the Edit distance probabity is the maximun of all
                                    for i in range(editdistance):
                                        if editpath[i] in Change:
                                            replaceProb += log(Change[editpath[i]] / changecount)
                                        else:
                                            replaceProb += log(1 / changecount)
                                    
                                    replaceSentence = sentence[::]
                                    replaceSentence[index] = word
                                    UnigramProb = 0
                                    BigramProb = 0
                                                            
                                    # Language Model Processing
                                    UnigramProb, BigramProb = self.languageModel(config, Unigram, Unigramcount, Bigram, Bigramlen, replaceSentence, smoothing = self.smoothingMethod, follow = Follow)
                                    replace[inword] = float(self.weight[0])* replaceProb + float(self.weight[1]) * UnigramProb + float(self.weight[2]) * BigramProb
                replace = sorted(replace.items(),key=lambda item:item[1],reverse=True)
                replace = replace[:100]
                print(origin, replace)
                i = 0
                hd = []
                '''
                Because in the real word error detection, 
                the word itself is also one of the alternative words, 
                so we need to remove the word itself when doing the replacement
                '''                
                for j in range(len(replace)):
                    sen = replace[j][0].split()
                    index = int(sen[0])
                    re = sen[1]
                    if (index not in hd) and (sentence[index].lower() != re.lower()):
                        sentence[index] = re
                        hd.append(index)
                        i += 1
                        if (i >= tailnumber):
                            break
            writeSentence = No + '\t' + ' '.join(sentence) + finalpun + '\n'
            # Save the result into the file
            with open('result.txt', "a+") as f:
                f.write(writeSentence)
                
                
            '''      
            #For vaildation      
            resultSentence = ' '.join(sentence) + finalpun
            resultset=set(nltk.word_tokenize(resultSentence))                 
            if ansset==resultset:
                correct+=1
            
        writeSentence = str(k) + ' ' + str(correct) + '\n'
        with open('vaildation.txt', "a+") as f:
            f.write(writeSentence)
            '''

            
                    
                
        

                
                
            
            
                            
                        