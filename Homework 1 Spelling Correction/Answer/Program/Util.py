from nltk.corpus import reuters
import random
def levenshtein(word1, word2):
    '''
    Calculate the Edit distance and return the number and path to edit
    '''
    len1 = len(word1) + 1
    len2 = len(word2) + 1
    dp = [[0 for col in range(len2)] for row in range(len1)]
    for i in range(len1):
        for j in range(len2):
            if (i == 0):
                dp[i][j] = j
            elif (j == 0):
                dp[i][j] = i

    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if (word1[i-1] == word2[j-1]):
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1, dp[i][j-1] + 1, dp[i-1][j] + 1 )
    reversecount = 0
    path = []
    if (dp[i][j] < 4):
        
        i = len1 - 1
        j = len2 - 1
        while ((i!=0) or (j!=0)):
            if (word1[i-1] == word2[j-1]):
                temp = 0
            else:
                temp = 1
            if ((dp[i][j] == (dp[i-1][j-1] + temp)) and (i>=1) and (j>=1)):
                if temp:
                    ins = word1[i-1] + '|' + word2[j-1]
                    path.append(ins)
                i -= 1
                j -= 1
            
            elif ((dp[i][j] == dp[i-1][j] + 1) and (i>=1)):
                ins = word1[i-2:i] + '|' + word2[j-1]
                path.append(ins)
                i -= 1
            
            elif ((dp[i][j] == dp[i][j-1] + 1) and (j>=1)):
                ins = word1[i-1] + '|' + word2[j-2:j]
                path.append(ins)
                j -= 1 
        
        for i in range(len(path)-1, 0,- 1):
            # Detect replacement
            if (path[i] == path[i-1][::-1]):
                reverse = path[i][0] + path[i-1][0] + '|' + path[i-1][0] + path[i][0]
                reversecount += 1
                path[i-1] = reverse
                del(path[i])
    return dp[len(word1)][len(word2)]-reversecount , path


def gettxt(path):
    '''
    Read the vocab text
    '''
    with open(path, "r") as f:
        vocab = f.read().split('\n')
    del(vocab[-1])
    return vocab
    
def randomsampling(length,mini,maxi):
    re = []
    while len(re)<length:
        inte = random.randint(mini,maxi)
        if inte not in re:
            re.append(inte)
    return re

    
def write(dictionary, path):
    '''
    Write the languaga modelto file
    '''
    f = open(path, "w")
    count = sum(dictionary.values())
    countl = str(count) + '\n'
    f.writelines(countl)
    for (key,value) in dictionary.items():
        wr = key + ' ' + str(value) + '\n'
        f.writelines(wr)
    f.close()
    return 1 
    
def Uniloader(path):
    '''
    Load the Unigram language
    '''
    with open(path,'r') as f:
        dic = f.read().split('\n')
    count = dic[0]
    del(dic[0])
    del(dic[-1])
    dictionary = {}   
    for line in dic:
        sp = line.split()
        dictionary[sp[0]] = int(sp[1])
    return count, dictionary

def Bigloader(path):
    '''
    Load the Bigram language model
    '''
    with open(path,'r') as f:
        dic = f.read().split('\n')
    count = dic[0]
    del(dic[0])
    del(dic[-1])
    dictionary = {}   
    for line in dic:
        sp = line.split()
        dictionary[sp[0] + ' '+ sp[1]] = int(sp[2])
    return count, dictionary

def loaderchange(path):
    '''
    Load the probability of edit distance
    '''
    with open(path,'r', encoding='UTF-8') as f:
        dic = f.read().split('\n')
    del(dic[-1])
    dictionary = {}   
    for line in dic:
        sp = line.split('\t')
        dictionary[sp[0]] = int(sp[1])
    return dictionary           
        
def reuter():
    '''
    Create the reuters corpus from nltk 
    '''
    word = []
    for i in reuters.fileids():
        word.append(reuters.words(fileids = [i]))
    return word 
    

if __name__ == '__main__':
    word1 = 'svploc'
    word2 = 'vsplo'
    dp, path = levenshtein(word1, word2)
    print(dp,path)
        