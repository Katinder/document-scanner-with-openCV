# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:55:02 2020

@author: PHULL
"""

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

stop_words = stopwords.words('english')
summarize_text = []
    
def sent_sim(sent1=sentences[i2-1],sent2=sentences[i2],stop_words=[]):
    sent1=[w.lower() for w in sent1]
    sent2=[w.lower() for w in sent2]
    
    allwords=list(set(sent1+sent2))
    
    v1=[0]*len(allwords)
    v2=[0]*len(allwords)
    
    for w in sent1:
        if w in stop_words:
            continue
        v1[allwords.index(w)]+=1
    
    for w in sent2:
        if w in stop_words:
            continue
        v2[allwords.index(w)]+=1
        
    return 1-cosine_distance(v1,v2)
    

ff='comphumor.txt'
file=open(ff,'r')
fdata=file.readlines()  ##paras to list elemnets
article=fdata[0].split(". ") ##first para
sentences=[]

for s in article:
    print("s= ",s)
    sentences.append(s.replace("[^a-zA-Z]"," ").split(" "))
    ##sentences.pop()
    
simmat=np.zeros((len(sentences),len(sentences)))

for i1 in range(len(sentences)):
    for i2 in range(len(sentences)):
        if i1==i2:
            continue
        simmat[i1][i2]=sent_sim(sentences[i1],sentences[i2],stopwords=[])

sentence_similarity_graph = nx.from_numpy_array(simmat)  ### makes an undirected graph
scores = nx.pagerank(sentence_similarity_graph)  ############# how it works??

ranked_sentence=sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)

print("Indexes of top ranked_sentence order are ", ranked_sentence)    

top_n=5
for i in range(top_n):
    summarize_text.append(" ".join(ranked_sentence[i][1]))

print("Summarize Text: \n", ". ".join(summarize_text))



















