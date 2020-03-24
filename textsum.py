# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:43:04 2020

@author: PHULL
"""

text='''
An important point is that while human teachers are capable of generating either “deep” questions involving complex inference or “shallow” factual questions, automatic techniques are much
more likely to be error prone when complex inference is involved than when it is not. On the other
hand, automated QG tools may be capable of generating large sets of shallow questions very quickly
and could help teachers to focus on generating good deep questions. 
'''


from urllib import request
from bs4 import BeautifulSoup as bs
import re
import nltk
import heapq

url="https://en.wikipedia.org/wiki/Computational_humor"
htmldoc=request.urlopen(url)
para_all=""
soupob=bs(htmldoc,'html.parser')
para=soupob.findAll('p')

print(para)

for p in para:
    para_all += p.text

print(para_all)

para_all_clean=re.sub(r'\[[0-9]*\]',' ',para_all)
para_all_clean=re.sub(r'\s+',' ',para_all_clean)
print(para_all_clean)

s_tokens=nltk.sent_tokenize(para_all_clean)

## remove all characters other than alphabets
para_all_clean=re.sub(r'[^a-zA-Z]',' ',para_all_clean)
para_all_clean=re.sub(r'\s+',' ',para_all_clean)
print(para_all_clean)

## tokenize

w_tokens=nltk.word_tokenize(para_all_clean)

## remove stop words
stopwords=nltk.corpus.stopwords.words('english')

word_freq={}

for w in w_tokens:
    if w not in stopwords:
        if w not in word_freq.keys():
            word_freq[w] = 1
        else:
            word_freq[w] += 1

print(word_freq)

## weighted frequencies
max_freq=max(word_freq.values())

for w in word_freq.keys():
    word_freq[w]= (word_freq[w]/max_freq)

print(word_freq)

## sentence score with each word weighted frequency

s_scores={}

for s in s_tokens:
    for w in nltk.word_tokenize(s.lower()):
        if w in word_freq.keys():
            if (len(s.split(' '))) < 30:
                if s not in s_scores.keys():
                    s_scores[s]= word_freq[w]
                else:
                    s_scores[s] += word_freq[w]
print(s_scores)   

summary= heapq.nlargest(10,s_scores,key=s_scores.get)
print(summary)
