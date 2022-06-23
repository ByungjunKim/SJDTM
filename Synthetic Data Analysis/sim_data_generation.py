import math
from math import log, exp
import random
import scipy.stats
from scipy.stats import poisson, dirichlet
from scipy import optimize
import copy
from copy import deepcopy
import json
import time
import numpy as np
def simu(inp_topic_nums, inp_topics_lag, inp_V=500, inp_doc_n_words_poisson=200):
    
    # Generating simulation data
    random.seed(300)
    pre_time_slice = [50,50,50,50,50,50,50,50,50,50]#Number of documents for pre-corpus
    post_time_slice = [50,50,50,50,50,50,50,50,50,50]#Number of documents for post-corpus
    topic_nums_shared = inp_topic_nums# Nmuber of shared topics
    topic_nums_pre = inp_topic_nums#Nmuber of lead topics
    topic_nums_post = inp_topic_nums#Nmuber of lag topics
    topics_lag = inp_topics_lag#list, the topic lag
    max_lag = max(topics_lag) + 2
    V = inp_V#Number of words
    T = len(pre_time_slice)# Nmuber of time slice

    N_topic_pre = topic_nums_shared + topic_nums_pre#Number of topics for pre-corpus
    N_topic_post = topic_nums_shared + topic_nums_post#Number of topics for post-corpus

    theta_dirichlet_pre = [1] * N_topic_pre#pre-corpus Dirichlet prior
    theta_dirichlet_post = [1] * N_topic_post#post-corpus Dirichlet prior
    doc_n_words_poisson = inp_doc_n_words_poisson#Number of words for each document
    doc_n_words_min = 50

    beta_variance_init = 2 #init topic-word distribution beta
    beta_variance_chain = 1.5
    beta_mean_init = 0

    pre_time = []
    post_time = []
    n = 0
    for t_pre, t_post in zip(pre_time_slice, post_time_slice):
        pre_time.extend([n]*t_pre)
        post_time.extend([n]*t_post)
        n += 1
    iteration = 1
    while iteration ==1 :

    # Generate beta
        beta_pre_true = [[[random.gauss(beta_mean_init, beta_variance_init) \
                        for y in range(topic_nums_pre)] \
                            for x in range(V)]]
        for t in range(T-1):
            beta_pre_true.append([[random.gauss(y, beta_variance_chain) for y in x] for x in beta_pre_true[t]])#方差不变，得到的是T*V*K维的beta

        beta_norm_pre_true = deepcopy(beta_pre_true)
        for t in range(T):
            for k in range(topic_nums_pre):
                s = sum([exp(x[k]) for x in beta_norm_pre_true[t]])
                for word in range(V):
                    beta_norm_pre_true[t][word][k] = exp(beta_norm_pre_true[t][word][k])/s


        beta_post_true = [[[random.gauss(beta_mean_init, beta_variance_init) \
                        for y in range(topic_nums_post)] \
                            for x in range(V)]]
        for t in range(T-1):
            beta_post_true.append([[random.gauss(y, beta_variance_chain) for y in x] for x in beta_post_true[t]])

        beta_norm_post_true = deepcopy(beta_post_true)
        for t in range(T):
            for k in range(topic_nums_post):
                s = sum([exp(x[k]) for x in beta_norm_post_true[t]])
                for word in range(V):
                    beta_norm_post_true[t][word][k] = exp(beta_norm_post_true[t][word][k])/s


        beta_shared_true = [[[random.gauss(beta_mean_init, beta_variance_init) \
                    for y in range(topic_nums_shared)] \
                        for x in range(V)]]
        for t in range(T-1+max_lag):
            beta_shared_true.append([[random.gauss(y, beta_variance_chain) for y in x] for x in beta_shared_true[t]])

        beta_norm_shared_true = deepcopy(beta_shared_true)
        for t in range(T+max_lag):
            for k in range(topic_nums_shared):
                s = sum([exp(x[k]) for x in beta_norm_shared_true[t]])
                for word in range(V):
                    beta_norm_shared_true[t][word][k] = exp(beta_norm_shared_true[t][word][k])/s

    # change to ndarray
        beta_pre_true_array = np.array(beta_pre_true)
        beta_norm_pre_true_array = np.array(beta_norm_pre_true)

        beta_post_true_array = np.array(beta_post_true)
        beta_norm_post_true_array = np.array(beta_norm_post_true)

        beta_shared_true_array = np.array(beta_shared_true)
        beta_norm_shared_true_array = np.array(beta_norm_shared_true)
        beta_norm_init=[]
        beta_norm_init.append(beta_norm_shared_true)
        beta_norm_init.append(beta_norm_pre_true)
        beta_norm_init.append(beta_norm_post_true)

    # Generate pre corpus
        pre_corpus = []
        theta_pre_true = []
        lambda_pre_true = []
        for i_doc, t in zip(range(sum(pre_time_slice)), pre_time):

            theta_doc = dirichlet.rvs(theta_dirichlet_pre)[0].tolist()#get doc-word distribution theta
            n_words = max(poisson.rvs(doc_n_words_poisson),50)#
            lambda_doc = [random.betavariate(1,1) for _ in range(N_topic_pre)]

            doc_orig = []
            for word in range(n_words):
                lambda_word = [random.choices([0,1], weights=[1-l, l], k=1)[0] \
                     for l in lambda_doc]#N_topic_pre维的0/1
                k = random.choices(range(N_topic_pre), theta_doc)[0]
                if lambda_word[k] == 1:
                    if k < topic_nums_shared:
                        word = random.choices(range(V), beta_norm_shared_true_array[t+max_lag][:,k])[0]
               #extract word，the first topic_nums_shared topics are shared topics, and the following are lead topics
                    else:
                        word = random.choices(range(V), beta_norm_pre_true_array[t][:,k-topic_nums_shared])[0]
                    doc_orig.append(word)
                else:
                    continue

            doc = dict()
            for word in doc_orig:
                if word in doc:
                    doc[word] += 1
                else:
                    doc[word] = 1
            doc = sorted(list(doc.items()))
            pre_corpus.append(doc)
            theta_pre_true.append(theta_doc)
    # Generate post corpus
        post_corpus = []
        theta_post_true = []
        for i_doc, t in zip(range(sum(post_time_slice)), post_time):

            theta_doc = dirichlet.rvs(theta_dirichlet_post)[0].tolist()
            n_words = max(poisson.rvs(doc_n_words_poisson),50)
            lambda_doc = [random.betavariate(1,1) for _ in range(N_topic_post)]

            doc_orig = []
            for word in range(n_words):
                lambda_word = [random.choices([0,1], weights=[1-l, l], k=1)[0] \
                     for l in lambda_doc]
                k = random.choices(range(N_topic_post), theta_doc)[0]
                if lambda_word[k] == 1:
                    if k < topic_nums_shared:
                        word = random.choices(range(V), beta_norm_shared_true_array[t+max_lag-topics_lag[k]][:,k])[0]
                    else:
                        word = random.choices(range(V), beta_norm_post_true_array[t][:,k-topic_nums_shared])[0]
                    doc_orig.append(word)
                else: continue

            doc = dict()
            for word in doc_orig:
                if word in doc:
                    doc[word] += 1
                else:
                    doc[word] = 1
            doc = sorted(list(doc.items()))
            post_corpus.append(doc)
            theta_post_true.append(theta_doc)
        id2word1 = utils.dict_from_corpus(pre_corpus)
        id2word2 = utils.dict_from_corpus(post_corpus)
        if len(id2word1)==V and len(id2word2)==V:
            print('generating same length corpus',len(id2word1))
            iteration+=1
        else: iteration=1 
    return pre_corpus, post_corpus, pre_time_slice, post_time_slice,max_lag,topic_nums_shared,topic_nums_pre,topic_nums_post,id2word1,beta_norm_init,theta_pre_true,theta_post_true,beta_norm_shared_true,beta_norm_pre_true,beta_norm_post_true