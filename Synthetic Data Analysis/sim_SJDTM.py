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

from DTMpart import SJDTM_para
from gensim import utils,matutils
from sim_data_generation import simu
from sim_evalution_matrics import sKL,sKL_sum,calculate_pmi,calculate_coherence_static,calculate_coherence

# SJDTM model
def TWTopicModel( \
        pre_corpus, post_corpus, pre_time_slice, post_time_slice, \
        max_lag=2, topic_nums_shared=3, topic_nums_pre=3, topic_nums_post=3, \
        id2word=None, V=None, min_iter=5, max_iter=20, min_iter_em=2, max_iter_em=10, \
        chain_variance=0.005, is_WordsSparse=True, beta_norm_init=None, s_KappaToRo=1,inp_doc_n_words_poisson=100 \
    ):
    

    if id2word:
        V = len(id2word)
    T = len(pre_time_slice)

    D_pre_corpus = len(pre_corpus)
    D_post_corpus = len(post_corpus)

    pre_time = []
    post_time = []
    n = 0
    for t_pre, t_post in zip(pre_time_slice, post_time_slice):
        pre_time.extend([n]*t_pre)
        post_time.extend([n]*t_post)
        n += 1

    N_topic = topic_nums_shared + topic_nums_pre + topic_nums_post
    N_topic_pre = topic_nums_shared + topic_nums_pre
    N_topic_post = topic_nums_shared + topic_nums_post


    print('------- Initializing.')

    # init beta
    if beta_norm_init:
        beta_norm_shared = beta_norm_init[0]
        beta_norm_pre = beta_norm_init[1]
        beta_norm_post = beta_norm_init[2]

    else:
        print('Error: No beta_norm_init.')
        return 

    # init pre-corpus params
    phi_pre = []
    psi_pre = []
    theta_pre = []
    lambda_pre = []
    bound_pre_corpus_doc = []
    bound_pre_corpus = 0
    for i_doc, (doc, t) in enumerate(zip(pre_corpus, pre_time)):
        phi_doc = [[1/(N_topic_pre)] * N_topic_pre for _ in range(len(doc))]
        theta_doc = [1/N_topic_pre] * N_topic_pre
        if is_WordsSparse:
            psi_doc = [[0.8] * N_topic_pre for _ in range(len(doc))]
            lambda_doc = [0.8] * N_topic_pre
        else:
            psi_doc = [[1] * N_topic_pre for _ in range(len(doc))]
            lambda_doc = [1] * N_topic_pre

        # doc bound
        term_1 = 0
        term_2 = 0
        term_3 = 0
        term_4 = 0
        term_5 = 0
        for n, (word, n_word) in enumerate(doc):
            for k in range(N_topic_pre):
                if k < topic_nums_shared:
                    beta_kv = beta_norm_shared[max_lag+t][word][k]
                else:
                    beta_kv = beta_norm_pre[t][word][k-topic_nums_shared]
                term_1 += n_word * phi_doc[n][k]*psi_doc[n][k]*log(beta_kv)
                term_2 += n_word * phi_doc[n][k]*log(theta_doc[k])
                term_4 += n_word * phi_doc[n][k]*log(phi_doc[n][k])
                if is_WordsSparse:
                    term_3 += n_word * (psi_doc[n][k]*log(lambda_doc[k])+(1-psi_doc[n][k])*log(1-lambda_doc[k]))
                    term_5 += n_word * (psi_doc[n][k]*log(psi_doc[n][k])+(1-psi_doc[n][k])*log(1-psi_doc[n][k]))

        bound_doc = term_1 + term_2 + term_3 - term_4 - term_5
        bound_pre_corpus += bound_doc

        bound_pre_corpus_doc.append(bound_doc)
        phi_pre.append(phi_doc)
        psi_pre.append(psi_doc)
        theta_pre.append(theta_doc)
        lambda_pre.append(lambda_doc)

    # init post-corpus params
    ro = [[1/(max_lag+1) for y in range(max_lag+1)] for x in range(topic_nums_shared)]
    kappa = [1/(max_lag+1) for y in range(max_lag+1)]

    phi_post = []
    psi_post = []
    theta_post = []
    lambda_post = []
    for i_doc, (doc, t) in enumerate(zip(post_corpus, post_time)):
        phi_doc = [[1/N_topic_post] * N_topic_post for _ in range(len(doc))]
        theta_doc = [1/N_topic_post] * N_topic_post
        if is_WordsSparse:
            psi_doc = [[0.8] * N_topic_post for _ in range(len(doc))]
            lambda_doc = [0.8] * N_topic_post
        else:
            psi_doc = [[1] * N_topic_post for _ in range(len(doc))]
            lambda_doc = [1] * N_topic_post

        phi_post.append(phi_doc)
        psi_post.append(psi_doc)
        theta_post.append(theta_doc)
        lambda_post.append(lambda_doc)

    bound_post_corpus = 0
    for i_doc, (doc, t) in enumerate(zip(post_corpus, post_time)):
        term_1 = 0
        term_2 = 0
        term_3 = 0
        term_4 = 0
        term_5 = 0
        term_6 = 0
        term_7 = 0
        for n, (word, n_word) in enumerate(doc):
            for k in range(N_topic_post):
                if k < topic_nums_shared:
                    for l in range(max_lag+1):
                        term_1 += n_word * phi_post[i_doc][n][k]*psi_post[i_doc][n][k]*ro[k][l]*log(beta_norm_shared[t-l][word][k])
                else:
                    term_1 += n_word * phi_post[i_doc][n][k]*psi_post[i_doc][n][k]*log(beta_norm_post[t][word][k-topic_nums_shared])
                term_2 += n_word * phi_post[i_doc][n][k]*log(theta_post[i_doc][k])
                term_5 += n_word * phi_post[i_doc][n][k]*log(phi_post[i_doc][n][k])
                if is_WordsSparse:
                    term_3 += n_word * psi_post[i_doc][n][k]*log(lambda_post[i_doc][k]) + \
                            (1-psi_post[i_doc][n][k])*log(1-lambda_post[i_doc][k])
                    term_6 += n_word * psi_post[i_doc][n][k]*log(psi_post[i_doc][n][k]) + \
                            (1-psi_post[i_doc][n][k])*log(1-psi_post[i_doc][n][k])
        for k in range(topic_nums_shared):
            for l in range(max_lag+1):
                term_4 += ro[k][l]*log(kappa[l])
                term_7 += ro[k][l]*log(ro[k][l])

        bound_doc = term_1 + term_2 + term_3 + term_4 - term_5 - term_6 - term_7
        bound_post_corpus += bound_doc

    bound = bound_pre_corpus + bound_post_corpus

    print('----bound_pre_corpus:'+str(bound_pre_corpus))
    print('----bound_post_corpus:'+str(bound_post_corpus))
    print('----bound:'+str(bound))

    # Updating params
    n_iter = 1
    bound_pre = bound - 1
    while (n_iter <= min_iter) or ((n_iter <= max_iter) and (bound - bound_pre > 1e-5)):

        print('--------------- '+time.asctime()+' ---------------------')
        print('n_iter:'+str(n_iter))
        
        bound_pre = bound
        n_iter += 1


        # Update pre-corpus parameters
        print('Updating pre_corpus. Start at '+time.asctime())
        bound_pre_corpus = 0
        for i_doc, (doc, t) in enumerate(zip(pre_corpus, pre_time)):
            
            # print('i_doc:'+str(i_doc)+'. Starting.')

            N_words_doc = sum([x[1] for x in doc])

            if N_words_doc==0 : 
                N_words_doc+=1
            else:
                N_words_doc = N_words_doc
            # init phi,psi,theta,lambda for pre-corpus
            phi_doc = deepcopy(phi_pre[i_doc])
            psi_doc = deepcopy(psi_pre[i_doc])
            theta_doc = deepcopy(theta_pre[i_doc])
            lambda_doc = deepcopy(lambda_pre[i_doc])

            bound_doc = bound_pre_corpus_doc[i_doc]

            n_iter_em = 1
            bound_doc_pre = bound_doc - 1
            while (n_iter_em <= min_iter_em) or ((n_iter_em <= max_iter_em) and (bound_doc -  bound_doc_pre > 1e-5)):

                bound_doc_pre = bound_doc
                n_iter_em += 1

                phi_doc_pre = deepcopy(phi_doc)
                psi_doc_pre = deepcopy(psi_doc)
                theta_doc_pre = deepcopy(theta_doc)
                lambda_doc_pre = deepcopy(lambda_doc)

                # E step
                for n, (word, n_word) in enumerate(doc):
                    for k in range(N_topic_pre):
                        if k < topic_nums_shared:
                            beta_kv = beta_norm_shared[t+max_lag][word][k]
                        else:
                            beta_kv = beta_norm_pre[t][word][k-topic_nums_shared]
                        phi_doc[n][k] = max(theta_doc[k]*(beta_kv**psi_doc_pre[n][k]), 1e-200)
                        if is_WordsSparse:
                            term_tmp = lambda_doc_pre[k]*(beta_kv**phi_doc_pre[n][k])
                            psi_doc[n][k] = max(term_tmp/((1-lambda_doc_pre[k])+term_tmp), 1e-200)

                    # phi normalization
                    phi_doc[n] = [x/sum(phi_doc[n]) for x in phi_doc[n]]

                # M step
                for k in range(N_topic_pre):
                    theta_doc[k] = sum([x[k]*y[1] for x,y in zip(phi_doc, doc)])
                    if is_WordsSparse:
                        lambda_doc[k] = sum([x[k]*y[1] for x,y in zip(psi_doc, doc)])/N_words_doc

                # theta normalization
                if sum(theta_doc)==0:
                    theta_doc =[1/N_topic_pre]*N_topic_pre
                else :
                    theta_doc = [x/sum(theta_doc) for x in theta_doc]

                # doc bound
                term_1 = 0
                term_2 = 0
                term_3 = 0
                term_4 = 0
                term_5 = 0
                for n, (word, n_word) in enumerate(doc):
                    for k in range(N_topic_pre):
                        if k < topic_nums_shared:
                            beta_kv = beta_norm_shared[max_lag+t][word][k]
                        else:
                            beta_kv = beta_norm_pre[t][word][k-topic_nums_shared]
                        term_1 += n_word * phi_doc[n][k]*psi_doc[n][k]*log(beta_kv)
                        term_2 += n_word * phi_doc[n][k]*log(theta_doc[k])
                        term_4 += n_word * phi_doc[n][k]*log(phi_doc[n][k])
                        if is_WordsSparse:
                            term_3 += n_word * (psi_doc[n][k]*log(lambda_doc[k])+(1-psi_doc[n][k])*log(1-lambda_doc[k]))
                            term_5 += n_word * (psi_doc[n][k]*log(psi_doc[n][k])+(1-psi_doc[n][k])*log(1-psi_doc[n][k]))

                bound_doc = term_1 + term_2 + term_3 - term_4 - term_5
            
            phi_pre[i_doc] = phi_doc
            psi_pre[i_doc] = psi_doc
            theta_pre[i_doc] = theta_doc
            lambda_pre[i_doc] = lambda_doc

            bound_pre_corpus_doc[i_doc] = bound_doc
            bound_pre_corpus += bound_doc
        
        print('----n_iter_em:'+str(n_iter_em-1))
        print('----bound_pre_corpus:'+str(bound_pre_corpus))


        # Update post-corpus parameters
        print('Updating post_corpus. Start at '+time.asctime())
        n_iter_em = 1
        bound_post_corpus_pre = bound_post_corpus - 1
        while (n_iter_em <= min_iter_em) or ((n_iter_em <= max_iter_em) and (bound_post_corpus -  bound_post_corpus_pre > 1e-5)):

            print('post-corpus, n_iter_em:'+str(n_iter_em))

            bound_post_corpus_pre = bound_post_corpus
            n_iter_em += 1

            # E step
            ro_new = [[0 for y in range(max_lag+1)] for x in range(topic_nums_shared)]
            for i_doc, (doc, t) in enumerate(zip(post_corpus, post_time)):

                phi_doc = deepcopy(phi_post[i_doc])
                psi_doc = deepcopy(psi_post[i_doc])
                theta_doc = deepcopy(theta_post[i_doc])
                lambda_doc = deepcopy(lambda_post[i_doc])

                phi_doc_pre = deepcopy(phi_post[i_doc])
                psi_doc_pre = deepcopy(psi_post[i_doc])
                theta_doc_pre = deepcopy(theta_post[i_doc])
                lambda_doc_pre = deepcopy(lambda_post[i_doc])

                for n, (word, n_word) in enumerate(doc):
                    for k in range(N_topic_post):
                        if k < topic_nums_shared:
                            term_phi_tmp = 1
                            term_psi_tmp = 1
                            for l in range(max_lag+1):
                                term_phi_tmp *= beta_norm_shared[t+max_lag-l][word][k]**(psi_doc_pre[n][k]*ro[k][l])
                                term_psi_tmp *= beta_norm_shared[t+max_lag-l][word][k]**(phi_doc_pre[n][k]*ro[k][l])
                                ro_new[k][l] += n_word * phi_doc_pre[n][k] * psi_doc_pre[n][k] * \
                                                log(beta_norm_shared[t+max_lag-l][word][k]) / D_post_corpus
                            ro_new[k] = [x - max(ro_new[k]) for x in ro_new[k]]  # prevent overflow
                        else:
                            term_phi_tmp = beta_norm_post[t][word][k-topic_nums_shared]**psi_doc_pre[n][k]
                            term_psi_tmp = beta_norm_post[t][word][k-topic_nums_shared]**phi_doc_pre[n][k]
                        phi_doc[n][k] = max(theta_doc_pre[k] * term_phi_tmp, 1e-200)
                        if is_WordsSparse:
                            psi_doc[n][k] = max((lambda_doc_pre[k]*term_psi_tmp) / \
                                                ((1-lambda_doc_pre[k]) + lambda_doc_pre[k]*term_psi_tmp), \
                                            1e-200)
                                
                    # phi normalization
                    phi_doc[n] = [x/sum(phi_doc[n]) for x in phi_doc[n]]

                phi_post[i_doc] = phi_doc
                psi_post[i_doc] = psi_doc
            
            for k in range(topic_nums_shared):
                for l in range(max_lag+1):
                    ro_new[k][l] = (kappa[l] ** s_KappaToRo) * exp(ro_new[k][l])
                for l in range(max_lag+1):
                    if ro_new[k][l]==0 : ro_new[k][l]=1e-5
                    else:ro_new[k][l]=ro_new[k][l]
                ro_new[k] = [x/sum(ro_new[k]) for x in ro_new[k]]
            ro = deepcopy(ro_new)

            # M step
            for i_doc, (doc, t) in enumerate(zip(post_corpus, post_time)):
                
                N_words_doc = sum([x[1] for x in doc])
                if N_words_doc==0 : 
                    N_words_doc=1
                else:
                    N_words_doc = N_words_doc
                phi_doc = phi_post[i_doc]
                psi_doc = psi_post[i_doc]
                theta_doc = [0] * N_topic_post
                lambda_doc = [0] * N_topic_post

                for n, (word, n_word) in enumerate(doc):
                    for k in range(N_topic_post):
                        theta_doc[k] += n_word * phi_doc[n][k]
                        if is_WordsSparse:
                            lambda_doc[k] += n_word * psi_doc[n][k] / N_words_doc
                
                # theta normalization
                if sum(theta_doc)==0:
                    theta_doc =[1/N_topic_post]*N_topic_post
                else :
                    theta_doc = [x/sum(theta_doc) for x in theta_doc]
                
                theta_post[i_doc] = deepcopy(theta_doc)
                if is_WordsSparse:
                    lambda_post[i_doc] = deepcopy(lambda_doc)
                
            for l in range(max_lag+1):
                kappa[l] = sum([x[l] for x in ro])
            for l in range(max_lag+1):
                if kappa[l]==0 : kappa[l]=1e-5
                else :kappa[l]= kappa[l]
            kappa = [x/sum(kappa) for x in kappa]

            bound_post_corpus = 0
            for i_doc, (doc, t) in enumerate(zip(post_corpus, post_time)):

                term_1 = 0
                term_2 = 0
                term_3 = 0
                term_4 = 0
                term_5 = 0
                term_6 = 0
                term_7 = 0
                for n, (word, n_word) in enumerate(doc):
                    for k in range(N_topic_post):
                        if k < topic_nums_shared:
                            for l in range(max_lag+1):
                                term_1 += n_word * phi_post[i_doc][n][k]*psi_post[i_doc][n][k] * \
                                        ro[k][l]*log(beta_norm_shared[t+max_lag-l][word][k])
                        else:
                            term_1 += n_word * phi_post[i_doc][n][k]*psi_post[i_doc][n][k] * \
                                    log(beta_norm_post[t][word][k-topic_nums_shared])
                        term_2 += n_word * phi_post[i_doc][n][k]*log(theta_post[i_doc][k])
                        term_5 += n_word * phi_post[i_doc][n][k]*log(phi_post[i_doc][n][k])
                        if is_WordsSparse:
                            term_3 += n_word * (psi_post[i_doc][n][k]*log(lambda_post[i_doc][k]) + \
                                    (1-psi_post[i_doc][n][k])*log(1-lambda_post[i_doc][k]))
                            term_6 += n_word * (psi_post[i_doc][n][k]*log(psi_post[i_doc][n][k]) + \
                                    (1-psi_post[i_doc][n][k])*log(1-psi_post[i_doc][n][k]))
                for k in range(topic_nums_shared):
                    for l in range(max_lag+1):
                        term_4 += ro[k][l]*log(kappa[l])
                        term_7 += ro[k][l]*log(ro[k][l])

                bound_doc = term_1 + term_2 + term_3 + term_4 - term_5 - term_6 - term_7
                bound_post_corpus += bound_doc

        print('----n_iter_em:'+str(n_iter_em-1))
        print('----bound_post_corpus:'+str(bound_post_corpus))

        print('Next update beta')
        n_tw_all = [0] * N_topic
        n_t_all = [0] * N_topic
        for k in range(N_topic):

            if k < topic_nums_shared:
                n_tw_k = [[0] * len(beta_norm_shared) for _ in range(V)]
                for i_doc, (doc, t) in enumerate(zip(pre_corpus, pre_time)):
                    t += max_lag
                    for n, (word, n_word) in enumerate(doc):
                        n_tw_k[word][t] += n_word * phi_pre[i_doc][n][k]* psi_pre[i_doc][n][k]#K*V*T，但是是从max_lag到t+max_lag)

                for i_doc, (doc, t) in enumerate(zip(post_corpus, post_time)):
                    t += max_lag
                    for n, (word, n_word) in enumerate(doc):
                        for l in range(max_lag+1):
                            n_tw_k[word][t-l] += n_word * phi_post[i_doc][n][k] * ro[k][l]*\
                                                          psi_post[i_doc][n][k]

            elif k < N_topic_pre:
                n_tw_k = [[0] * len(beta_norm_pre) for _ in range(V)]#K*V*(0-T)
                for i_doc, (doc, t) in enumerate(zip(pre_corpus, pre_time)):
                    for n, (word, n_word) in enumerate(doc):
                        n_tw_k[word][t] += n_word * phi_pre[i_doc][n][k] * \
                                                    psi_pre[i_doc][n][k]

            else:
                n_tw_k = [[0] * len(beta_norm_post) for _ in range(V)]
                for i_doc, (doc, t) in enumerate(zip(post_corpus, post_time)):
                    for n, (word, n_word) in enumerate(doc):
                        n_tw_k[word][t] += n_word * phi_post[i_doc][n][k-topic_nums_pre] * \
                                            psi_post[i_doc][n][k-topic_nums_pre]
            #print('num_time_slice:',len(n_tw_k[0]))
            n_t_k = []
            for t in range(len(n_tw_k[0])):
                n_t_k.append(sum([x[t] for x in n_tw_k]))
            for t in range(len(n_tw_k[0])):
                prod = 1
                while sum([x[t]*prod for x in n_tw_k]) < 5*inp_doc_n_words_poisson*topic_nums_shared:
                    prod *= 10
                for word in range(len(n_tw_k)):
                    n_tw_k[word][t] = n_tw_k[word][t] * prod
            n_tw_all[k] = deepcopy(n_tw_k)#K*V*T 
            n_t_all[k] = deepcopy(n_t_k)

        # Update beta
        print('Updating beta. Start at '+time.asctime())
        for k in range(N_topic):
            
            print('topic '+str(k)+', starting.')

            if k < topic_nums_shared:
                time_slice_for_beta = [1] * len(beta_norm_shared)
            elif k < N_topic_pre:
                time_slice_for_beta = [1] * len(beta_norm_pre)
            else:
                time_slice_for_beta = [1] * len(beta_norm_post)
            dtm_model_1topic=SJDTM_para(id2word=id2word,time_slice=time_slice_for_beta,topic_suffstats=np.array(n_tw_all[k]),num_topics= 1,obs_variance=0.05,chain_variance=1.5)

            if k < topic_nums_shared:
                for t in range(len(beta_norm_shared)):
                    topwords_tmp = dtm_model_1topic.print_topic(0, t, top_terms=V)
                    for wordtoken,prob in topwords_tmp:
                        #word = id2word.token2id[wordtoken]
                        beta_norm_shared[t][int(wordtoken)][k] = prob
            elif k < N_topic_pre:
                for t in range(len(beta_norm_pre)):
                    topwords_tmp = dtm_model_1topic.print_topic(0, t, top_terms=V)
                    for wordtoken,prob in topwords_tmp:
                        #word = id2word.token2id[wordtoken]
                        beta_norm_pre[t][int(wordtoken)][k-topic_nums_shared] = prob
            else:
                for t in range(len(beta_norm_post)):
                    topwords_tmp = dtm_model_1topic.print_topic(0, t, top_terms=V)
                    for wordtoken,prob in topwords_tmp:
                        #word = id2word.token2id[wordtoken]
                        beta_norm_post[t][int(wordtoken)][k-N_topic_pre] = prob

        bound = bound_pre_corpus + bound_post_corpus
        
        print('----bound:'+str(bound))

    results = {}

    results['phi_pre'] = phi_pre
    results['psi_pre'] = psi_pre
    results['theta_pre'] = theta_pre
    results['lambda_pre'] = lambda_pre

    results['phi_post'] = phi_post
    results['psi_post'] = psi_post
    results['theta_post'] = theta_post
    results['lambda_post'] = lambda_post
    results['ro'] = ro
    results['kappa'] = kappa

    results['n_tw_all'] = n_tw_all
    results['beta_norm_shared'] = beta_norm_shared
    results['beta_norm_pre'] = beta_norm_pre
    results['beta_norm_post'] = beta_norm_post

    results['id2word'] = id2word
    results['pre_time_slice'] = pre_time_slice
    results['post_time_slice'] = post_time_slice

    return results




topics_lag_dic = [[1]]
inp_V = 500
inp_doc_n_words_poisson = 100
results=[]
for topic_nums in [1]:
    for topics_lag in topics_lag_dic:
        SKL_SJDTM=0
        PMI_SJDTM=0
        COH_SJDTM=0
        results_tmp = []
        for iteration in range(10):
            print('Starting at: '+time.asctime(time.localtime()))
            print('inp_V: '+str(inp_V))
            print('inp_doc_n_words_poisson: '+str(inp_doc_n_words_poisson))

            results_tmp = []
            print(str(topic_nums)+','+str(topics_lag)+','\
                  +',   '+time.asctime(time.localtime()))
            inp_V = inp_V

            inp_topic_nums = topic_nums
            inp_topics_lag = topics_lag
            sim_data= simu(inp_topic_nums, inp_topics_lag, inp_V, inp_doc_n_words_poisson)
            pre_corpus=sim_data[0]
            post_corpus=sim_data[1]
            pre_time_slice =sim_data[2]
            post_time_slice =sim_data[3]
            max_lag =sim_data[4]
            topic_nums_shared=sim_data[5]
            topic_nums_pre=sim_data[6]
            topic_nums_post=sim_data[7]
            id2word=sim_data[8]
            beta_norm_init=sim_data[9]
            num_topics = 2*inp_topic_nums
            min_iter=20
            max_iter=50
            min_iter_em=20
            max_iter_em=50
            chain_variance=1.5
            print('update start!')
            twtmodel1 = TWTopicModel(pre_corpus= pre_corpus,post_corpus = post_corpus,pre_time_slice=pre_time_slice,\
                             post_time_slice=post_time_slice ,max_lag=max_lag,topic_nums_shared= topic_nums_shared,\
                             topic_nums_pre=topic_nums_pre,topic_nums_post=topic_nums_post,id2word=id2word,\
                             beta_norm_init=beta_norm_init,min_iter=min_iter,max_iter=max_iter,min_iter_em=min_iter_em,max_iter_em=max_iter_em,\
                             chain_variance=chain_variance,s_KappaToRo=1)
            pre_dict = utils.dict_from_corpus(pre_corpus)
            pre_size=len(pre_dict)
            post_dict = utils.dict_from_corpus(post_corpus)
            post_size=len(post_dict)
            
            beta_norm_shared_SJDTM=copy.deepcopy(twtmodel1['beta_norm_shared'])
            beta_norm_pre_SJDTM=copy.deepcopy(twtmodel1['beta_norm_pre'])
            beta_norm_post_SJDTM=copy.deepcopy(twtmodel1['beta_norm_post'])
            theta_pre_SJDTM=copy.deepcopy(twtmodel1['theta_pre'])
            theta_post_SJDTM=copy.deepcopy(twtmodel1['theta_post'])
            for i in range(10):
                for j in range(inp_V):
                    for k in range(inp_topic_nums):
                        beta_norm_shared_SJDTM[(i+max_lag)][j].append(beta_norm_pre_SJDTM[i][j][k]) 
            beta_lead_SJDTM=beta_norm_shared_SJDTM[max_lag:]
            beta_norm_shared_SJDTM=copy.deepcopy(twtmodel1['beta_norm_shared'])
            ro=copy.deepcopy(twtmodel1['ro'])
            est_lag=ro[0].index(max(ro[0]))
            for i in range(10):
                for j in range(inp_V):
                    for k in range(inp_topic_nums):
                        beta_norm_shared_SJDTM[(i+max_lag-est_lag)][j].append(beta_norm_post_SJDTM[i][j][k]) 
            beta_lag_SJDTM=beta_norm_shared_SJDTM[(max_lag-est_lag):(10+max_lag-est_lag)]
                        
            pre_SKL_SJDTM=sKL_sum(beta_lead_SJDTM,num_topics,len(pre_time_slice))
            post_SKL_SJDTM=sKL_sum(beta_lag_SJDTM,num_topics,len(post_time_slice))
            pre_coherence_SJDTM = calculate_coherence(np.array(beta_lead_SJDTM), pre_corpus, num_topics,10,pre_time_slice)
            post_coherence_SJDTM = calculate_coherence(np.array(beta_lag_SJDTM), post_corpus, num_topics,10,post_time_slice)
            COH_SJDTM=pre_coherence_SJDTM+post_coherence_SJDTM
            ITER_SKL_SJDTM = pre_SKL_SJDTM+post_SKL_SJDTM
            SKL_SJDTM=ITER_SKL_SJDTM
            pre_PMI_SJDTM=calculate_pmi(np.array(beta_lead_SJDTM), pre_time_slice,10, pre_corpus)
            post_PMI_SJDTM=calculate_pmi(np.array(beta_lag_SJDTM), post_time_slice,10, post_corpus)
            PMI_SJDTM = pre_PMI_SJDTM+post_PMI_SJDTM
            print('SKL_SJDTM:',SKL_SJDTM)
            print('PMI_SJDTM:',PMI_SJDTM)
            print('COH_SJDTM:',COH_SJDTM)
            #results_tmp.append([SKL_SJDTM,PMI_SJDTM,COH_SJDTM,COH_DSTM])
        #results.append([topic_nums, topics_lag, results_tmp])
        
#str_results = [','.join([str(x_x) for x_x in x]) for x in results]
#with open('/home/luxiaoling/Yichao/datacode/real/Table2/result'+str(inp_V)+\
#          '_simW_'+str(inp_doc_n_words_poisson)+'1'+'.txt', 'w') as f:
#    f.writelines('\n'.join(str_results))
