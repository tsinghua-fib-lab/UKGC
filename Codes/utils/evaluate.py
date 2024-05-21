from operator import pos
from numpy.core.fromnumeric import shape
from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
import random
from time import time
#
cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks, test_items_auc):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]
    item_score_auc = {}
    for i in test_items_auc:
        item_score_auc[i] = rating[i]
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc, logloss= get_auc_logloss(item_score_auc, user_pos_test)
    return r, auc, logloss

def get_auc_logloss(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    logloss = LogLoss(ground_truth=r, prediction=posterior)
    return auc, logloss

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks, test_items_auc):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]
    item_score_auc = {}
    for i in test_items_auc:
        item_score_auc[i] = rating[i]
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc, logloss= get_auc_logloss(item_score_auc, user_pos_test)
    return r, auc, logloss

def get_performance(user_pos_test, r, auc, Ks, logloss):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc, 'logloss':logloss}

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]
    user_neg_test = list()
    user_neg_test_auc = list()
    all_items = set(range(0, n_pois))

    test_items = list(all_items - set(training_items))
    neg_items = list(all_items - set(training_items) - set(user_pos_test))
    user_neg_test_auc = random.sample(neg_items, len(user_pos_test))
    test_items_auc = user_pos_test + user_neg_test_auc
    if args.test_flag == 'part':
        r, auc, logloss= ranklist_by_heapq(user_pos_test, test_items, rating, Ks, test_items_auc)
    else:
        r, auc, logloss= ranklist_by_sorted(user_pos_test, test_items, rating, Ks, test_items_auc)

    return get_performance(user_pos_test, r, auc, Ks, logloss)

def test(model, user_dict, n_params):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.,
              'logloss':0.}

    global n_users, n_pois
    n_pois = n_params['n_pois']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    p_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    entity_emb, user_emb, entity_emb_geo, user_emb_geo= model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start:end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_emb[user_batch]
        u_g_embeddings_geo = user_emb_geo[user_batch]

        if batch_test_flag:
            n_poi_batchs = n_pois // p_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_pois))

            p_count = 0
            for p_batch_id in range(n_poi_batchs):
                p_start = p_batch_id * p_batch_size
                p_end = min((p_batch_id + 1) * p_batch_size, n_pois)

                poi_batch = torch.LongTensor(np.array(range(p_start, p_end))).view(p_end-p_start).to(device)
                p_g_embeddings = entity_emb[poi_batch]
                p_g_embeddings_geo = entity_emb_geo[poi_batch]

                p_rate_batch = model.rating(u_g_embeddings, p_g_embeddings, u_g_embeddings_geo, p_g_embeddings_geo).detach().cpu()

                rate_batch[:, p_start : p_end] = p_rate_batch
                p_count += p_rate_batch.shape[1]
            
            assert p_count == n_pois
        
        else:
            poi_batch = torch.LongTensor(np.array(range(0, n_pois))).view(n_pois, -1).to(device)
            p_g_embeddings = entity_emb[poi_batch]
            p_g_embeddings_geo = entity_emb_geo[poi_batch]
            rate_batch = model.rating(u_g_embeddings, p_g_embeddings, u_g_embeddings_geo, p_g_embeddings_geo).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
            result['logloss'] += re['logloss']/n_test_users
    
    assert count == n_test_users
    pool.close()
    return result