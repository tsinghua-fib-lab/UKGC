import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import random

import torch
import numpy as np

from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from modules.UKGC import Recommender
from utils.evaluate import test
from utils.helper import early_stopping
#
n_users = 0
n_pois = 0
n_geo_entities = 0
n_geo_nodes = 0
n_geo_relations = 0
n_fun_entities = 0
n_fun_nodes = 0
n_fun_relations = 0

def get_feed_dict(train_entity_pairs, start, end, train_user_set):

    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_pois, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_pois'] = entity_pairs[:, 1]
    feed_dict['neg_pois'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, geo_graph, fun_graph, mat_list_geo, mat_list_fun = load_data(args)
    adj_mat_list_geo, norm_mat_list_geo, mean_mat_list_geo = mat_list_geo
    adj_mat_list_fun, norm_mat_list_fun, mean_mat_list_fun = mat_list_fun

    n_users = n_params['n_users']
    n_pois = n_params['n_pois']
    n_geo_entities = n_params['n_geo_entities']
    n_geo_relations = n_params['n_geo_relations']
    n_geo_nodes = n_params['n_geo_nodes']
    n_fun_entities = n_params['n_fun_entities']
    n_fun_relations = n_params['n_fun_relations']
    n_fun_nodes = n_params['n_fun_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, geo_graph, fun_graph, mean_mat_list_geo[0], mean_mat_list_fun[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    print('Start training ...')
    for epoch in range(args.epoch):
        """training CF"""
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()

        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'])
            batch_loss, batch_cor = model(batch)

            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            cor_loss += batch_cor
            s += args.batch_size

        train_e_t = time()
        
        if epoch % 10 == 9 or epoch == 1:
            """testing"""
            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio", "AUC", "LogLoss"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio'], ret['auc'], ret['logloss']]
            )
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['auc'], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if ret['auc'] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + '.ckpt')

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            if (type(cor_loss) == int):
                print('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f' % (train_e_t - train_s_t, epoch, loss.item(), cor_loss))
            else:
                print('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f' % (train_e_t - train_s_t, epoch, loss.item(), cor_loss.item()))

    print('early stopping at %d, AUC:%.4f' % (epoch, cur_best_pre_0))       
