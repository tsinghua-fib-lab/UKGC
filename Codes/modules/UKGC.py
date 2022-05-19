from operator import ne
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max

class Aggregator(nn.Module):
    def __init__(self, n_users, n_factors, agg_mode):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors
        self.agg_mode = agg_mode
    def forward(self, entity_emb, user_emb, latent_emb,
                edge_index, edge_type, interact_mat,
                weight, disen_weight_att):

        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        if self.agg_mode == 'mean':
            entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        elif self.agg_mode == 'max':
            entity_agg = scatter_max(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)[0]
        else:
            raise NotImplementedError
        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_emb.t())
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
                                weight).expand(n_users, n_factors, channel)
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        return entity_agg, user_agg

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_factors, n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1, agg_mode='mean'):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.agg_mode = agg_mode
        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        self.disen_weight_att = nn.Parameter(disen_weight_att)
        self.W = nn.ModuleList()
        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_factors=n_factors, agg_mode=self.agg_mode))
            self.W.append(nn.Linear(2*channel, channel))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    # def _cul_cor_pro(self):
    #     # disen_T: [num_factor, dimension]
    #     disen_T = self.disen_weight_att.t()
    #
    #     # normalized_disen_T: [num_factor, dimension]
    #     normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)
    #
    #     pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
    #     ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)
    #
    #     pos_scores = torch.exp(pos_scores / self.temperature)
    #     ttl_scores = torch.exp(ttl_scores / self.temperature)
    #
    #     mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
    #     return mi_score

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = self._cul_cor()
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, self.disen_weight_att)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
            # entity_res_emb = torch.cat((entity_res_emb, entity_emb), dim=1)
            # user_res_emb = torch.cat((user_res_emb, user_emb),dim=1)
            # entity_res_emb = nn.LeakyReLU()(self.W[i](entity_res_emb))
            # user_res_emb = nn.LeakyReLU()(self.W[i](user_res_emb))
            # entity_res_emb = F.normalize(entity_res_emb)
            # user_res_emb = F.normalize(user_res_emb)
        return entity_res_emb, user_res_emb, cor

class Recommender(nn.Module):
    def __init__(self, data_config, args_config, geo_graph, fun_graph, adj_mat_geo, adj_mat_fun):
        super(Recommender, self).__init__()
        self.n_users = data_config['n_users']
        self.n_pois = data_config['n_pois']
        self.n_geo_relations = data_config['n_geo_relations']
        self.n_geo_entities = data_config['n_geo_entities']
        self.n_geo_nodes = data_config['n_geo_nodes']
        self.n_fun_relations = data_config['n_fun_relations']
        self.n_fun_entities = data_config['n_fun_entities']
        self.n_fun_nodes = data_config['n_fun_nodes']
        
        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.alpha = args_config.alpha
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.rates = args_config.rates
        self.agg_mode = args_config.scatter
        self.cr = args_config.cr
        # self.alpha = nn.Parameter(self.alpha) # trainable alpha


        self.adj_mat_geo = adj_mat_geo
        self.adj_mat_fun = adj_mat_fun
        self.geo_graph = geo_graph
        self.fun_graph = fun_graph
        self.geo_edge_index, self.geo_edge_type = self._get_edges(geo_graph)
        self.fun_edge_index, self.fun_edge_type = self._get_edges(fun_graph)

        self._init_weight()
        self.all_embed_geo = nn.Parameter(self.all_embed_geo)
        self.latent_emb_geo = nn.Parameter(self.latent_emb_geo)
        self.all_embed_fun = nn.Parameter(self.all_embed_fun)
        self.latent_emb_fun = nn.Parameter(self.latent_emb_fun)

        

        self.gcn_geo, self.gcn_fun = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed_geo = initializer(torch.empty(self.n_geo_nodes, self.emb_size))
        self.latent_emb_geo = initializer(torch.empty(self.n_factors, self.emb_size))

        self.all_embed_fun = initializer(torch.empty(self.n_fun_nodes, self.emb_size))
        self.latent_emb_fun = initializer(torch.empty(self.n_factors, self.emb_size)) 
        self.interact_mat_geo = self._convert_sp_mat_to_sp_tensor(self.adj_mat_geo).to(self.device)
        self.interact_mat_fun = self._convert_sp_mat_to_sp_tensor(self.adj_mat_fun).to(self.device)

    def _init_model(self):
        gcn_geo = GraphConv(channel=self.emb_size,
                            n_hops=self.context_hops,
                            n_users=self.n_users,
                            n_relations=self.n_geo_relations,
                            n_factors=self.n_factors,
                            interact_mat=self.interact_mat_geo,
                            ind=self.ind,
                            node_dropout_rate=self.node_dropout_rate,
                            mess_dropout_rate=self.mess_dropout_rate,
                            agg_mode=self.agg_mode)
        gcn_fun = GraphConv(channel=self.emb_size,
                            n_hops=self.context_hops,
                            n_users=self.n_users,
                            n_relations=self.n_fun_relations,
                            n_factors=self.n_factors,
                            interact_mat=self.interact_mat_fun,
                            ind=self.ind,
                            node_dropout_rate=self.node_dropout_rate,
                            mess_dropout_rate=self.mess_dropout_rate,
                            agg_mode=self.agg_mode)
        return gcn_geo, gcn_fun

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)
    
    def forward(self, batch=None):
        user = batch['users']
        pos_poi = batch['pos_pois']
        neg_poi = batch['neg_pois']

        user_emb_geo = self.all_embed_geo[:self.n_users, :]
        user_emb_fun = self.all_embed_fun[:self.n_users, :]
        item_emb_geo = self.all_embed_geo[self.n_users:, :]
        item_emb_fun = self.all_embed_fun[self.n_users:, :]

        entity_gcn_emb_geo, user_gcn_emb_geo, cor_geo = self.gcn_geo(user_emb_geo, 
                                                                     item_emb_geo,
                                                                     self.latent_emb_geo,
                                                                     self.geo_edge_index,
                                                                     self.geo_edge_type,
                                                                     self.interact_mat_geo, 
                                                                     mess_dropout=self.mess_dropout,
                                                                     node_dropout=self.node_dropout)

        entity_gcn_emb_fun, user_gcn_emb_fun, cor_fun = self.gcn_fun(user_emb_fun, 
                                                                     item_emb_fun,
                                                                     self.latent_emb_fun,
                                                                     self.fun_edge_index,
                                                                     self.fun_edge_type,
                                                                     self.interact_mat_fun, 
                                                                     mess_dropout=self.mess_dropout,
                                                                     node_dropout=self.node_dropout)
        u_e_geo = user_gcn_emb_geo[user]
        u_e_fun = user_gcn_emb_fun[user]
        pos_e_geo = entity_gcn_emb_geo[pos_poi]
        pos_e_fun = entity_gcn_emb_fun[pos_poi]
        neg_e_geo = entity_gcn_emb_geo[neg_poi]
        neg_e_fun = entity_gcn_emb_fun[neg_poi]


        return self.create_bpr_loss(u_e_geo, u_e_fun, pos_e_geo, pos_e_fun, neg_e_geo, neg_e_fun, cor_geo, cor_fun)

    def generate(self):
        
        user_emb_geo = self.all_embed_geo[:self.n_users, :]
        user_emb_fun = self.all_embed_fun[:self.n_users, :]
        entity_emb_geo = self.all_embed_geo[self.n_users:, :]
        entity_emb_fun = self.all_embed_fun[self.n_users:, :]

        entity_emb_geo, user_emb_geo = self.gcn_geo(user_emb_geo,
                                                    entity_emb_geo,
                                                    self.latent_emb_geo,
                                                    self.geo_edge_index,
                                                    self.geo_edge_type,
                                                    self.interact_mat_geo,
                                                    mess_dropout=False, node_dropout=False)[:-1]

        entity_emb_fun, user_emb_fun = self.gcn_fun(user_emb_fun,
                                                    entity_emb_fun,
                                                    self.latent_emb_fun,
                                                    self.fun_edge_index,
                                                    self.fun_edge_type,
                                                    self.interact_mat_fun,
                                                    mess_dropout=False, node_dropout=False)[:-1]
        poi_emb = (entity_emb_geo[:self.n_pois, :] + entity_emb_fun[:self.n_pois, :]) * 0.5
        user_emb = (user_emb_geo + user_emb_fun) * 0.5

        return poi_emb, user_emb, entity_emb_geo[:self.n_pois, :], user_emb_geo


    
    def rating(self, u_g_embeddings, i_g_embeddings, u_g_embeddings_geo, i_g_embeddings_geo):
        
        pre_rating = torch.matmul(u_g_embeddings_geo, i_g_embeddings_geo.t())
        post_rating = torch.matmul(u_g_embeddings, i_g_embeddings.t())
        if self.rates == 0: # MUL-Sigmoid
            score = (post_rating - torch.mean(post_rating, 1, True)) * torch.sigmoid(pre_rating)
        elif self.rates == 1 or self.rates == 2 or self.rates == 3: # SUM-Linear, SUM-sigmoid, SUM-tanh
            score = post_rating - torch.mean(post_rating, 1, True)
        elif self.rates == 4:
            score = (post_rating - torch.mean(post_rating, 1, True)) * torch.tanh(pre_rating)
        else:
            raise NotImplementedError
        if self.cr:
            return torch.sigmoid(score)
        else:
            return torch.sigmoid(post_rating * torch.tanh(pre_rating))

    def create_bpr_loss(self, users_geo, users_fun, pos_geo, pos_fun, neg_geo, neg_fun, cor_geo, cor_fun):
        batch_size = users_geo.shape[0]
        pos_scores_pre = torch.sum(torch.mul(users_geo, pos_geo), axis=1)
        neg_scores_pre = torch.sum(torch.mul(users_geo, neg_geo), axis=1)
        pos_scores_fun = torch.sum(torch.mul(users_fun, pos_fun), axis=1)
        neg_scores_fun = torch.sum(torch.mul(users_fun, neg_fun), axis=1)

        users_post = (users_geo + users_fun) * 0.5
        pos_post = (pos_geo + pos_fun) * 0.5
        neg_post = (neg_geo + neg_fun) * 0.5

        pos_scores_post = torch.sum(torch.mul(users_post, pos_post), axis=1)
        neg_scores_post = torch.sum(torch.mul(users_post, neg_post), axis=1)
        if self.rates == 0: # MUL-Sigmoid
            pos_scores = pos_scores_post * torch.sigmoid(pos_scores_pre)
            neg_scores = neg_scores_post * torch.sigmoid(neg_scores_pre)
        elif self.rates == 1: # SUM-Linear
            pos_scores = pos_scores_post + pos_scores_pre
            neg_scores = neg_scores_post + neg_scores_pre
        elif self.rates == 2: # SUM-Sigmoid
            pos_scores = pos_scores_post + torch.sigmoid(pos_scores_pre)
            neg_scores = neg_scores_post + torch.sigmoid(neg_scores_pre)
        elif self.rates == 3: # SUM-tanh
            pos_scores = pos_scores_post + torch.tanh(pos_scores_pre)
            neg_scores = neg_scores_post + torch.tanh(neg_scores_pre)
        elif self.rates == 4: # MUL-tanh
            pos_scores = pos_scores_post * torch.tanh(pos_scores_pre)
            neg_scores = neg_scores_post * torch.tanh(neg_scores_pre)
        else:
            raise NotImplementedError

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        mf_loss_pre = -1 * torch.mean(nn.LogSigmoid()(pos_scores_pre - neg_scores_pre))

        regularizer = (torch.norm(users_post) ** 2 + torch.norm(pos_post) ** 2 + torch.norm(neg_post) ** 2) / 2
        regularize_pre = (torch.norm(users_geo) ** 2 + torch.norm(pos_geo) ** 2 + torch.norm(neg_geo) ** 2) / 2

        emb_loss = self.decay * regularizer / batch_size
        emb_loss_pre = self.decay * regularize_pre / batch_size
        cor_loss = self.sim_decay * (cor_geo + cor_fun)

        return mf_loss + emb_loss + cor_loss + self.alpha * (mf_loss_pre + emb_loss_pre), cor_geo + cor_fun
