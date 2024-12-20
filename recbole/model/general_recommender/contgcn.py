# -*- coding: utf-8 -*-
# @Time   : 2020/8/31
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

# UPDATE:
# @Time   : 2020/9/16, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com

r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class ContGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ContGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config["embedding_size"]  # int type:the embedding size of lightGCN
        self.llm_latent_dim = config["llm_embedding_size"]
        self.emb_selection = config["emb_selection"]
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config["reg_weight"]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]
            
        # define node degree
        data = torch.tensor(self.interaction_matrix.data).cuda()
        row = torch.tensor(self.interaction_matrix.row, dtype=torch.long).cuda()
        col = torch.tensor(self.interaction_matrix.col, dtype=torch.long).cuda()
        indices = torch.stack([row, col])
        self.sparse_A =  torch.sparse_coo_tensor(indices, data, self.interaction_matrix.shape).cuda()
        self.user_degrees = torch.sum(self.sparse_A, dim=1).to_dense()
        self.user_degrees[0] = 1
        # breakpoint()
        self.item_degrees = torch.sum(self.sparse_A, dim=0).to_dense()
        self.item_degrees[0] = 1
        
        # # Create the embedding layer
        if self.emb_selection:
            if self.emb_selection == 'rand':
                self.sample_strategy = self.random_sample()
            elif self.emb_selection == 'uni':
                self.sample_strategy = self.even_sample()
            elif self.emb_selection == 'var':
                self.sample_strategy = self.var_sample()
            # item embedding
            self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)
            # Samping 
            sampled_indices = self.sample_strategy     
            self.item_embedding.weight = nn.Parameter(self.item_embedding_context[:, sampled_indices])
            self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        else: 
            self.item_embedding = torch.nn.Embedding(self.n_items, self.llm_latent_dim)
            self.item_embedding.weight = torch.nn.Parameter(self.item_embedding_context)
            self.user_embedding = torch.nn.Embedding(self.n_users, self.llm_latent_dim)
            
        # define the user embedding layer
        self.user_embedding.weight = torch.nn.Parameter(torch.sparse.mm(self.sparse_A, \
                                            self.item_embedding.weight)/self.user_degrees.unsqueeze(dim=1))
        
        # define loss
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        
        # parameters initialization
        # self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def even_sample(self):
        # Evenly spaced sampling indices
        step_size = self.llm_latent_dim // self.latent_dim
        sampled_indices = torch.arange(0, self.llm_latent_dim, step_size)[:self.latent_dim]
        return sampled_indices
    
    def random_sample(self):
        # Uniformly sample indices from LLM latent dimensions
        sampled_indices = torch.randint(0, self.llm_latent_dim, (self.latent_dim,))
        return sampled_indices

    def var_sample(self):
        # step 0: select popular items
        ratio = torch.tensor(0.2)
        _ , popular_indices = torch.topk(self.item_degrees, (self.item_degrees.size(0)*ratio).to(torch.int))
        item_embedding = self.item_embedding_context[popular_indices,:]
        # Step 1: Compute variance of each dimension
        variances = torch.var(item_embedding, dim=0)  # Size: (N,)
        # Step 2: Select top-K dimensions based on variance
        _, selected_indices = torch.topk(variances, self.latent_dim)
        sorted_indices, indices = torch.sort(selected_indices, descending=False)
        return sorted_indices

    def LLMinit(self, train_data, valid_data):
            pass
    
    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL
    
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    
    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        
        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

