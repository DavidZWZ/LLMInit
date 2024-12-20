# -*- coding: utf-8 -*-
r"""
MixSGCL
################################################
"""

import numpy as np
import torch
import torch.nn.functional as F
from recbole.model.general_recommender import LightGCN
from recbole.utils import InputType
import torch.nn as nn

class ContSGCL(LightGCN):
    def __init__(self, config, dataset):
        super(ContSGCL, self).__init__(config, dataset)
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.temperature = config['temperature']
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
    
    def forward(self):
        all_embs = self.get_ego_embeddings()
        embeddings_list = [all_embs]

        for layer_idx in range(self.n_layers):
            all_embs = torch.sparse.mm(self.norm_adj_matrix, all_embs)
            embeddings_list.append(all_embs)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)

        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings
    
    def forward_layer(self):
        all_embs = self.get_ego_embeddings()
        embeddings_list = [all_embs]

        for layer_idx in range(self.n_layers):
            all_embs = torch.sparse.mm(self.norm_adj_matrix, all_embs)
            embeddings_list.append(all_embs)
       
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings_mix = self.layer_mix(lightgcn_all_embeddings)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        user_all_embeddings_mix, item_all_embeddings_mix = torch.split(lightgcn_all_embeddings_mix, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, user_all_embeddings_mix, item_all_embeddings_mix
    
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        self.batch_size = user.size(0)
        
        u_emb, i_emb = self.forward()
        u_embeddings, i_embeddings = u_emb[user], i_emb[pos_item]
        
        u_embeddings, i_embeddings = F.normalize(u_embeddings, dim=-1), F.normalize(i_embeddings, dim=-1)

        pos_score = (u_embeddings * i_embeddings).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)

        ttl_u_score = torch.matmul(u_embeddings, u_embeddings.transpose(0, 1))
        ttl_u_score = torch.exp(ttl_u_score / self.temperature).sum(dim=1)
        ttl_i_score = torch.matmul(i_embeddings, i_embeddings.transpose(0, 1))
        ttl_i_score = torch.exp(ttl_i_score / self.temperature).sum(dim=1)
        
        ttl_score = ttl_u_score + ttl_i_score
        sup_cl_loss = -torch.log(pos_score / ttl_score).sum()
        return sup_cl_loss
