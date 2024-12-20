# -*- coding: utf-8 -*-
r"""
MixSGCL
################################################
"""
import torch
import torch.nn.functional as F
from recbole.model.general_recommender import LightGCN
from recbole.utils import InputType

class MixSGCL(LightGCN):
    def __init__(self, config, dataset):
        super(MixSGCL, self).__init__(config, dataset)
        self.temperature = config['temperature']
        
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
    
    def layer_mix(self, lightgcn_all_embeddings):
        mix_ratio = torch.rand(lightgcn_all_embeddings.size(0), lightgcn_all_embeddings.size(1), device='cuda')
        mix_ratio = (mix_ratio/mix_ratio.sum(dim=1).unsqueeze(dim=1)).unsqueeze(dim=1)
        lightgcn_all_embeddings_mix = torch.matmul(mix_ratio, lightgcn_all_embeddings).squeeze(dim=1)
        return lightgcn_all_embeddings_mix

    def pair_mix(self, u_embeddings, i_embeddings):
        mix_ratio = 0.5 * torch.rand(self.batch_size, device='cuda').unsqueeze(dim=1)
        u_embeddings_mixed = u_embeddings * mix_ratio + i_embeddings * (1-mix_ratio)
        u_embeddings_mixed = F.normalize(u_embeddings_mixed, dim=-1)
        i_embeddings_mixed = i_embeddings * mix_ratio + u_embeddings * (1-mix_ratio)
        i_embeddings_mixed = F.normalize(i_embeddings_mixed, dim=-1)
        return u_embeddings_mixed, i_embeddings_mixed
    
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
        
        u_emb, i_emb, u_emb_mix, i_emb_mix = self.forward_layer()
        u_embeddings, i_embeddings = u_emb[user], i_emb[pos_item]
        u_embeddings_mix, i_embeddings_mix = u_emb_mix[user], i_emb_mix[pos_item]
        
        u_embeddings, i_embeddings = F.normalize(u_embeddings, dim=-1), F.normalize(i_embeddings, dim=-1)
        u_embeddings_pairmix, i_embeddings_pairmix = self.pair_mix(u_embeddings, i_embeddings)
        u_embeddings_mix, i_embeddings_mix = F.normalize(u_embeddings_mix, dim=-1), F.normalize(i_embeddings_mix, dim=-1)

        u_embeddings_cat = torch.cat((u_embeddings, u_embeddings_mix, u_embeddings_pairmix), dim=0)
        i_embeddings_cat = torch.cat((i_embeddings, i_embeddings_mix, i_embeddings_pairmix), dim=0)
        
        pos_score = (u_embeddings_cat * i_embeddings_cat).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)

        ttl_u_score = torch.matmul(u_embeddings, u_embeddings.transpose(0, 1))
        ttl_u_score = torch.exp(ttl_u_score / self.temperature).sum(dim=1)
        ttl_i_score = torch.matmul(i_embeddings, i_embeddings.transpose(0, 1))
        ttl_i_score = torch.exp(ttl_i_score / self.temperature).sum(dim=1)

        ttl_umix_score = torch.matmul(u_embeddings_mix, u_embeddings_mix.transpose(0, 1))
        ttl_umix_score = torch.exp(ttl_umix_score / self.temperature).sum(dim=1)
        ttl_imix_score = torch.matmul(i_embeddings_mix, i_embeddings_mix.transpose(0, 1))
        ttl_imix_score = torch.exp(ttl_imix_score / self.temperature).sum(dim=1)

        ttl_pair_umix_score = torch.matmul(u_embeddings_pairmix, u_embeddings_pairmix.transpose(0, 1))
        ttl_pair_umix_score = torch.exp(ttl_pair_umix_score / self.temperature).sum(dim=1)
        ttl_pair_imix_score = torch.matmul(i_embeddings_pairmix, i_embeddings_pairmix.transpose(0, 1))
        ttl_pair_imix_score = torch.exp(ttl_pair_imix_score / self.temperature).sum(dim=1)
        
        ttl_score = ttl_u_score + ttl_i_score
        ttl_score2 = ttl_umix_score + ttl_imix_score
        ttl_score3 = ttl_pair_umix_score + ttl_pair_imix_score

        sup_cl_loss = -torch.log(pos_score / torch.cat((ttl_score, ttl_score2, ttl_score3), dim=0)).sum()
        return sup_cl_loss
