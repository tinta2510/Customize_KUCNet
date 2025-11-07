import torch
import torch.nn as nn
from torchdrug.layers import functional
from torch_scatter import scatter
from ppr import get_ppr
import numpy as np
import time
class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_user, n_item, n_rel,n_node, ppr, K,act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.attn_dim = attn_dim
        self.act = act
        self.n_node = n_node
        self.ppr = ppr
        self.K = K
        # # n_rel*2 (inverse rels) + 1 (self-loop) + 2 (user-item, item-user)
        # self.rela_embed = nn.Embedding(2*n_rel+1+2, in_dim) # n_rel*2 (inverse rels) + 1 (self-loop) + 2 (user-item, item-user)

        # self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)  
        # self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        # self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        # self.w_alpha  = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)   
        
        # n_rel*2 (inverse rels) + 1 (self-loop) + 2 (user-item, item-user)
        self.rel_alpha = nn.Embedding(2*n_rel+1+2, 1)
        nn.init.ones_(self.rel_alpha.weight) 
        
    def forward(self, q_sub, q_rel, hidden, edges, nodes, id_layer, n_layer ,old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        # nodes:  [batch_idx, tail]
        l1 = len(edges)
        sampled_nodes_idx = torch.gt(nodes[:,1], -1) & torch.lt(nodes[:,1], self.n_node+1)
        
        
        if id_layer > 0 and id_layer < n_layer - 1: 
            max_ent_per_rel = self.K
            _, ind1 = torch.unique(edges[:,0:3],dim=0, sorted=True,return_inverse=True)
            _, ind2 = torch.sort(ind1)
            edges = edges[ind2]             # sort edges
            _, index, counts = torch.unique(edges[:,0:3], dim=0, return_inverse=True, return_counts=True)

            qu = q_sub[edges[:,0]].view(-1,1)
            qv = edges[:,3].view(-1,1)
            u_v = torch.cat((qu,qv),1)

            probs = self.ppr[u_v[:,0],u_v[:,1]].cuda()
            topk_value, topk_index = functional.variadic_topk(probs, counts, k=max_ent_per_rel)
            
            cnt_sum = torch.cumsum(counts,dim=0)
            cnt_sum[1:] = cnt_sum[:-1] + 0
            cnt_sum[0] = 0
            topk_index = topk_index + cnt_sum.unsqueeze(1)
            
            mask = topk_index.view(-1,1).squeeze()
            mask = torch.unique(mask)
            edges = edges[mask]

            nodes, tail_index = torch.unique(edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)
            edges = torch.cat([edges[:,0:5], tail_index.unsqueeze(1)], 1)
            
            head_index = edges[:,4]
            idd_mask = edges[:,2] == (self.n_rel*2 + 2) # mask self-loop
            _, old_idx = head_index[idd_mask].sort()
            old_nodes_new_idx = tail_index[idd_mask][old_idx]
        
        t_nodes = nodes
        
        # last layer
        if id_layer == n_layer - 1 :
            
            sampled_nodes_idx = torch.gt(nodes[:,1], self.n_user-1) & torch.lt(nodes[:,1], self.n_user+self.n_item)
            item_tail_index = torch.gt(edges[:,3], self.n_user-1) & torch.lt(edges[:,3], self.n_user+self.n_item)
            edges = edges[item_tail_index]
            nodes, tail_index = torch.unique(edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)
            edges = torch.cat([edges[:,0:5], tail_index.unsqueeze(1)], 1)

            final_nodes = nodes  
        else:
            final_nodes = torch.tensor([0])
        
        l2 = len(edges)
        sub = edges[:,4]  #old_idx  
        rel = edges[:,2]  #rela     
        obj = edges[:,5]  #new_idx  

        # hs = hidden[sub]         
        # hr = self.rela_embed(rel) 
        
        # r_idx = edges[:,0]
        # h_qr = self.rela_embed(q_rel)[r_idx]

        # message = hs + hr

        # alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))# TODO user embedding
        # message = alpha * message

        # message_agg = scatter(message, index=obj, dim=0, dim_size=nodes.size(0), reduce='sum') 
       
        # hidden_new = self.act(self.W_h(message_agg)) 
        
        # return hidden_new, t_nodes, final_nodes, old_nodes_new_idx, sampled_nodes_idx, alpha, edges
        
        hs = hidden[sub] # source node embeddings
        rela_alpha = self.rel_alpha(rel)
        message = rela_alpha * hs
        
        # --- Degree normalization ---
        ones = torch.ones_like(sub, dtype=message.dtype, device=message.device)
        deg_src = torch.zeros(nodes.size(0), dtype=message.dtype, device=message.device).index_add_(0, sub, ones)
        deg_dst = torch.zeros(nodes.size(0), dtype=message.dtype, device=message.device).index_add_(0, obj, ones)
        w = 1.0 / (torch.sqrt(deg_src[sub] * deg_dst[obj] + 1e-8)).unsqueeze(1)
        message = message * w

        message_agg = scatter(
            message,         # shape [E, d] — one per edge
            index=obj,       # shape [E]   — tells which node each edge ends at
            dim=0,           # aggregate along the "node" dimension
            dim_size=nodes.size(0),  # total number of nodes in this layer
            reduce='sum'     # sum all messages per node
        )

        # MAY ADD ACTIVATION
        # hidden_new = self.act(self.W_h(message_agg))
        hidden_new = message_agg
                
        return hidden_new, t_nodes, final_nodes, old_nodes_new_idx, sampled_nodes_idx, rela_alpha, edges

class KUCNet_trans(torch.nn.Module):
    def __init__(self, params, loader):
        super(KUCNet_trans, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_users = params.n_users
        self.n_items = params.n_items
        self.n_nodes = params.n_nodes
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x} 
        act = acts[params.act]
        self.K = params.K
        self.ppr = get_ppr(self.loader)
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim,self.n_users, self.n_items, self.n_rel, self.n_nodes, self.ppr, self.K,act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
       
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)         # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        
        # NEW: 4 learned type embeddings (target user, item, other user, entity)
        self.type_embed = nn.Embedding(4, self.hidden_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)

    def forward(self, subs, rels, mode='train'):
        """
        Args:
            subs (list of int): list of target user indices
            rels (list of int): list of target relation indices
        """
        n = len(subs)

        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()
       
        h0 = torch.zeros((1, n,self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), 
                           q_sub.unsqueeze(1)], 1) # 2-column tensor: (batch_idx, node_idx)
        # hidden = torch.zeros(n, self.hidden_dim).cuda() # Initialize node embeddings 
        hidden = self.type_embed.weight[0].unsqueeze(0).repeat(n, 1).contiguous().cuda()
        scores_all = []

        for i in range(self.n_layer):
            # nodes: next-layer nodes
            # edges: next-layer edges
            # old_nodes_new_idx: mapping from old node idx to new node idx (via self-loops)
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
            
            # hidden: current-layer node embeddings
            # nodes: current-layer nodes
            # final_nodes: final-layer nodes (for prediction)
            # old_nodes_new_idx: mapping from old node idx to new node idx (via self-loops) 
            # sampled_nodes_idx: mask for target items in final layer
            # alpha: attention scores (for analysis)
            # edges: after pruning
            hidden, nodes, final_nodes, old_nodes_new_idx, sampled_nodes_idx, alpha, edges = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes, i, self.n_layer,old_nodes_new_idx)
            
            if i == self.n_layer - 1:
                # Copy the old h0 into the positions that correspond to the old nodes’ new indices.
                h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)#此处删去一个cuda()
                h0 = h0[0, sampled_nodes_idx, :].unsqueeze(0) # only keep item-type nodes for final layer
            else:
                h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            
            hidden = self.dropout(hidden)
            # Pass hidden and h0 through the GRU gate to get updated hidden (h0) and output (hidden)
            hidden, h0 = self.gate (hidden.unsqueeze(0), h0)  
            hidden = hidden.squeeze(0)
            
            # NEW: add type bias each layer (residual inject of node-type embedding)
            # nodes: [batch_row, raw_node_id]
            raw = nodes[:, 1]
            batch = nodes[:, 0]
            # compute type ids
            is_user  = (raw < self.n_users)
            is_item  = (raw >= self.n_users) & (raw < self.n_users + self.n_items)
            # target user == q_sub[batch]
            is_target = is_user & (raw == q_sub[batch])
            type_ids = torch.where(is_target, torch.tensor(0, device=raw.device),
                          torch.where(is_item,   torch.tensor(1, device=raw.device),
                          torch.where(is_user,   torch.tensor(2, device=raw.device),
                                                  torch.tensor(3, device=raw.device))))
            hidden = hidden + self.type_embed(type_ids)           
 
        # Dot product to get final scores
        scores = self.W_final(hidden).squeeze(-1)   

        scores_all = torch.zeros((n, self.n_items)).cuda()  
        scores_all[[final_nodes[:,0], final_nodes[:,1]-self.n_users]] = scores   
        
        return scores_all     
 
