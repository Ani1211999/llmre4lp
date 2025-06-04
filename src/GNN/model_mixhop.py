import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn


# --- MixHop Layer ---
class MixHopLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_hops=2, combine='concat'):
        super(MixHopLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hops = num_hops
        self.combine = combine

        # One weight matrix per hop
        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_dim, out_dim)) for _ in range(num_hops)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, graph, feat):
        with graph.local_scope():
            results = []
            device = feat.device
            feat = feat.to(device)

            all_feats = [feat]

            # Precompute multi-hop features
            for k in range(1, self.num_hops):
                new_feat = feat
                for _ in range(k):  # Apply adjacency multiplication k times
                    norm = graph.ndata['norm']
                    new_feat = norm * new_feat
                    graph.ndata['h'] = new_feat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    new_feat = graph.ndata['h']
                    new_feat = norm * new_feat
                all_feats.append(new_feat)

            # Apply linear transformations
            for k in range(self.num_hops):
                h_k = torch.matmul(all_feats[k], self.weights[k])
                results.append(h_k)

            # Combine results
            if self.combine == 'concat':
                output = torch.cat(results, dim=1)
            elif self.combine == 'sum':
                output = torch.stack(results).sum(dim=0)
            else:
                raise ValueError("Combine method must be 'concat' or 'sum'")
                
            return output


# --- Full MixHop Model ---
class MixHopGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_hops=2, combine='concat', dropout=0.5):
        super(MixHopGCN, self).__init__()
        self.g = g
        self.dropout = dropout
        self.layer1 = MixHopLayer(in_dim, hidden_dim, num_hops=num_hops, combine=combine)
        self.layer2 = MixHopLayer(hidden_dim * num_hops if combine == 'concat' else hidden_dim,
                                  out_dim, num_hops=num_hops, combine=combine)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        pass  # Already initialized in MixHopLayer

    def forward(self, features):
        x = F.dropout(features, p=self.dropout, training=self.training)
        x = self.layer1(self.g, x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.layer2(self.g, x)
        return F.log_softmax(x, dim=1)