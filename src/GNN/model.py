import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn


class FALayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

        # REGISTERED PARAMETERS
        self.yes_weight = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))
        self.no_weight = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        yes_no = torch.tanh((edges.data['yes_no'] * self.yes_weight) +
                            ((1 - edges.data['yes_no']) * self.no_weight))
        e = ((g + yes_no) / 2) * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))
        return self.g.ndata['z']


class FAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)

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