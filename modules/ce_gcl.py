

import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, layer_norm=False, batch_norm=False):
        super(Encoder, self).__init__()

        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.mlp = torch.nn.Linear(in_channels, hidden_channels)
        self.ln = torch.nn.LayerNorm(hidden_channels)
        self.bn = torch.nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        x = self.mlp(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.layer_norm:
            x = self.ln(x)
        if self.batch_norm:
            x = self.bn(x)
        return x


class CE_GCL(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, alpha, temperature, dropout, layer_norm=False, batch_norm=False,
                        use_global_topology=True, global_topo_weight=0.30, global_topo_decay=0.60, global_topo_hops=3):
        super(CE_GCL, self).__init__()

        self.alpha = alpha
        self.temperature = temperature
        self._combination = False

        self.enc = Encoder(in_channels, hidden_channels, dropout, layer_norm, batch_norm)

        self.use_global_topology = use_global_topology 
        self.global_topo_weight = global_topo_weight
        self.global_topo_hops = global_topo_hops
        self.global_topo_decay = global_topo_decay
        self.eps = 1e-12

    def _new_row_normalize_dense(self, A: torch.Tensor) -> torch.Tensor:
        deg = A.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return A / deg

    def _new_build_topology_priors(self, A_P: torch.Tensor):
        A = A_P.float()
        A = 0.5 *(A + A.t())
        A = torch.relu(A)

        num_par = A.size(0)
        I = torch.eye(num_par, device=A.device, dtype=A.dtype)

        local_topo = self._new_row_normalize_dense(A + I)

        if (not self.use_global_topology) or self.global_topo_hops <= 1:
            print("warning global_topo_hops is too short ")
            return local_topo, local_topo

        diffuse = local_topo.clone()
        power = local_topo.clone()

        coeff_sum = 1.0 
        coeff = 1.0 

        for _ in range(2, self.global_topo_hops + 1):
            power = torch.mm(power, local_topo)
            coeff = coeff * self.global_topo_decay
            diffuse = diffuse + coeff * power
            coeff_sum += coeff

        diffuse = diffuse / coeff_sum

        local_mask = (A > 0).float()
        local_mask.fill_diagonal_(1.0)

        global_topo = diffuse * (1.0 - local_mask)

        row_sum = global_topo.sum(dim=1, keepdim=True)
        global_topo = torch.where(
            row_sum > 0,
            global_topo / row_sum.clamp_min(self.eps),
            local_topo
        )
        return local_topo, global_topo


    def get_loss(self, x, node_to_par, P, A_P):
        q_n = F.sigmoid(x)
        k_n = torch.spmm(P.T, q_n)
        s_n = torch.sum(q_n, dim=0)
        pos_n = torch.sum(q_n * k_n[node_to_par], dim=1)
        neg_n = torch.sum(q_n * s_n, dim=1)

        k_p_local = F.normalize(torch.spmm(P.T, x), p=2, dim=1)

        local_topo, global_topo = self._new_build_topology_priors(A_P)

        k_p_global = F.normalize(torch.mm(global_topo, k_p_local), p=2, dim=1)

        gamma = self.global_topo_weight
        k_p = F.normalize(
            (1.0 - gamma) * k_p_local + gamma * k_p_global,
            p = 2,
            dim = 1
        )

        sim_p = torch.exp(torch.mm(k_p, k_p.t())/ self.temperature)

        pos_p_local = torch.sum(local_topo * sim_p, dim=1)

        pos_p_global = torch.sum(global_topo * sim_p, dim = 1)

        pos_p = (1.0 - gamma) * pos_p_local + gamma * pos_p_global
        neg_p = torch.sum(sim_p, dim=1)

        pos_score = self.alpha * pos_n + (1.0 - self.alpha) * pos_p[node_to_par]
        neg_score = self.alpha * neg_n + (1.0 - self.alpha) * neg_p[node_to_par]

        loss = - torch.log(pos_score.clamp_min(self.eps)) + \
         torch.log(neg_score.clamp_min(self.eps))
        return loss.mean() 


    def forward(self, x, node_to_par, P, A_P, graph=None):
        x = self.enc(x)
        loss = self.get_loss(x, node_to_par, P, A_P)
        return loss

    def get_embed(self, x):
        x = self.enc(x)
        return x

