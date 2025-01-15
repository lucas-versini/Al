import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.nn import GINConv
from torch_geometric.nn import EdgeConv

CONV = EdgeConv

from torch_geometric.nn import global_add_pool

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x_ = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x_
        adj = adj + torch.transpose(adj, 1, 2)

        return adj#, adj_non_binary

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(CONV(nn.Sequential(nn.Linear(2 * input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(CONV(nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(VariationalAutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
        self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def loss_function(self, data, beta=0.05):
        x_g  = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)

        # Some sort of contrastive learning
        sim_x_g = x_g[:, None, :] - x_g[None, :, :]
        sim_x_g = sim_x_g.norm(dim=-1)
        sim_x_g = F.softmax(sim_x_g, dim=-1)

        sim_stats = data.stats[:, None, :] - data.stats[None, :, :]
        sim_stats = sim_stats.norm(dim=-1)
        sim_stats = F.softmax(sim_stats, dim=-1)

        loss_sim = F.kl_div(sim_x_g, sim_stats, reduction = 'sum')
        adj = self.decoder(x_g)
        
        recon = F.binary_cross_entropy(adj, data.A, reduction = 'sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld + 10 * loss_sim

        return loss, recon, kld
