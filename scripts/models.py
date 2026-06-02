import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, SGConv, JumpingKnowledge, APPNP as APPNP_layer, GATConv, GATv2Conv, GINConv
from torch_sparse import matmul
from torch.nn import ModuleList, Dropout, Linear, Sequential, ReLU

class GCN(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout
			
    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.convs[-1](x, adj_t)
        return F.log_softmax(x, dim=1)


class SAGE(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
 
    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.convs[-1](x, adj_t)
        return F.log_softmax(x, dim=-1)
 
class GCNJKNet(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout, mode):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.jump = JumpingKnowledge(mode=mode, channels=hidden_channels, num_layers=num_layers)
        if mode == 'cat':
            self.lin = torch.nn.Linear(num_layers * hidden_channels, out_channels)
        else:
            self.lin = torch.nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x, adj_t):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
            xs += [x]

        x = self.jump(xs)
        x = self.lin(x)

        return F.log_softmax(x, dim=-1)


class SGC(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels):
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K=num_layers,
                            cached=True)

    def forward(self, x, adj_t):
        x = self.conv(x, adj_t)

        return F.log_softmax(x, dim=1)


class SIGN(torch.nn.Module):
    """
    The MLP part of the SIGN model.
    This model expects the node features to have already been pre-computed
    (i.e., concatenated features from multiple propagation steps).
    """
    def __init__(self, in_channels, out_channels, hidden_channels, dropout, num_mlp_layers):
        """
        Args:
            in_channels (int): The size of the pre-computed input features.
            out_channels (int): The size of the output channels.
            hidden_channels (int): The size of the hidden layer.
            dropout (float): The dropout rate.
            num_mlp_layers (int): The number of layers in the MLP.
        """
        super().__init__()
        
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_mlp_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x_dict, adj_t_dict):
        # This model ignores adj_t_dict because propagation is pre-computed.
        # It assumes the pre-computed features are stored in x_dict['paper'].
        x = x_dict['paper']
        
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        
        return {'paper': F.log_softmax(x, dim=-1)}


class APPNP(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
        # Fixed: Use 'K' instead of 'iterations'
        self.prop = APPNP_layer(K=num_layers, alpha=0.1)
        
        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, adj_t)
        return F.log_softmax(x, dim=1)


# This block is the core of the reversible architecture.
class ReversibleBlock(torch.nn.Module):
    def __init__(self, F_func, G_func):
        """
        Args:
            F_func (torch.nn.Module): The first function in the reversible block.
            G_func (torch.nn.Module): The second function in the reversible block.
        """
        super(ReversibleBlock, self).__init__()
        self.F_func = F_func
        self.G_func = G_func

    def forward(self, x1, x2, adj_t):
        """ The forward pass of the reversible block. """
        # The GNN layers (F and G) need the graph structure (adj_t).
        y1 = x1 + self.F_func(x2, adj_t)
        y2 = x2 + self.G_func(y1, adj_t)
        return y1, y2

    def reverse(self, y1, y2, adj_t):
        """ The reverse pass, used to recover activations and save memory. """
        x2 = y2 - self.G_func(y1, adj_t)
        x1 = y1 - self.F_func(x2, adj_t)
        return x1, x2

# This is the full, corrected RevGAT model.
class RevGAT(torch.nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, hidden_channels, dropout, heads=4):
        """
        Args:
            num_layers (int): The number of reversible blocks.
            in_channels (int): The dimension of input features.
            out_channels (int): The number of output classes.
            hidden_channels (int): The dimension of each of the two feature streams.
            dropout (float): The dropout rate.
            heads (int): The number of attention heads in each GATConv layer.
        """
        super(RevGAT, self).__init__()
        
        # The full hidden dimension is twice the size of one stream.
        self.full_hidden_channels = 2 * hidden_channels
        
        # Initial linear layer to project input features to the full hidden dimension.
        self.lin1 = Linear(in_channels, self.full_hidden_channels)
        
        # A list to hold all the reversible blocks.
        self.rev_blocks = ModuleList()
        for _ in range(num_layers):
            # Each block has two GATConv layers (F and G).
            # We use concat=False to average heads, keeping the dimensions consistent.
            F_func = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
            G_func = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
            self.rev_blocks.append(ReversibleBlock(F_func, G_func))
            
        # Final linear layer to project from the full hidden dimension to the output classes.
        self.lin2 = Linear(self.full_hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, adj_t):
        # 1. Apply initial projection and dropout.
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        
        # 2. Split the features into two streams, x1 and x2.
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        # 3. Sequentially apply the reversible blocks.
        for block in self.rev_blocks:
            x1, x2 = block(x1, x2, adj_t)
            
        # 4. Concatenate the two streams back together.
        x = torch.cat((x1, x2), dim=-1)
        
        # 5. Apply final projection to get class logits.
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)
        
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, n_heads, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # Use GATv2Conv for potentially better performance
        # First layer: maps input features to hidden dimensions with multiple heads
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=n_heads, dropout=dropout))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * n_heads, hidden_channels, heads=n_heads, dropout=dropout))
        
        # Final layer: maps to output classes with a single head (averaging)
        self.convs.append(GATv2Conv(hidden_channels * n_heads, out_channels, heads=1, concat=False, dropout=dropout))
        
        self.dropout = dropout

    # 1. CHANGE THIS LINE:
    def forward(self, x, adj_t): # Was 'edge_index'
        # Apply dropout to the input features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Process through all but the final layer
        for i, conv in enumerate(self.convs[:-1]):
            # 2. CHANGE THIS LINE:
            x = conv(x, adj_t) # Was 'edge_index'
            x = F.elu(x) # ELU is a common activation function for GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Process through the final layer
        # 3. AND CHANGE THIS LINE:
        x = self.convs[-1](x, adj_t) # Was 'edge_index'
        
        return F.log_softmax(x, dim=-1)
        
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Define the MLP for the GINConv layers
        mlp1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.convs.append(GINConv(mlp1, train_eps=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            mlp = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)