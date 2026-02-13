### implementation by adapting the basic classes from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/basic_gnn.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.models.basic_gnn import BasicGNN

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
from torch_geometric.nn.conv import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    NNConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
    GENConv,
)
from torch_geometric.nn.models import MLP, DeepGCNLayer
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)


__all__ = ['GCN', 'DeeperGCN', 'GIN', 'GINE', 'GAT', 'MPNN', 'GraphSAGE', 'PNA', 'AtomEncoder', 'BondEncoder']

pooling_options = {"sum":global_add_pool,
                "mean":global_mean_pool,
                "max": global_max_pool,
                }
        
class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        num_tasks (int): Number of tasks for multi-label classification. 
        emb_dim (int): Size of embedding for each atom and/or bond and graph.
        num_layers (int): Number of message passing layers.
        atom_features_dims:(list or None, optional): Size of each category feature for atoms(nodes). if a list is not provided, the atom_features_dims will be filled by funtion `get_atom_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        graph_pooling (str): Pooling function to generate whole-graph embeddings. (:obj:`"mean"`, :obj:`"sum"`, :obj:`"max"`, :obj:`"attention"`, :obj:`"set2set"`). (default: :obj:`"mean"`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.5`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"BatchNorm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def __init__(self,
        num_tasks: int,
        emb_dim: int,
        num_layers: int,
        atom_features_dims:Union[list, None]  = None,
        graph_pooling = "mean",
        dropout: float = 0.5,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = 'BatchNorm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = "last",
        **kwargs,):
        super().__init__(emb_dim, emb_dim, num_layers, emb_dim,
                         dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk, **kwargs)

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        
        self.atom_encoder = AtomEncoder(emb_dim, atom_features_dims)
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling in pooling_options:
            self.pool = pooling_options[self.graph_pooling]
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, 2*in_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)
    
    def forward(self, x, edge_index, batch):
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input category features for atoms(nodes).
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        
        ### computing input node embedding
        x = self.atom_encoder(x)
        
        node_emb = super().forward(x, edge_index)
        graph_emb = self.pool(node_emb, batch)

        return self.graph_pred_linear(graph_emb)  
    
class GINE(BasicGNN):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper so that it is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        num_tasks (int): Number of tasks for multi-label classification. 
        emb_dim (int): Size of embedding for each atom(node) and/or bond(edge) and graph.
        num_layers (int): Number of message passing layers.
        atom_features_dims:(list or None, optional): Size of each category feature for atoms(nodes). if a list is not provided, the atom_features_dims will be filled by funtion `get_atom_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        bond_features_dims:(list or None, optional): Size of each category feature for bonds(edges). if a list is not provided, the bond_features_dims will be filled by funtion `get_bond_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        graph_pooling (str): Pooling function to generate whole-graph embeddings. (:obj:`"mean"`, :obj:`"sum"`, :obj:`"max"`, :obj:`"attention"`, :obj:`"set2set"`). (default: :obj:`"mean"`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.5`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"BatchNorm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINEConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def __init__(self,
        num_tasks: int,
        emb_dim: int,
        num_layers: int,
        atom_features_dims:Union[list, None]  = None,
        bond_features_dims:Union[list, None]  = None,
        graph_pooling = "mean",
        dropout: float = 0.5,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = 'BatchNorm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = "last",
        **kwargs,):
        super().__init__(emb_dim, emb_dim, num_layers, emb_dim,
                         dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk, **kwargs)

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        
        self.atom_encoder = AtomEncoder(emb_dim, atom_features_dims)
        self.bond_encoder = BondEncoder(emb_dim, bond_features_dims)
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling in pooling_options:
            self.pool = pooling_options[self.graph_pooling]
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, 2*in_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, **kwargs)
    
    def forward(self, x, edge_index, edge_attr, batch):
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input category features for atoms(nodes).
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The input category features for bonds(edges).
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        
        ### computing input node and edge embedding
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        
        node_emb = super().forward(x, edge_index, edge_attr = edge_attr)
        graph_emb = self.pool(node_emb, batch)

        return self.graph_pred_linear(graph_emb)

class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        num_tasks (int): Number of tasks for multi-label classification. 
        emb_dim (int): Size of embedding for each atom and/or bond and graph.
        num_layers (int): Number of message passing layers.
        atom_features_dims:(list or None, optional): Size of each category feature for atoms(nodes). if a list is not provided, the `atom_features_dims` will be filled by funtion `get_atom_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        graph_pooling (str): Pooling function to generate whole-graph embeddings. (:obj:`"mean"`, :obj:`"sum"`, :obj:`"max"`, :obj:`"attention"`, :obj:`"set2set"`). (default: :obj:`"mean"`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.5`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"BatchNorm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def __init__(self,
        num_tasks: int,
        emb_dim: int,
        num_layers: int,
        atom_features_dims:Union[list, None]  = None,
        graph_pooling = "mean",
        dropout: float = 0.5,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = 'BatchNorm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = "last",
        **kwargs,):
        super().__init__(emb_dim, emb_dim, num_layers, emb_dim,
                         dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk, **kwargs)

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        
        self.atom_encoder = AtomEncoder(emb_dim, atom_features_dims)
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling in pooling_options:
            self.pool = pooling_options[self.graph_pooling]
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        return GCNConv(in_channels, out_channels, **kwargs)
    
    def forward(self, x, edge_index, batch):
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input category features for atoms(nodes).
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        
        ### computing input node and edge embedding
        x = self.atom_encoder(x)
               
        node_emb = super().forward(x, edge_index)
        graph_emb = self.pool(node_emb, batch)

        return self.graph_pred_linear(graph_emb)

class DeeperGCN(torch.nn.Module):
    r"""The Graph Neural Network from the `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_ paper, using the
    :class:`~torch_geometric.nn.GENConv` operator for message passing. 
    The skip connection operations from the
    `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_ and `"All You Need to Train Deeper
    GCNs" <https://arxiv.org/abs/2006.07739>`_ papers, including the pre-activation residual
    connection (:obj:`"res+"`), the residual connection (:obj:`"res"`),
    the dense connection (:obj:`"dense"`) and no connections (:obj:`"plain"`).

    Args:
        num_tasks (int): Number of tasks for multi-label classification. 
        emb_dim (int): Size of embedding for each atom(node) and/or bond(edge) and graph.
        num_layers (int): Number of message passing layers.
        atom_features_dims:(list or None, optional): Size of each category feature for atoms(nodes). if a list is not provided, the atom_features_dims will be filled by funtion `get_atom_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        bond_features_dims:(list or None, optional): Size of each category feature for bonds(edges). if a list is not provided, the bond_features_dims will be filled by funtion `get_bond_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        graph_pooling (str): Pooling function to generate whole-graph embeddings. (:obj:`"mean"`, :obj:`"sum"`, :obj:`"max"`, :obj:`"attention"`, :obj:`"set2set"`). (default: :obj:`"mean"`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        aggr (str or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
            (:obj:`"softmax"`, :obj:`"powermean"`, :obj:`"add"`, :obj:`"mean"`,
            :obj:`max`). (default: :obj:`"softmax"`)
        t (float, optional): Initial inverse temperature for softmax
            aggregation. (default: :obj:`0.1`)
        learn_t (bool, optional): If set to :obj:`True`, will learn the value
            :obj:`t` for softmax aggregation dynamically.
            (default: :obj:`True`)
        p (float, optional): Initial power for power mean aggregation.
            (default: :obj:`1.0`)
        learn_p (bool, optional): If set to :obj:`True`, will learn the value
            :obj:`p` for power mean aggregation dynamically.
            (default: :obj:`False`)
        block (str, optional): The skip connection operation to use
            (:obj:`"res+"`, :obj:`"res"`, :obj:`"dense"` or :obj:`"plain"`).
            (default: :obj:`"res+"`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"BatchNorm"`)
    """
    def __init__(self, 
        num_tasks: int,
        emb_dim: int,
        num_layers: int,
        atom_features_dims:Union[list, None]  = None,
        bond_features_dims:Union[list, None]  = None,   
        graph_pooling = "mean",    
        dropout: float = 0.0,  
        aggr: Optional[Union[str, List[str], Aggregation]] = 'softmax',
        t: float = 0.1,
        learn_t: bool = True,
        p: float = 1.0,
        learn_p: bool = False,
        block: str = 'res+',  
        act: Union[str, Callable, None] = "relu",    
        norm: Union[str, Callable, None] = 'BatchNorm', 
        ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.dropout = dropout
        
        self.atom_encoder = AtomEncoder(emb_dim, atom_features_dims)
        self.bond_encoder = BondEncoder(emb_dim, bond_features_dims)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(emb_dim, emb_dim, aggr=aggr,
                           t = t, learn_t=learn_t, p=p, learn_p=learn_p, num_layers=2, norm='batch')

            norm_layer = normalization_resolver(norm, emb_dim)
            act_layer = activation_resolver(act)
            
            layer = DeepGCNLayer(conv, norm_layer, act_layer, block=block, dropout=dropout,)
            self.layers.append(layer)
     
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling in pooling_options:
            self.pool = pooling_options[self.graph_pooling]
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, x, edge_index, edge_attr, batch):
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input category features for atoms(nodes).
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The input category features for bonds(edges).
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        ### computing input node and edge embedding
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        
        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        # x = self.layers[0].act(self.layers[0].norm(x))
        x = self.layers[0].norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        graph_emb = self.pool(x, batch)
        
        return self.graph_pred_linear(graph_emb)
    
class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        num_tasks (int): Number of tasks for multi-label classification. 
        emb_dim (int): Size of embedding for each atom(node) and/or bond(edge) and graph.
        num_layers (int): Number of message passing layers.
        v2 (bool, optional): If set to :obj:`True`, will make use of: class:`~torch_geometric.nn.conv.GATv2Conv` rather than :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        atom_features_dims:(list or None, optional): Size of each category feature for atoms(nodes). if a list is not provided, the atom_features_dims will be filled by funtion `get_atom_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        bond_features_dims:(list or None, optional): Size of each category feature for bonds(edges). if a list is not provided, the bond_features_dims will be filled by funtion `get_bond_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        graph_pooling (str): Pooling function to generate whole-graph embeddings. (:obj:`"mean"`, :obj:`"sum"`, :obj:`"max"`, :obj:`"attention"`, :obj:`"set2set"`). (default: :obj:`"mean"`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"BatchNorm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]
    
    def __init__(self,
        num_tasks: int,
        emb_dim: int,
        num_layers: int,
        v2 = False,
        atom_features_dims:Union[list, None]  = None,
        bond_features_dims:Union[list, None]  = None,
        graph_pooling = "mean",
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "elu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = 'BatchNorm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = "last",
        **kwargs,):
        super().__init__(emb_dim, emb_dim, num_layers, emb_dim, 
                         dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk,
                         v2 = v2, edge_dim=emb_dim, **kwargs)

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        
        self.atom_encoder = AtomEncoder(emb_dim, atom_features_dims)
        self.bond_encoder = BondEncoder(emb_dim, bond_features_dims)
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling in pooling_options:
            self.pool = pooling_options[self.graph_pooling]
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
            
    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, **kwargs)

    def forward(self, x, edge_index, edge_attr, batch):
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input category features for atoms(nodes).
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The input category features for bonds(edges).
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        
        ### computing input node and edge embedding
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        
        node_emb = super().forward(x, edge_index, edge_attr = edge_attr)
        graph_emb = self.pool(node_emb, batch)

        return self.graph_pred_linear(graph_emb)

class MPNN(BasicGNN):
    r"""The Graph Neural Network from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper, using the
    :class:`~torch_geometric.nn.NNConv` operator for message passing.

    Args:
        num_tasks (int): Number of tasks for multi-label classification. 
        emb_dim (int): Size of embedding for each atom(node) and/or bond(edge) and graph.
        num_layers (int): Number of message passing layers.
        atom_features_dims:(list or None, optional): Size of each category feature for atoms(nodes). if a list is not provided, the atom_features_dims will be filled by funtion `get_atom_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        bond_features_dims:(list or None, optional): Size of each category feature for bonds(edges). if a list is not provided, the bond_features_dims will be filled by funtion `get_bond_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        graph_pooling (str): Pooling function to generate whole-graph embeddings. (:obj:`"mean"`, :obj:`"sum"`, :obj:`"max"`, :obj:`"attention"`, :obj:`"set2set"`). (default: :obj:`"mean"`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.2`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"BatchNorm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.NNConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def __init__(self,
        num_tasks: int,
        emb_dim: int,
        num_layers: int,
        atom_features_dims:Union[list, None]  = None,
        bond_features_dims:Union[list, None]  = None,
        graph_pooling = "mean",
        dropout: float = 0.2,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = 'BatchNorm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = "last",
        **kwargs,):
        super().__init__(emb_dim, emb_dim, num_layers, emb_dim,
                         dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk, **kwargs)

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        
        self.atom_encoder = AtomEncoder(emb_dim, atom_features_dims)
        self.bond_encoder = BondEncoder(emb_dim, bond_features_dims)
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling in pooling_options:
            self.pool = pooling_options[self.graph_pooling]
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, in_channels, in_channels*out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return NNConv(in_channels, out_channels, mlp, **kwargs)
    
    def forward(self, x, edge_index, edge_attr, batch):
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input category features for atoms(nodes).
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The input category features for bonds(edges).
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        
        ### computing input node and edge embedding
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        
        node_emb = super().forward(x, edge_index, edge_attr = edge_attr)
        graph_emb = self.pool(node_emb, batch)

        return self.graph_pred_linear(graph_emb)

class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        num_tasks (int): Number of tasks for multi-label classification. 
        emb_dim (int): Size of embedding for each atom and/or bond and graph.
        num_layers (int): Number of message passing layers.
        atom_features_dims:(list or None, optional): Size of each category feature for atoms(nodes). if a list is not provided, the atom_features_dims will be filled by funtion `get_atom_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        graph_pooling (str): Pooling function to generate whole-graph embeddings. (:obj:`"mean"`, :obj:`"sum"`, :obj:`"max"`, :obj:`"attention"`, :obj:`"set2set"`). (default: :obj:`"mean"`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.2`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"BatchNorm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def __init__(self,
        num_tasks: int,
        emb_dim: int,
        num_layers: int,
        atom_features_dims:Union[list, None]  = None,
        graph_pooling = "mean",
        dropout: float = 0.2,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = 'BatchNorm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = "last",
        **kwargs,):
        super().__init__(emb_dim, emb_dim, num_layers, emb_dim,
                         dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk, **kwargs)

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.atom_encoder = AtomEncoder(emb_dim, atom_features_dims)
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling in pooling_options:
            self.pool = pooling_options[self.graph_pooling]
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
            
    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)

    def forward(self, x, edge_index, batch):
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input category features for atoms(nodes).
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        
        ### computing input node embedding
        x = self.atom_encoder(x)
        
        node_emb = super().forward(x, edge_index)
        graph_emb = self.pool(node_emb, batch)

        return self.graph_pred_linear(graph_emb)

class PNA(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        num_tasks (int): Number of tasks for multi-label classification. 
        emb_dim (int): Size of embedding for each atom(node) and/or bond(edge) and graph.
        num_layers (int): Number of message passing layers.
        atom_features_dims:(list or None, optional): Size of each category feature for atoms(nodes). if a list is not provided, the atom_features_dims will be filled by funtion `get_atom_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        bond_features_dims:(list or None, optional): Size of each category feature for bonds(edges). if a list is not provided, the bond_features_dims will be filled by funtion `get_bond_feature_dims() <https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py>`_.
        graph_pooling (str): Pooling function to generate whole-graph embeddings. (:obj:`"mean"`, :obj:`"sum"`, :obj:`"max"`, :obj:`"attention"`, :obj:`"set2set"`). (default: :obj:`"mean"`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.2`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"BatchNorm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def __init__(self,
        num_tasks: int,
        emb_dim: int,
        num_layers: int,
        atom_features_dims:Union[list, None]  = None,
        bond_features_dims:Union[list, None]  = None,
        graph_pooling = "mean",
        dropout: float = 0.2,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = 'BatchNorm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = "last",
        **kwargs,):
        super().__init__(emb_dim, emb_dim, num_layers, emb_dim,
                         dropout, act, act_first, act_kwargs, norm, norm_kwargs, jk,edge_dim=emb_dim, **kwargs)

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        
        self.atom_encoder = AtomEncoder(emb_dim, atom_features_dims)
        self.bond_encoder = BondEncoder(emb_dim, bond_features_dims)
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling in pooling_options:
            self.pool = pooling_options[self.graph_pooling]
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
            
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        return PNAConv(in_channels, out_channels, aggregators, scalers, **kwargs)

    def forward(self, x, edge_index, edge_attr, batch):
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input category features for atoms(nodes).
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The input category features for bonds(edges).
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        
        ### computing input node and edge embedding
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        
        node_emb = super().forward(x, edge_index, edge_attr = edge_attr)
        graph_emb = self.pool(node_emb, batch)

        return self.graph_pred_linear(graph_emb)   
    

class AtomEncoder(torch.nn.Module):
    r"""Convert discrete atom(node) features to embeddings"""
    def __init__(self, emb_dim, atom_features_dims):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()

        if atom_features_dims is not None:
            full_atom_feature_dims = atom_features_dims
        else:
            full_atom_feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    r"""Convert discrete bond(edge) features to embeddings"""
    def __init__(self, emb_dim, bond_features_dims):
        super(BondEncoder, self).__init__()

        if bond_features_dims is not None:
            full_bond_feature_dims = bond_features_dims
        else:
            full_bond_feature_dims = get_bond_feature_dims()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   