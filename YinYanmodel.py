import itertools
import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from dgl.nn.pytorch import GraphConv
from dgl.dataloading.negative_sampler import GlobalUniform
from torch.utils.data import DataLoader
import tqdm
import argparse

def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ogbl-citation2', choices=['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation2', 'ogbl-vessel'], type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--prop_step", default=8, type=int)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--batch_size", default=8192, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--interval", default=50, type=int)
    parser.add_argument("--step_lr_decay", action='store_true', default=False)
    parser.add_argument("--metric", default='hits@20', type=str)
    parser.add_argument("--filter_year", default=2010, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--relu", action='store_true', default=False)
    parser.add_argument("--model", default='GCN', choices=['GCN', 'YinYanGNN'], type=str)
    parser.add_argument("K", default=1, type=int)
    args = parser.parse_args()
    return args

args = parse()

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def forward(self, x_i, x_j):
        x = x_i * x_j
        x = self.W1(x)
        x = F.relu(x)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.W2(x)
        return x.squeeze()
    
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        for i in range(1, args.prop_step):
            if args.relu:
                h = F.relu(h)
            h = self.conv2(g, h)
        return h
    
class YinYanGNN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(YinYanGNN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        self.conv = GraphConv(h_feats, h_feats, weight=False, bias=False)

    def forward(self, g, neg_g, in_feat):
        ori_h = self.mlp(in_feat)
        for i in range(args.prop_step):
            if args.relu:
                h = F.relu(h)
            h = self.conv(g, h) - self.conv(neg_g, h) + ori_h
        return h

def train(model, g, neg_g, train_pos_edge, optimizer, neg_sampler, pred):
    model.train()
    pred.train()

    dataloader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    total_loss = 0

    for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
        if args.model == 'YinYanGNN':
            h = model(g, neg_g, g.ndata['feat'])
        elif args.model == 'GCN':
            h = model(g, g.ndata['feat'])
        else:
            raise NotImplementedError
        pos_edge = train_pos_edge[edge_index]
        neg_train_edge = neg_sampler(g, pos_edge.t()[0])
        neg_train_edge = torch.stack(neg_train_edge, dim=0)
        neg_train_edge = neg_train_edge.t()
        neg_edge = neg_train_edge
        pos_score = pred(h[pos_edge[:,0]], h[pos_edge[:,1]])
        neg_score = pred(h[neg_edge[:,0]], h[neg_edge[:,1]])
        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.dataset == 'ogbl-collab' or args.dataset == 'ogbl-citation2':
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        elif args.dataset == 'ogbl-ddi':
            torch.nn.utils.clip_grad_norm_(g.ndata['feat'], 1.0)
        total_loss += loss.item()

    return total_loss / len(dataloader)

def test(model, g, neg_g, pos_test_edge, neg_test_edge, evaluator, pred):
    model.train()
    pred.train()

    with torch.no_grad():
        if args.model == 'YinYanGNN':
            h = model(g, neg_g, g.ndata['feat'])
        elif args.model == 'GCN':
            h = model(g, g.ndata['feat'])
        else:
            raise NotImplementedError
        if args.dataset == 'ogbl-citation2':
            dataloader = DataLoader(range(pos_test_edge.size(0)), args.batch_size, shuffle=True)
            pos_score = []
            for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
                pos_edge = pos_test_edge[edge_index]
                pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
                pos_score.append(pos_pred)
            pos_score = torch.cat(pos_score, dim=0)
            dataloader = DataLoader(range(neg_test_edge.size(0)), args.batch_size, shuffle=True)
            neg_score = []
            for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
                neg_edge = neg_test_edge[edge_index]
                neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
                neg_score.append(neg_pred)
            neg_score = torch.cat(neg_score, dim=0)
            neg_score = neg_score.view(-1, 1000)
            results = {}
            results[args.metric] = evaluator.eval({
                'y_pred_pos': pos_score,
                'y_pred_neg': neg_score,
            })['mrr_list'].mean().item()
        else:
            pos_score = pred(h[pos_test_edge[:, 0]], h[pos_test_edge[:, 1]])
            neg_score = pred(h[neg_test_edge[:, 0]], h[neg_test_edge[:, 1]])
            results = {}
            results[args.metric] = evaluator.eval({
                'y_pred_pos': pos_score,
                'y_pred_neg': neg_score,
            })[args.metric]
    return results

def eval(model, g, neg_g, pos_valid_edge, neg_valid_edge, evaluator, pred):
    model.eval()
    pred.eval()

    with torch.no_grad():
        if args.model == 'YinYanGNN':
            h = model(g, neg_g, g.ndata['feat'])
        elif args.model == 'GCN':
            h = model(g, g.ndata['feat'])
        else:
            raise NotImplementedError
        if args.dataset == 'ogbl-citation2':
            dataloader = DataLoader(range(pos_valid_edge.size(0)), args.batch_size, shuffle=True)
            pos_score = []
            for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
                pos_edge = pos_valid_edge[edge_index]
                pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
                pos_score.append(pos_pred)
            pos_score = torch.cat(pos_score, dim=0)
            dataloader = DataLoader(range(neg_valid_edge.size(0)), args.batch_size, shuffle=True)
            neg_score = []
            for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
                neg_edge = neg_valid_edge[edge_index]
                neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
                neg_score.append(neg_pred)
            neg_score = torch.cat(neg_score, dim=0)
            neg_score = neg_score.view(-1, 1000)
            results = {}
            results[args.metric] = evaluator.eval({
                'y_pred_pos': pos_score,
                'y_pred_neg': neg_score,
            })['mrr_list'].mean().item()
        else:         
            pos_score = pred(h[pos_valid_edge[:, 0]], h[pos_valid_edge[:, 1]])
            neg_score = pred(h[neg_valid_edge[:, 0]], h[neg_valid_edge[:, 1]])
            results = {}
            results[args.metric] = evaluator.eval({
                'y_pred_pos': pos_score,
                'y_pred_neg': neg_score,
            })[args.metric]
    return results

# Load the dataset
dataset = DglLinkPropPredDataset(name=args.dataset)
split_edge = dataset.get_edge_split()

device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

graph = dataset[0]
graph = dgl.add_self_loop(graph)
graph = dgl.to_bidirected(graph, copy_ndata=True).to(device)

if args.dataset =="ogbl-citation2":
    for name in ['train','valid','test']:
        u=split_edge[name]["source_node"]
        v=split_edge[name]["target_node"]
        split_edge[name]['edge']=torch.stack((u,v),dim=0).t()
    for name in ['valid','test']:
        u=split_edge[name]["source_node"].repeat(1, 1000).view(-1)
        v=split_edge[name]["target_node_neg"].view(-1)
        split_edge[name]['edge_neg']=torch.stack((u,v),dim=0).t()

if dataset.name == 'ogbl-collab':
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= args.filter_year).nonzero(as_tuple=False), (-1, ))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]

train_pos_edge = split_edge['train']['edge'].to(device)
valid_pos_edge = split_edge['valid']['edge'].to(device)
valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
test_pos_edge = split_edge['test']['edge'].to(device)
test_neg_edge = split_edge['test']['edge_neg'].to(device)

# Create negative samples for training
neg_sampler = GlobalUniform(args.num_neg)

pred = Hadamard_MLPPredictor(args.hidden).to(device)

if args.dataset == 'ogbl-ddi' or dataset.name == 'ogbl-ppa':
    embedding = torch.nn.Embedding(graph.num_nodes(), args.hidden).to(device)
    torch.nn.init.xavier_uniform_(embedding.weight)
    graph.ndata['feat'] = embedding.weight

if args.model == 'GCN':
    model = GCN(in_feats=graph.ndata['feat'].shape[1], h_feats=args.hidden).to(device)
elif args.model == 'YinYanGNN':
    model = YinYanGNN(in_feats=graph.ndata['feat'].shape[1], h_feats=args.hidden).to(device)

if args.dataset == 'ogbl-ddi' or dataset.name == 'ogbl-ppa':
    parameter = itertools.chain(model.parameters(), pred.parameters(), embedding.parameters())
else:
    parameter = itertools.chain(model.parameters(), pred.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)
evaluator = Evaluator(name=args.dataset)

best_val = 0
final_test_result = None
best_epoch = 0


for epoch in range(args.epochs):
    if args.model == 'YinYanGNN':
        g = g.to('cpu')
        neg_prop_edge = neg_sampler(g, torch.LongTensor(range(g.num_edges())))
        neg_g = dgl.graph(neg_prop_edge, num_nodes=g.num_nodes())
        neg_g = dgl.to_bidirected(neg_g, copy_ndata=True)
        neg_g = neg_g.to(device)
        g = g.to(device)
    elif args.model == 'GCN':
        g = g.to(device)
    else:
        raise NotImplementedError
    loss = train(model, graph, neg_g, train_pos_edge, optimizer, neg_sampler, pred)
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
    valid_results = eval(model, graph, neg_g, valid_pos_edge, valid_neg_edge, evaluator, pred)
    if args.dataset == 'ogbl-collab':
        graph_t = graph.clone()
        u, v = valid_pos_edge.t()
        graph_t.add_edges(u, v)
        graph_t.add_edges(v, u)
    else:
        graph_t = graph
    test_results = test(model, graph_t, neg_g, test_pos_edge, test_neg_edge, evaluator, pred)
    if valid_results[args.metric] > best_val:
        best_val = valid_results[args.metric]
        best_epoch = epoch
        final_test_result = test_results
    if args.dataset == 'ogbl-collab':
        if epoch - best_epoch >= 200:
            break
    elif args.dataset == 'ogbl-citation2' or args.dataset == 'ogbl-ppa':
        if epoch - best_epoch >= 100:
            break
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Validation hit: {valid_results[args.metric]:.4f}, Test hit: {test_results[args.metric]:.4f}")

print(f"Test hit: {final_test_result[args.metric]:.4f}")
