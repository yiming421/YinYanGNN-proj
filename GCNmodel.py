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
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--interval", default=50, type=int)
    parser.add_argument("--step_lr_decay", action='store_true', default=False)
    args = parser.parse_args()
    args.mlp = not args.nomlp
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
            h = self.conv2(g, h)
        return h

def train(model, g, train_pos_edge, optimizer, neg_sampler, pred):
    model.train()
    pred.train()

    dataloader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    total_loss = 0
    for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
        h = model(g, g.ndata['feat'])
        pos_edge = train_pos_edge[edge_index]
        neg_train_edge = neg_sampler(g, pos_edge.t()[0])
        neg_train_edge = torch.stack(neg_train_edge, dim=0)
        neg_train_edge = neg_train_edge.t()
        neg_edge = neg_train_edge
        print(pos_edge, neg_edge)
        pos_score = pred(h[pos_edge[:,0]], h[pos_edge[:,1]])
        neg_score = pred(h[neg_edge[:,0]], h[neg_edge[:,1]])
        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(g.ndata['feat'], 1.0)
        total_loss += loss.item()

    return total_loss / len(dataloader)

def test(model, g, pos_test_edge, neg_test_edge, evaluator, pred):
    model.eval()
    pred.eval()
    with torch.no_grad():
        h = model(g, g.ndata['feat'])
        pos_score = pred(h[pos_test_edge[0]], h[pos_test_edge[1]])
        neg_score = pred(h[neg_test_edge[0]], h[neg_test_edge[1]])
        results = {}
        results[args.metric] = evaluator.eval({
            'y_pred_pos': pos_score,
            'y_pred_neg': neg_score,
        })[args.metric]
    return results

def eval(model, g, pos_valid_edge, neg_valid_edge, evaluator, pred):
    model.eval()
    with torch.no_grad():
        h = model(g, g.ndata['feat'])
        pos_score = pred(h[pos_valid_edge[0]], h[pos_valid_edge[1]])
        neg_score = pred(h[neg_valid_edge[0]], h[neg_valid_edge[1]])
        results = {}
        results[args.metric] = evaluator.eval({
            'y_pred_pos': pos_score,
            'y_pred_neg': neg_score,
        })[args.metric]
    return results

# Load the dataset
dataset = DglLinkPropPredDataset(name=args.dataset)
split_edge = dataset.get_edge_split()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

graph = dataset[0]
graph = dgl.add_self_loop(graph)
graph = dgl.to_bidirected(graph, copy_ndata=True).to(device)

train_pos_edge = split_edge['train']['edge'].to(device)
valid_pos_edge = split_edge['valid']['edge'].to(device)
valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
test_pos_edge = split_edge['test']['edge'].to(device)
test_neg_edge = split_edge['test']['edge_neg'].to(device)

# Create negative samples for training
neg_sampler = GlobalUniform(args.num_neg)

pred = Hadamard_MLPPredictor(args.hidden).to(device)
embedding = torch.nn.Embedding(graph.num_nodes(), args.hidden)
torch.nn.init.xavier_uniform_(embedding.weight)
graph.ndata['feat'] = embedding.weight

model = GCN(in_feats=args.hidden, h_feats=args.hidden).to(device)
parameter = itertools.chain(model.parameters(), pred.parameters(), embedding.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)
evaluator = Evaluator(name=args.dataset)

for epoch in range(args.epochs):
    loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred)
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epoch, args.lr)
    valid_results = eval(model, graph, valid_pos_edge, valid_neg_edge, evaluator, pred)
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Validation hit: {valid_results[args.metric]:.4f}")

test_results = test(model, graph, test_pos_edge, test_neg_edge, evaluator, pred)
print(f"Test hit@20: {test_results[args.metric]:.4f}")
