import numpy as np
import numpy.random as rd
import torch
import torch.optim as optim
import torch.nn as nn
import networkx as nx

from deptree import *
from torch.nn.functional import pad
from torch.utils import data
from torch.utils.data import DataLoader,SequentialSampler
from math import sqrt
from tqdm import tqdm
from random import sample,shuffle

class DependencyDataset(data.Dataset):
    """
    A representation of the DepBank for efficient processing.
    This is a sorted dataset
    """
    UNK_WORD     = '<unk>'
    PAD_WORD     = '<pad>'
    PAD_WORD_IDX = -1
    ROOT         = '<root>'
    ROOT_GOV_IDX = -1
    
    def __init__(self,filename,use_vocab=None,use_labels=None):
        istream       = open(filename)
        self.treelist = []
        tree = DepGraph.read_tree(istream) 
        while tree:
            if len(tree) < 15: #problem of memory explosion later with very long sentences.
                self.treelist.append(tree)
                print(tree)
                print('',flush=True)
                break
            tree = DepGraph.read_tree(istream)             
        istream.close()
        shuffle(self.treelist)
        #self.treelist.sort(key=lambda x:len(x)) # we do not make real batches later
        if use_vocab:
            self.itos = use_vocab
            self.stoi = {token:idx for idx,token in enumerate(self.itos)}
        else:
            self.init_vocab(self.treelist)
            
        if use_labels:
            self.itolab = use_labels
            self.labtoi = {label:idx for idx,label in enumerate(self.itolab)}
        else:
            self.init_labels(self.treelist)
            
        self.preprocess_edges()
        self.preprocess_labels()

    def shuffle(self):
        self.treelist.shuffle()
        self.treelist.sort(key=lambda x:len(x))
        self.preprocess_edges()
        self.preprocess_labels()
    
    def init_vocab(self,treelist):
        """
        Extracts the set of tokens found in the data and orders it 
        """
        vocab = set([ word  for tree in treelist for word in tree.words ])
        vocab.update([DependencyDataset.ROOT,DependencyDataset.UNK_WORD,DependencyDataset.PAD_WORD])

        self.itos = list(vocab)
        self.stoi = {token:idx for idx,token in enumerate(self.itos)}
        DependencyDataset.PAD_WORD_IDX = self.stoi[DependencyDataset.PAD_WORD]

    def init_labels(self,treelist):
        labels      = set([ lbl for tree in treelist for (gov,lbl,dep) in tree.get_all_edges()])
        self.itolab = list(labels)
        self.labtoi = {label:idx for idx,label in enumerate(self.itolab)}

    def preprocess_edges(self):
        """
        Encodes the dataset and makes it ready for processing.
        This is the encoding for the edge prediction task 
        """ 
        self.xdep     = [ ] 
        self.refidxes = [ ] 
        
        for tree in self.treelist:
            word_seq      = [DependencyDataset.ROOT] + tree.words
            depword_idxes = [self.stoi.get(tok,self.stoi[DependencyDataset.UNK_WORD]) for tok in word_seq]
            gov_idxes     = [DependencyDataset.ROOT_GOV_IDX] + DependencyDataset.oracle_ancestors(tree)
            self.xdep.append(depword_idxes)
            self.refidxes.append(gov_idxes)
            
    def preprocess_labels(self): 
        """
        Encodes the dataset and makes it ready for processing.
        This is the encoding for the edge prediction task 
        """
        self.refdeps   = [ ]
        self.refgovs   = [ ]
        self.reflabels = [ ]
        self.tokens    = [ ]
        for tree in self.treelist:
            #print(tree) # +1 comes from the dummy root padding
            self.refgovs.append(   [gov+1 for (gov,lbl,dep) in tree.get_all_edges()] )
            self.refdeps.append(   [dep+1 for (gov,lbl,dep) in tree.get_all_edges()] )
            self.reflabels.append( [self.labtoi[lbl] for (gov,lbl,dep) in tree.get_all_edges() if lbl in self.labtoi] ) #in case the label is unknown, skip it ! 
            self.tokens.append(tree.words)
            
    def __len__(self):      
        return len(self.treelist)
    
    def __getitem__(self,idx):
        return {'xdep':self.xdep[idx],\
                'refidx':self.refidxes[idx],\
                'refdeps':self.refdeps[idx],\
                'refgovs':self.refgovs[idx],\
                'reflabels':self.reflabels[idx],\
                'wordlist':self.tokens[idx]}

    @staticmethod
    def oracle_ancestors(depgraph):
        """
        Returns a list where each element list[i] is the index of
        the position of the governor of the word at position i.
        Returns:
        a tensor of size N.
        """
        N         = len(depgraph)
        edges     = depgraph.get_all_edges()
        rev_edges = dict([(dep,gov) for (gov,label,dep) in edges])
        return [rev_edges.get(idx,-1)+1 for idx in range(N)]           #<--- check +1 addition here !!

def dep_collate_fn(batch):
    """
    That's the collate function for batching edges
    """
    return [ ((torch.tensor(elt['xdep']), torch.tensor(elt['refidx'])),(torch.tensor(elt['refdeps']), torch.tensor(elt['refgovs']),torch.tensor(elt['reflabels'])),elt['wordlist']) for elt in batch ]


class MLP(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(MLP, self).__init__()
        self.Wdown = nn.Linear(input_size,hidden_size)
        self.Wup   = nn.Linear(hidden_size,output_size)
        self.g     = nn.ReLU() 
        
    def forward(self,input):
        return self.Wup(self.g(self.Wdown(input)))
     
class Biaffine(nn.Module):
    """
    Biaffine module whose implementation works efficiently on GPU too
    """
    def __init__(self,input_size,label_size= 2 ):
        super(Biaffine, self).__init__()
        sqrtk  = sqrt(input_size)
        self.B = nn.Bilinear(input_size,input_size,label_size,bias=False)
        self.W = nn.Parameter( -sqrtk + torch.rand(label_size,input_size*2)/(2*sqrtk))      
        self.b = nn.Parameter( -sqrtk + torch.rand(1)-0.5)        
     
    def forward(self,xdep,xhead):
        """
        Performs the forward pass on a batch of tokens.
        This computes a score for each couple (xdep,xhead) or a vector
        of label scores for each such couple.

        Each argument as an expected input of dimension [batch,embedding_size]
        The returned results are scores of dimension    [batch,nlabels]
        """
        assert(len(xdep.shape)  == 2)
        assert(len(xhead.shape) == 2)
        assert(xdep.shape ==  xhead.shape)
        batch,emb = xdep.shape

        dephead = torch.cat([xdep,xhead],dim=1).t()
        return self.B(xdep,xhead) + (self.W @ dephead).t() + self.b
             
class GraphParser(nn.Module):
    
    def __init__(self,vocab,labels,word_embedding_size,lstm_hidden,arc_mlp_hidden,lab_mlp_hidden):
        
        super(GraphParser, self).__init__()
        #self.code_vocab(vocab)
        #self.code_labels(labels)
        self.allocate(word_embedding_size,len(vocab),len(labels),lstm_hidden,arc_mlp_hidden,lab_mlp_hidden)
        
    def allocate(self,word_embedding_size,vocab_size,label_size,lstm_hidden,arc_mlp_hidden,lab_mlp_hidden):
        self.E              = nn.Embedding(vocab_size,word_embedding_size)
        self.edge_biaffine  = Biaffine(lstm_hidden,1)
        self.label_biaffine = Biaffine(lstm_hidden,label_size)
        self.head_arc       = MLP(lstm_hidden*2,arc_mlp_hidden,lstm_hidden)
        self.dep_arc        = MLP(lstm_hidden*2,arc_mlp_hidden,lstm_hidden)
        self.head_lab       = MLP(lstm_hidden*2,lab_mlp_hidden,lstm_hidden)
        self.dep_lab        = MLP(lstm_hidden*2,lab_mlp_hidden,lstm_hidden)
        self.rnn            = nn.LSTM(word_embedding_size,lstm_hidden,bidirectional=True)
         
    def forward_edges(self,dep_embeddings,head_embeddings):
        """
        Performs predictions for pointing to roots 
        """
        return self.edgeB(dep_embeddings.t(),head_embeddings.t())

    def forward_labels(self,dep_embeddings,head_embeddings):
        """
        Performs label predictions
        """
        return self.labB(dep_embeddings,head_embeddings) 
    
    def train(self,trainset,devset,epochs):
        
        print("N =",len(trainset))
        edge_loss_fn  = nn.CrossEntropyLoss(reduction = 'sum',ignore_index=DependencyDataset.ROOT_GOV_IDX) #ignores the dummy root index
        label_loss_fn = nn.CrossEntropyLoss(reduction = 'sum') 
        optimizer     = optim.Adam( self.parameters() )

        for ep in range(epochs):
            eNLL,eN,lNLL,lN = 0,0,0,0
            dataloader = DataLoader(trainset, batch_size=16,shuffle=False, num_workers=4,collate_fn=dep_collate_fn)
            for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader)): 
                for (edgedata,labeldata,tok_sequence) in batch:
                    optimizer.zero_grad()  
                    word_emb_idxes,ref_gov_idxes = edgedata[0].to(xdevice),edgedata[1].to(xdevice)
                    N = len(word_emb_idxes)
                    #print('edges',list(zip(ref_gov_idxes.cpu().numpy(),range(N))))
                    #1. Run LSTM on raw input and get word embeddings
                    embeddings        = self.E(word_emb_idxes).unsqueeze(dim=0)
                    input_seq,end     = self.rnn(embeddings)
                    input_seq         = input_seq.squeeze(dim=0)
                    #2.  Compute edge attention from flat matrix representation
                    deps_embeddings   = torch.repeat_interleave(input_seq,repeats=N,dim=0)
                    gov_embeddings    = input_seq.repeat(N,1)
                    attention_scores  = self.edge_biaffine(self.dep_arc(deps_embeddings),self.head_arc(gov_embeddings),)
                    attention_matrix  = attention_scores.view(N,N)
                    #3. Compute loss and backprop for edges
                    eloss = edge_loss_fn(attention_matrix,ref_gov_idxes)
                    eloss.backward(retain_graph=True)
                    eN   += N
                    eNLL += eloss.item()
                    #4. Compute loss and backprop for labels
                    ref_deps_idxes,ref_gov_idxes,ref_labels = labeldata[0].to(xdevice),labeldata[1].to(xdevice),labeldata[2].to(xdevice)
                    deps_embeddings   = input_seq[ref_deps_idxes]
                    gov_embeddings    = input_seq[ref_gov_idxes]
                    #print('labels',list(zip(ref_gov_idxes.cpu().numpy(),ref_deps_idxes.cpu().numpy(),ref_labels.cpu().numpy())))
                    #print(trainset.itolab)
                    label_predictions = self.label_biaffine(self.dep_lab(deps_embeddings),self.head_lab(gov_embeddings))
                    lloss  = label_loss_fn(label_predictions,ref_labels)
                    lloss.backward( )
                    lN   += len(ref_labels)
                    lNLL += lloss.item()
                    optimizer.step( )
            print("epoch",ep)
            print('TRAIN: mean NLL(edges)',eNLL/eN,'mean NLL(labels)',lNLL/lN)
            deveNLL,devlNLL = self.eval(devset)
            print('DEV  : mean NLL(edges)',deveNLL,'mean NLL(labels)',devlNLL)

    def eval(self,dataset):
        
        edge_loss_fn  = nn.CrossEntropyLoss(reduction = 'sum',ignore_index=DependencyDataset.ROOT_GOV_IDX) #ignores the dummy root index
        label_loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
        
        with torch.no_grad():
            eNLL,eN,lNLL,lN = 0,0,0,0
            dataloader = DataLoader(dataset, batch_size=32,shuffle=False, num_workers=4,collate_fn=dep_collate_fn,sampler=SequentialSampler(dataset))
            for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader)): 
                for (edgedata,labeldata,tok_sequence) in batch:
                    word_emb_idxes,ref_gov_idxes = edgedata[0].to(xdevice),edgedata[1].to(xdevice)
                    N = len(word_emb_idxes)
                    #1. Run LSTM on raw input and get word embeddings
                    embeddings        = self.E(word_emb_idxes).unsqueeze(dim=0)
                    input_seq,end     = self.rnn(embeddings)
                    input_seq         = input_seq.squeeze(dim=0)
                    #2.  Compute edge attention from flat matrix representation
                    deps_embeddings   = torch.repeat_interleave(input_seq,repeats=N,dim=0)
                    gov_embeddings    = input_seq.repeat(N,1)
                    attention_scores  = self.edge_biaffine(self.dep_arc(deps_embeddings),self.head_arc(gov_embeddings))
                    attention_matrix  = attention_scores.view(N,N)
                    #3. Compute loss for edges
                    eloss = edge_loss_fn(attention_matrix,ref_gov_idxes)
                    eN   += N
                    eNLL += eloss.item()
                    #4. Compute loss for labels
                    ref_deps_idxes,ref_gov_idxes,ref_labels = labeldata[0].to(xdevice),labeldata[1].to(xdevice),labeldata[2].to(xdevice)
                    print('labels',list(zip(ref_deps_idxes.cpu().numpy(),ref_gov_idxes.cpu().numpy(),ref_labels.cpu().numpy())))
                    deps_embeddings   = input_seq[ref_deps_idxes]
                    gov_embeddings    = input_seq[ref_gov_idxes]
                    label_predictions = self.label_biaffine(self.dep_lab(deps_embeddings),self.head_lab(gov_embeddings))
                    lloss  = label_loss_fn(label_predictions,ref_labels)
                    lN   += len(ref_labels)
                    lNLL += lloss.item()
            return (eNLL/eN,lNLL/lN)
        
    def predict(self,dataset):

        softmax = nn.Softmax(dim=1)
        
        with torch.no_grad():
            dataloader = DataLoader(dataset,batch_size=32,shuffle=False, num_workers=4,collate_fn=dep_collate_fn,sampler=SequentialSampler(dataset))
            for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader)): 
                for (edgedata,labeldata,tok_sequence) in batch:
                    word_emb_idxes,ref_gov_idxes = edgedata[0].to(xdevice),edgedata[1].to(xdevice)
                    N = len(word_emb_idxes)
                    #1. Run LSTM on raw input and get word embeddings
                    embeddings        = self.E(word_emb_idxes).unsqueeze(dim=0)
                    input_seq,end     = self.rnn(embeddings)
                    input_seq         = input_seq.squeeze(dim=0)
                    #2.  Compute edge attention from flat matrix representation
                    deps_embeddings   = torch.repeat_interleave(input_seq,repeats=N,dim=0)
                    gov_embeddings    = input_seq.repeat(N,1)
                    attention_scores  = self.edge_biaffine(self.dep_arc(deps_embeddings),self.head_arc(gov_embeddings))
                    attention_matrix  = softmax(attention_scores.view(N,N))
                    #3. Compute max spanning tree
                    M                   = attention_matrix.cpu().numpy()[1:,1:].T         
                    G                   = nx.from_numpy_matrix(M,create_using=nx.DiGraph)
                    A                   = nx.maximum_spanning_arborescence(G,default=0)    #this performs a max (sum of scores)
                    #4. Compute edge labels 
                    edgelist            = list(A.edges)
                    print('edges',edgelist)
                     #print('labels',list(zip(ref_gov_idxes.cpu().numpy(),ref_deps_idxes.cpu().numpy(),ref_labels.cpu().numpy())))
                    if edgelist:
                        gov_embeddings      = input_seq [ torch.tensor( [ gov for (gov,dep) in edgelist ] ) ]
                        deps_embeddings     = input_seq [ torch.tensor( [ dep for (gov,dep) in edgelist ] ) ]
                        label_predictions   = softmax(self.label_biaffine(self.dep_lab(deps_embeddings),self.head_lab(gov_embeddings)))
                        pred_idxes          = torch.argmax(label_predictions,dim=1)
                        pred_labels         = [ dataset.itolab[idx] for idx in pred_idxes ]
                        print(label_predictions)
                        print( pred_labels)
                        print(trainset.itolab)
                        dg                  = DepGraph([ (gov,label,dep) for ( (gov,dep),label) in zip(edgelist,pred_labels)],wordlist=tok_sequence)
                        yield dg
                    else:
                        print('failure.')

xdevice = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('device',xdevice)

trainset  = DependencyDataset('spmrl/train.French.gold.conll')
devset    = DependencyDataset('spmrl/dev.French.gold.conll',use_vocab=trainset.itos,use_labels=trainset.itolab)
testset   = DependencyDataset('spmrl/test.French.gold.conll',use_vocab=trainset.itos,use_labels=trainset.itolab)

#istream = open('spmrl/dev.French.gold.conll')
#treelist = [ DepGraph.read_tree(istream) for _ in range(100)    ]
#treelist = [ tree  for tree in treelist if len(tree.words) < 20 ]
#vocab    = [ word  for tree in treelist for word in tree.words  ]
#labels   = [ label for tree in treelist for label in tree.get_all_labels() ]

emb_size    = 130
arc_mlp     = 150
lab_mlp     = 150
lstm_hidden = 200
model       = GraphParser(trainset.itos,trainset.itolab,emb_size,lstm_hidden,arc_mlp,lab_mlp)
model.to(xdevice)
model.train(trainset,trainset,30)
print('running test')
ostream = open('testout.conll','w')
for tree in model.predict(trainset):
    print(tree,file=ostream)
    print('',file=ostream)
ostream.close()
