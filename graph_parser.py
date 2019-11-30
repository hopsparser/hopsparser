import numpy as np
import numpy.random as rd
import torch
import torch.optim as optim
import torch.nn as nn



from deptree import *
from torch.nn.functional import pad
from torch.utils import data
from torch.utils.data import DataLoader,SequentialSampler
from math import sqrt
from tqdm import tqdm
from random import sample,shuffle,random
from collections import Counter

torch.multiprocessing.set_sharing_strategy('file_system')


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
    
    def __init__(self,filename,use_vocab=None,use_labels=None,min_vocab_freq=0):
        istream       = open(filename)
        self.treelist = []
        tree = DepGraph.read_tree(istream) 
        while tree:
            if len(tree) <= 10: 
                self.treelist.append(tree)
            else:
                print('dropped sentence',len(tree))
            tree = DepGraph.read_tree(istream)
        istream.close()
        shuffle(self.treelist)
        #self.treelist.sort(key=lambda x:len(x)) # we do not make real batches later
        if use_vocab:
            self.itos = use_vocab
            self.stoi = {token:idx for idx,token in enumerate(self.itos)}
        else:
            self.init_vocab(self.treelist,threshold=min_vocab_freq)
        if use_labels:
            self.itolab = use_labels
            self.labtoi = {label:idx for idx,label in enumerate(self.itolab)}
        else:
            self.init_labels(self.treelist)

        self.word_dropout = 0.0
        self.preprocess_edges()
        self.preprocess_labels()

    def save_vocab(self,filename):
        
        out = open(filename,'w')
        print(' '.join(self.itos),file=out)
        print(' '.join(self.itolab),file=out)
        out.close()

    @staticmethod
    def load_vocab(filename):

        reloaded = open(filename)
        itos     = reloaded.readline().split()
        itolab   = reloaded.readline().split()
        reloaded.close()
        return itos,itolab
        
    def shuffle(self):
        self.treelist.shuffle()
        self.treelist.sort(key=lambda x:len(x))
        self.preprocess_edges()
        self.preprocess_labels()
    
    def init_vocab(self,treelist,threshold):
        """
        Extracts the set of tokens found in the data and orders it 
        """
        vocab = Counter([ word  for tree in treelist for word in tree.words ])
        vocab = set([tok for (tok,counts) in vocab.most_common() if counts > threshold])
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

        def word_sampler(word_idx,dropout):
            return self.stoi[DependencyDataset.UNK_WORD]  if random() < dropout else word_idx
          
        for tree in self.treelist:
            word_seq      = [DependencyDataset.ROOT] + tree.words
            depword_idxes = [self.stoi.get(tok,self.stoi[DependencyDataset.UNK_WORD]) for tok in word_seq]
            if self.word_dropout:
                depword_idxes = [word_sampler(widx,self.word_dropout) for widx in depword_idxes]
            print(word_seq)
            print(depword_idxes)
            print('unk word idx',self.stoi[DependencyDataset.UNK_WORD])
            gov_idxes     = [DependencyDataset.PAD_WORD_IDX] + DependencyDataset.oracle_ancestors(tree)
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


def pad_batch_matrix(batch_matrix):
    """
    Pads rightwards
    """
    batch_len = max( [ len(elt) for elt in batch_matrix] )
    
    for line in batch_matrix:
        line.extend([DependencyDataset.PAD_WORD_IDX]*(batch_len-len(line)))
    return batch_matrix
        
def dep_collate_fn(batch):
    """
    That's the collate function for batching edges
    """
    #edges
    XDEP    = pad_batch_matrix( [ elt['xdep']    for elt in batch] )
    REFGOV  = pad_batch_matrix( [ elt['refidx']  for elt in batch] )
    #labels
    LABDEPS = pad_batch_matrix( [ elt['refdeps'] for elt in batch] )
    LABGOVS = pad_batch_matrix( [ elt['refgovs'] for elt in batch] )
    LABVAL  = pad_batch_matrix( [ elt['reflabels'] for elt in batch] )
    #tokens
    TOKENS  = [elt['wordlist'] for elt in batch]
    return ( (torch.tensor(XDEP),torch.tensor(REFGOV)) , (torch.tensor(LABDEPS),torch.tensor(LABGOVS),torch.tensor(LABVAL)) , TOKENS )
    
class MLP(nn.Module):

    def __init__(self,input_size,hidden_size,output_size,dropout=0.0):
        super(MLP, self).__init__()
        self.Wdown    = nn.Linear(input_size,hidden_size)
        self.Wup      = nn.Linear(hidden_size,output_size)
        self.g        = nn.ReLU()
        self.dropout  = nn.Dropout(p=dropout)
        
    def forward(self,input):
        return self.Wup(self.dropout(self.g(self.Wdown(input))))

class Biaffine(nn.Module):
    
    """Biaffine attention layer."""
    def __init__(self, input_dim, output_dim):
        super(Biaffine, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.rand(output_dim, input_dim, input_dim)-0.5/sqrt(input_dim))

    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)
    
    
class GraphParser(nn.Module):

    def __init__(self,vocab,labels,word_embedding_size,lstm_hidden,arc_mlp_hidden,lab_mlp_hidden,dropout=0.1):
        
        super(GraphParser, self).__init__()
        self.allocate(word_embedding_size,len(vocab),len(labels),lstm_hidden,arc_mlp_hidden,lab_mlp_hidden,dropout)

        
    @staticmethod
    def load_model(filename):
        reloaded = torch.load(filename)
        model    = GraphParser([0]*reloaded['vocab_len'],[0]*reloaded['label_len'],\
                                   reloaded['word_embedding_size'],reloaded['lstm_hidden'],reloaded['arc_mlp_hidden'],reloaded['lab_mlp_hidden'])
        model.load_state_dict(reloaded['state_dict'])
        model = model.to(xdevice)
        return model
    
    def save_model(self,filename):
        vocab_len,word_embedding_size   = tuple(self.E.weight.size())
        label_len,_                     = tuple(self.label_biaffine.W.size())
        arc_mlp_hidden,lstm_hidden_size = tuple(self.dep_arc.Wdown.weight.size())
        lab_mlp_hidden,_                = tuple(self.dep_lab.Wdown.weight.size())
        
        torch.save({'vocab_len':vocab_len,\
                    'label_len':label_len,\
                    'word_embedding_size':word_embedding_size,\
                    'lstm_hidden':int(lstm_hidden_size/2),\
                    'arc_mlp_hidden':arc_mlp_hidden,\
                    'lab_mlp_hidden':lab_mlp_hidden,\
                    'state_dict':self.state_dict()},filename)
                    
    def allocate(self,word_embedding_size,vocab_size,label_size,lstm_hidden,arc_mlp_hidden,lab_mlp_hidden,dropout):

        self.E              = nn.Embedding(vocab_size,word_embedding_size)
        self.edge_biaffine  = Biaffine(lstm_hidden,1)
        self.label_biaffine = Biaffine(lstm_hidden,label_size)
        self.head_arc       = MLP(lstm_hidden*2,arc_mlp_hidden,lstm_hidden,dropout=dropout)
        self.dep_arc        = MLP(lstm_hidden*2,arc_mlp_hidden,lstm_hidden,dropout=dropout)
        self.head_lab       = MLP(lstm_hidden*2,lab_mlp_hidden,lstm_hidden,dropout=dropout)
        self.dep_lab        = MLP(lstm_hidden*2,lab_mlp_hidden,lstm_hidden,dropout=dropout)
        self.rnn            = nn.LSTM(word_embedding_size,lstm_hidden,bidirectional=True,num_layers=1,dropout=dropout)
        
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
    
    def train_model(self,trainset,devset,epochs,dropout=0.0):

        trainset.word_dropout = dropout
        print("N =",len(trainset))
        edge_loss_fn  = nn.CrossEntropyLoss(reduction = 'sum',ignore_index=DependencyDataset.PAD_WORD_IDX) #ignores the dummy root/pad indexes
        label_loss_fn = nn.CrossEntropyLoss(reduction = 'sum') 
        optimizer     = optim.Adam( self.parameters() )
        
        bestNLL = 100
        for ep in range(epochs):
            self.train()
            eNLL,eN,lNLL,lN = 0,0,0,0
            print("epoch",ep)
            try:
                
                dataloader = DataLoader(trainset, batch_size=2,shuffle=True, num_workers=4,collate_fn=dep_collate_fn)
                for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader)):
                    
                    edgedata,labeldata,tok_sequence = batch
                    optimizer.zero_grad()
                    word_emb_idxes,ref_gov_idxes = edgedata[0].to(xdevice),edgedata[1].to(xdevice)
                    N = len(word_emb_idxes)
                    #print('word idxes',word_emb_idxes)
                    
                    #1. Run Lexer and LSTM on raw input and get word embeddings
                    embeddings        = self.E(word_emb_idxes)

                    input_seq,end     = self.rnn(embeddings)
                    input_seq         = input_seq
                    #print('lstm_repr',input_seq)
                        
                    dep_vectors  = self.dep_arc(input_seq)
                    head_vectors = self.head_arc(input_seq)
                        
                    #2.  Compute edge attention from flat matrix representation
                    #deps_embeddings   = torch.repeat_interleave(input_seq,repeats=N,dim=0)
                    #gov_embeddings    = input_seq.repeat(N,1)
                    attention_matrix  = self.edge_biaffine(head_vectors,dep_vectors)
                    #attention_matrix  = attention_scores.view(N,N)
                    #print('attention',attention_matr)
                    #3. Compute loss and backprop for edges
                    eloss = edge_loss_fn(attention_matrix,ref_gov_idxes)
                    eloss.backward() #
                    eN   += N
                    eNLL += eloss.item()
                    print(eNLL)
                    #4. Compute loss and backprop for labels
                    #ref_deps_idxes,ref_gov_idxes,ref_labels = labeldata[0].to(xdevice),labeldata[1].to(xdevice),labeldata[2].to(xdevice)
                    #deps_embeddings   = input_seq[ref_deps_idxes]
                    #gov_embeddings    = input_seq[ref_gov_idxes]
                    #label_predictions = self.label_biaffine(self.dep_lab(deps_embeddings),self.head_lab(gov_embeddings))
                    #lloss  = label_loss_fn(label_predictions,ref_labels)
                    #lloss.backward( )
                    #lN   += len(ref_labels)
                    #lNLL += lloss.item()
                    optimizer.step( )
                deveNLL,devlNLL = self.eval_model(devset)
                if deveNLL+devlNLL < bestNLL:
                    print('   saving model.')
                    bestNLL = deveNLL+devlNLL
                    self.save_model('test_biaffine.pt2')
                print('\n  TRAIN: mean NLL(edges)',eNLL/eN,'mean NLL(labels)',lNLL/lN)
                print('  DEV  : mean NLL(edges)',deveNLL,'mean NLL(labels)',devlNLL)
            except KeyboardInterrupt:
                print('Received SIGINT. Aborting training.')
                self.load_state_dict(torch.load('test_biaffine.pt2'))
                return
        self.load_state_dict(torch.load('test_biaffine.pt2')['state_dict'])
                
    def eval_model(self,dataset):
        
        edge_loss_fn  = nn.CrossEntropyLoss(reduction = 'sum',ignore_index=DependencyDataset.ROOT_GOV_IDX) #ignores the dummy root index
        label_loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
        
        with torch.no_grad():
            self.eval()
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
                    deps_embeddings   = input_seq[ref_deps_idxes]
                    gov_embeddings    = input_seq[ref_gov_idxes]
                    label_predictions = self.label_biaffine(self.dep_lab(deps_embeddings),self.head_lab(gov_embeddings))
                    lloss  = label_loss_fn(label_predictions,ref_labels)
                    lN   += len(ref_labels)
                    lNLL += lloss.item()
            return (eNLL/eN,lNLL/lN)
        
    def predict(self,dataset):

        softmax = nn.Softmax(dim=1) #should not be a softmax for Edmonds (sum of logs works worse ??)
        with torch.no_grad():
            self.eval()
            dataloader = DataLoader(dataset,batch_size=32,shuffle=False, num_workers=4,collate_fn=dep_collate_fn,sampler=SequentialSampler(dataset))
            for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader)): 
                for (edgedata,labeldata,tok_sequence) in batch:
                    word_emb_idxes,ref_gov_idxes = edgedata[0].to(xdevice),edgedata[1].to(xdevice)
                    N = len(word_emb_idxes)
                    if N == 2: 
                        yield DepGraph([(0,DependencyDataset.ROOT,1)],with_root=False,wordlist=tok_sequence)
                        continue #abort this sentence. There is just a dummy root and a single token
                    
                    #1. Run LSTM on raw input and get word embeddings
                    embeddings        = self.E(word_emb_idxes).unsqueeze(dim=0)
                    input_seq,end     = self.rnn(embeddings)
                    input_seq         = input_seq.squeeze(dim=0)
                    #2.  Compute edge attention from flat matrix representation
                    deps_embeddings   = torch.repeat_interleave(input_seq,repeats=N,dim=0)
                    gov_embeddings    = input_seq.repeat(N,1)
                    attention_scores  = self.edge_biaffine(self.dep_arc(deps_embeddings),self.head_arc(gov_embeddings))
                    attention_matrix  = attention_scores.view(N,N) #use normalized raw scores to compute the MST 
                    #3. Compute max spanning tree
                    M                   = attention_matrix.cpu().numpy()[1:,1:].T         
                    G                   = mst.numpy2graph(M)
                    A                   = mst.mst_one_out_root(G)    #this performs a max (sum of scores)
                    #4. Compute edge labels 
                    edgelist            = mst.edgelist(A)
                    gov_embeddings      = input_seq [ torch.tensor( [ gov+1 for (gov,dep) in edgelist ] ) ]
                    deps_embeddings     = input_seq [ torch.tensor( [ dep+1 for (gov,dep) in edgelist ] ) ]                        
                    label_predictions   = softmax(self.label_biaffine(self.dep_lab(deps_embeddings),self.head_lab(gov_embeddings)))
                    pred_idxes          = torch.argmax(label_predictions,dim=1)
                    pred_labels         = [ dataset.itolab[idx] for idx in pred_idxes ]
                    dg                  = DepGraph([ (gov,label,dep) for ( (gov,dep),label) in zip(edgelist,pred_labels)],wordlist=tok_sequence)
                    yield dg

emb_size    = 5
arc_mlp     = 10
lab_mlp     = 15
lstm_hidden = 10                    
xdevice = 'cpu'#torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('device used',xdevice)

trainset    = DependencyDataset('spmrl/train.French.gold.conll',min_vocab_freq=0)
itos,itolab = trainset.itos,trainset.itolab
devset      = DependencyDataset('spmrl/dev.French.gold.conll' ,use_vocab=itos,use_labels=itolab)
trainset.save_vocab('model.vocab')

model       = GraphParser(trainset.itos,trainset.itolab,emb_size,lstm_hidden,arc_mlp,lab_mlp,dropout=0.5)
model.to(xdevice)
model.train_model(trainset,devset,50)

#model       = GraphParser.load_model('test_biaffine.pt2')
#itos,itolab = DependencyDataset.load_vocab('model.vocab')

testset     = DependencyDataset('spmrl/test.French.gold.conll',use_vocab=itos,use_labels=itolab)
ostream     = open('testoutref.conll','w')
for tree in testset.treelist:
    print(tree,file=ostream)
    print('',file=ostream)
ostream.close()
print('running test')
ostream = open('testout.conll2','w')
for tree in model.predict(testset):
    print(tree,file=ostream)
    print('',file=ostream,flush=True)
ostream.close()
