import numpy as np
import numpy.random as rd
import torch
import torch.optim as optim
import torch.nn as nn

from mst import mst
from deptree import *
from torch.nn.functional import pad
from torch.utils import data
from torch.utils.data import DataLoader,SequentialSampler
from math import sqrt
from tqdm import tqdm
from random import sample,shuffle,random
from collections import Counter
from torch.autograd import Variable

class DependencyDataset(data.Dataset):
    """
    A representation of the DepBank for efficient processing.
    This is a sorted dataset
    """
    PAD_IDX            = 0
    PAD_TOKEN          = '<pad>'
    UNK_WORD           = '<unk>'
    
    def __init__(self,filename,use_vocab=None,use_labels=None,min_vocab_freq=0):
        istream       = open(filename)
        self.treelist = []
        tree = DepGraph.read_tree(istream) 
        while tree:
            if len(tree) <= 30: 
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
        self.preprocess_data()

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
    
    def init_vocab(self,treelist,threshold):
        """
        Extracts the set of tokens found in the data and orders it 
        """
        vocab = Counter([ word  for tree in treelist for word in tree.words ])
        vocab = set([tok for (tok,counts) in vocab.most_common() if counts > threshold])
        vocab.update([DependencyDataset.UNK_WORD])

        self.itos = [DependencyDataset.PAD_TOKEN] +list(vocab)
        self.stoi = {token:idx for idx,token in enumerate(self.itos)}

    def init_labels(self,treelist):
        labels      = set([ lbl for tree in treelist for (gov,lbl,dep) in tree.get_all_edges()])
        self.itolab = [DependencyDataset.PAD_TOKEN] + list(labels)
        self.labtoi = {label:idx for idx,label in enumerate(self.itolab)}
        print(self.labtoi)
        
    def preprocess_data(self):
        """
        Encodes the dataset and makes it ready for processing.
        This is the encoding for the edge prediction task 
        """ 
        self.xdep      = [ ] 
        self.gov_idxes = [ ] 
        self.lab_idxes = [ ] 
        self.tokens    = [ ]
        def word_sampler(word_idx,dropout):
            return self.stoi[DependencyDataset.UNK_WORD]  if random() < dropout else word_idx
          
        for tree in self.treelist:
            word_seq      = tree.words
            depword_idxes = [self.stoi.get(tok,self.stoi[DependencyDataset.UNK_WORD]) for tok in word_seq]
            if self.word_dropout:
                depword_idxes = [word_sampler(widx,self.word_dropout) for widx in depword_idxes]
            govs          = self.oracle_governors(tree)
            lab_idxes     = self.oracle_labels(tree)
            self.xdep.append(depword_idxes)
            self.gov_idxes.append(govs)
            self.lab_idxes.append(lab_idxes)
            self.tokens.append(tree.words)
            
            
    def __len__(self):      
        return len(self.treelist)
    
    def __getitem__(self,idx):
        return {'xdep':self.xdep[idx],'govidx':self.gov_idxes[idx],'labidx':self.lab_idxes[idx],'wordlist':self.tokens[idx]}

    def oracle_labels(self,depgraph):
        """
        Returns a list where each element list[i] is the label of
        the position of the governor of the word at position i.
        Returns:
        a tensor of size N.
        """
        N          = len(depgraph)
        edges      = depgraph.get_all_edges( )
        rev_labels = dict( [ (dep,label) for (gov,label,dep) in edges ] )
        return [ self.labtoi.get(rev_labels.get(idx,DependencyDataset.PAD_IDX),DependencyDataset.PAD_IDX) for idx in range(N) ]  
        
    def oracle_governors( self,depgraph ): 
        """
        Returns a list where each element list[i] is the index of
        the position of the governor of the word at position i.
        Returns:
        a tensor of size N.
        """
        N         = len( depgraph )
        edges     = depgraph.get_all_edges( )
        rev_edges = dict( [(dep,gov) for (gov,label,dep) in edges ] )
        return [ rev_edges.get(idx,0) for idx in range(N) ]

def pad_batch_matrix(batch_matrix,pad_value):
    """
    Pads rightwards
    """
    batch_len = max( [ len(elt) for elt in batch_matrix] )
    
    for line in batch_matrix:
        line.extend([pad_value]*(batch_len-len(line)))
    return batch_matrix
        
def dep_collate_fn(batch):
    """
    That's the collate function for batching edges
    """
    #edges
    XDEP    = pad_batch_matrix( [ elt['xdep']    for elt in batch],DependencyDataset.PAD_IDX )
    REFGOV  = pad_batch_matrix( [ elt['govidx']  for elt in batch],DependencyDataset.PAD_IDX )
    LABVAL  = pad_batch_matrix( [ elt['labidx'] for elt in batch], DependencyDataset.PAD_IDX )
    TOKENS  = [elt['wordlist'] for elt in batch]
    LENGTHS = [len(elt) for elt in TOKENS]
    return ( torch.tensor(XDEP),torch.tensor(REFGOV), torch.tensor(LABVAL), torch.tensor(LENGTHS), TOKENS )


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

        self.E              = nn.Embedding(vocab_size,word_embedding_size,padding_idx=DependencyDataset.PAD_IDX)
        self.edge_biaffine  = Biaffine(lstm_hidden,1)
        self.label_biaffine = Biaffine(lstm_hidden,label_size)
        self.head_arc       = MLP(lstm_hidden*2,arc_mlp_hidden,lstm_hidden,dropout=dropout)
        self.dep_arc        = MLP(lstm_hidden*2,arc_mlp_hidden,lstm_hidden,dropout=dropout)
        self.head_lab       = MLP(lstm_hidden*2,lab_mlp_hidden,lstm_hidden,dropout=dropout)
        self.dep_lab        = MLP(lstm_hidden*2,lab_mlp_hidden,lstm_hidden,dropout=dropout)
        self.rnn            = nn.LSTM(word_embedding_size,lstm_hidden,bidirectional=True,num_layers=1,dropout=dropout)
        
    def train_model(self,trainset,devset,epochs,dropout=0.0):

        trainset.word_dropout = dropout
        print("N =",len(trainset))
        nlabels       = self.label_biaffine.U.size(0)
        edge_loss_fn  = nn.CrossEntropyLoss(reduction = 'sum',ignore_index=-1) #dummy root is excluded 
        label_loss_fn = nn.CrossEntropyLoss(reduction = 'sum',ignore_index=DependencyDataset.PAD_IDX)
        optimizer     = optim.Adam( self.parameters() )
        
        bestNLL = 100 
        for ep in range(epochs):
            self.train()
            eNLL,eN,lNLL,lN = 0,0,0,0
            print("epoch",ep) 
            try:
                dataloader = DataLoader(trainset, batch_size=32,shuffle=True, num_workers=8,collate_fn=dep_collate_fn)
                for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader)):

                    optimizer.zero_grad()  
                    word_emb_idxes,ref_gov_posn,reflabel_idxes,true_lengths,tok_sequence = batch
                    word_emb_idxes,ref_gov_posn = word_emb_idxes.to(xdevice),ref_gov_posn.to(xdevice)

                    batch_len            = true_lengths.max( ) 
                    truelength_mask      = torch.arange(batch_len)[None,:] < true_lengths[:,None]   #2d mask

                    #1. Run Lexer and LSTM on raw input and get word embeddings
                    embeddings        = self.E(word_emb_idxes)
                    input_seq,end     = self.rnn(embeddings)  
                    input_seq         = input_seq           
                        
                    dep_vectors       = self.dep_arc(input_seq) 
                    head_vectors      = self.head_arc(input_seq)

                    #2.  Compute edge attention from flat matrix representation
                    attention_matrix  = self.edge_biaffine(head_vectors,dep_vectors)             # [batch, sent_len, sent_len]
                    attention_matrix.masked_fill_(~truelength_mask.unsqueeze(1),float('-inf'))   # applies -inf mask with mask broadcasting to 3 dim tensor for columns
                    attention_mask    = truelength_mask.unsqueeze(1).repeat(1,batch_len,1).transpose(1,2) #creates a mask for the whole batch, selecting rows
                    attention_matrix  = torch.masked_select(attention_matrix,attention_mask).view(-1,batch_len) # selects the non padded rows in the batch of attention matrices

                    #3. Compute loss and backprop for edges
                    ref_gov_posn_edge = torch.masked_select(ref_gov_posn,truelength_mask)        #extracts the unpadded ref values as a flat list [batch*sent_truelen]
                    eloss = edge_loss_fn(attention_matrix,ref_gov_posn_edge)
                    eN   += float(sum(true_lengths))
                    eNLL += eloss.item( )

                    #4. Compute loss and backprop for labels
                    labdep_vectors  = self.dep_lab(input_seq)   
                    labhead_vectors = self.head_lab(input_seq)
                    label_matrix    = self.label_biaffine(labhead_vectors,labdep_vectors)
                    ref_gov_posn    = ref_gov_posn.unsqueeze(1).unsqueeze(2)                                   # [batch, 1, 1, sent_len]
                    ref_gov_posn    = ref_gov_posn.expand(-1, label_matrix.size(1), -1, -1)                    # [batch, n_labels, 1, sent_len]
                    label_matrix    = torch.gather(label_matrix, 2, ref_gov_posn).squeeze(2).transpose(-1, -2) # [batch, n_labels, sent_len]
                    label_matrix    = torch.masked_select(label_matrix,truelength_mask.unsqueeze(2).repeat(1,1,nlabels)).view(-1,nlabels)

                    reflabel_idxes    = reflabel_idxes.to(xdevice)
                    reflabel_idxes    = torch.masked_select(reflabel_idxes,truelength_mask)
                    lloss  = label_loss_fn(label_matrix,reflabel_idxes)

                    loss =  eloss + lloss
                    loss.backward( ) 
                    lN   += len(reflabel_idxes)
                    lNLL += lloss.item()
                    optimizer.step( )
                     
                devePPL,devlPPL = self.eval_model(devset)
                if devePPL+devlPPL < bestNLL:
                    print('   saving model.')
                    bestNLL = devePPL+devlPPL
                    torch.save(self.state_dict(),'test_biaffine.pt2')
                    #self.save_model('test_biaffine.pt2')
                print('\n  TRAIN: mean NLL(edges)',eNLL/eN,'mean NLL(labels)',lNLL/lN)
                print('  DEV  : mean NLL(edges)',devePPL,'mean NLL(labels)',devlPPL)
            except KeyboardInterrupt:
                print('Received SIGINT. Aborting training.')
                self.load_state_dict(torch.load('test_biaffine.pt2'))
                return
        self.load_state_dict(torch.load('test_biaffine.pt2'))

        
    def eval_model(self,dataset):
        
        edge_loss_fn  = nn.CrossEntropyLoss(reduction = 'sum',ignore_index=-1)                        #ignores the dummy root index
        label_loss_fn = nn.CrossEntropyLoss(reduction = 'sum',ignore_index=DependencyDataset.PAD_IDX) #ignores the pad indexes

        nlabels       = self.label_biaffine.U.size(0)
        print('eval mode',len(dataset))
        with torch.no_grad():
            self.eval()
            eNLL,eN,lNLL,lN = 0,0,0,0
            dataloader = DataLoader(dataset, batch_size=32,shuffle=True, num_workers=4,collate_fn=dep_collate_fn)
            for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader)):

                word_emb_idxes,ref_gov_posn,reflabel_idxes,true_lengths,tok_sequence = batch
                word_emb_idxes,ref_gov_posn = word_emb_idxes.to(xdevice),ref_gov_posn.to(xdevice)

                batch_len            = true_lengths.max( ) 
                truelength_mask      = torch.arange(batch_len)[None,:] < true_lengths[:,None]   #2d mask
                
                #1. Run Lexer and LSTM on raw input and get word embeddings
                embeddings        = self.E(word_emb_idxes)
                input_seq,end     = self.rnn(embeddings)  
                input_seq         = input_seq           
                        
                dep_vectors       = self.dep_arc(input_seq) 
                head_vectors      = self.head_arc(input_seq)

                #2.  Compute edge attention from flat matrix representation
                attention_matrix  = self.edge_biaffine(head_vectors,dep_vectors)             # [batch, sent_len, sent_len]
                attention_matrix.masked_fill_(~truelength_mask.unsqueeze(1),float('-inf'))   # applies -inf mask with mask broadcasting to 3 dim tensor for columns
                attention_mask    = truelength_mask.unsqueeze(1).repeat(1,batch_len,1).transpose(1,2) #creates a mask for the whole batch, selecting rows
                attention_matrix  = torch.masked_select(attention_matrix,attention_mask).view(-1,batch_len) # selects the non padded rows in the batch of attention matrices

                #3. Compute loss and backprop for edges
                ref_gov_posn_edge = torch.masked_select(ref_gov_posn,truelength_mask)        #extracts the unpadded ref values as a flat list [batch*sent_truelen]
                eloss = edge_loss_fn(attention_matrix,ref_gov_posn_edge)
                eN   += float(sum(true_lengths))
                eNLL += eloss.item( )

                #4. Compute loss and backprop for labels
                labdep_vectors  = self.dep_lab(input_seq)   
                labhead_vectors = self.head_lab(input_seq)
                label_matrix    = self.label_biaffine(labhead_vectors,labdep_vectors)
                ref_gov_posn    = ref_gov_posn.unsqueeze(1).unsqueeze(2)                                   # [batch, 1, 1, sent_len]
                ref_gov_posn    = ref_gov_posn.expand(-1, label_matrix.size(1), -1, -1)                    # [batch, n_labels, 1, sent_len]
                label_matrix    = torch.gather(label_matrix, 2, ref_gov_posn).squeeze(2).transpose(-1, -2) # [batch, n_labels, sent_len]
                label_matrix    = torch.masked_select(label_matrix,truelength_mask.unsqueeze(2).repeat(1,1,nlabels)).view(-1,nlabels)

                reflabel_idxes    = reflabel_idxes.to(xdevice)
                reflabel_idxes    = torch.masked_select(reflabel_idxes,truelength_mask)
                lloss  = label_loss_fn(label_matrix,reflabel_idxes)

                loss =  eloss + lloss
                lN   += len(reflabel_idxes)
                lNLL += lloss.item()

            return (eNLL/eN,lNLL/lN)
        
    def predict(self,dataset):

        with torch.no_grad():
            self.eval()
            dataloader = DataLoader(dataset,batch_size=8,shuffle=False, num_workers=4,collate_fn=dep_collate_fn,sampler=SequentialSampler(dataset))
            for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader)): 

                word_emb_idxes,_,_,true_lengths,tok_sequence = batch
                word_emb_idxes = word_emb_idxes.to(xdevice)

                print(tok_sequence)
                          
                #1. Run Lexer and LSTM on raw input and get word embeddings
                embeddings        = self.E(word_emb_idxes)
                input_seq,end     = self.rnn(embeddings)
                input_seq         = input_seq
                        
                dep_vectors  = self.dep_arc(input_seq)
                head_vectors = self.head_arc(input_seq)
                    
                #2.  Compute edge attention from flat matrix representation
                attention_matrix  = self.edge_biaffine(head_vectors,dep_vectors)                       # [batch, sent_len, sent_len]
                #attention_matrix  = attention_matrix.contiguous().view(-1, attention_matrix.size(-1))  # [batch*sent_len, sent_len]
                #print('AA',attention_matrix)
                #3. Compute labels 
                labdep_vectors  = self.dep_lab(input_seq)   
                labhead_vectors = self.head_lab(input_seq)  
                label_matrix    = self.label_biaffine(labhead_vectors,labdep_vectors)
                
                # Predict heads
                
                for attentn_matrix,label_tensor in zip(attention_matrix,label_matrix):
                    
                    mat = attentn_matrix.T.numpy()
                    gov_idxes = mst(mat)
                    select = torch.LongTensor(gov_idxes).unsqueeze(0).expand(label_tensor.size(0), -1)
                    select = Variable(select)
                    selected = torch.gather(label_tensor, 1, select.unsqueeze(1)).squeeze(1)
                    _, labels = selected.max(dim=0)
                    labels = labels.data.numpy()
                    edges = [ (gidx,dataset.itolab[lbl],didx)  for (didx,lbl,gidx) in zip(range(len(gov_idxes)),labels,gov_idxes) if didx != gidx ]
                    G =  DepGraph(edges,with_root=False)
                    G.words=tok_sequence[0]
                    print(G)
                    yield G
                    
                #Predict labels
                #select = torch.LongTensor(ref_gov_posn).unsqueeze(0).expand(label_scores.size(0), -1)
                #select = Variable(select)
                #selected = torch.gather(label_scores, 1, select.unsqueeze(1)).squeeze(1)
                #_, labels = selected.max(dim=0)
                #labels = labels.data.numpy()
                #print('preds',ref_gov_posn, labels)
                
                #yield dg

emb_size    = 100
arc_mlp     = 400
lab_mlp     = 100
lstm_hidden = 400                    
xdevice = 'cpu'#torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('device used',xdevice)

trainset    = DependencyDataset('spmrl/train.French.gold.conll',min_vocab_freq=0)
itos,itolab = trainset.itos,trainset.itolab
devset      = DependencyDataset('spmrl/dev.French.gold.conll')#,use_vocab=itos,use_labels=itolab)
trainset.save_vocab('model.vocab')
 
model       = GraphParser(itos,itolab,emb_size,lstm_hidden,arc_mlp,lab_mlp,dropout=0.0)
model.to(xdevice)
model.train_model(trainset,devset,50)

#model       = GraphParser.load_model('test_biaffine.pt2')
#itos,itolab = DependencyDataset.load_vocab('model.vocab')

testset     = DependencyDataset('spmrl/test.French.gold.conll',use_vocab=itos,use_labels=itolab)
ostream     = open('testoutref.conll','w')
for tree in devset.treelist:
    print(tree,file=ostream)
    print('',file=ostream)
ostream.close()
print('running test')
ostream = open('testout.conll2','w')
#for tree in model.predict(devset):
#    print(tree,file=ostream)
##    print('',file=ostream,flush=True)
#ostream.close()
#itos,itolab = devset.itos,devset.itolab
#print(itos)
