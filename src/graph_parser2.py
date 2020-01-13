import torch
import numpy as np
from deptree import *
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from random import sample,shuffle,random
from collections import Counter,defaultdict
from mst import chuliu_edmonds

class DependencyDataset:
    """
    A representation of the DepBank for efficient processing.
    This is a sorted dataset.
    """
    PAD_IDX            = 0
    PAD_TOKEN          = '<pad>'
    UNK_WORD           = '<unk>'
    
    def __init__(self,filename,use_vocab=None,use_labels=None,min_vocab_freq=0):
        istream       = open(filename)
        self.treelist = []
        tree = DepGraph.read_tree(istream) 
        while tree:
            self.treelist.append(tree)
            tree = DepGraph.read_tree(istream)
        istream.close()
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
        self.encode()
        
    def encode(self):

        def word_sampler(word_idx,dropout):
            return self.stoi[DependencyDataset.UNK_WORD]  if random() < dropout else word_idx

        self.deps, self.heads,self.labels = [],[],[]
        self.words = []
        for tree in self.treelist:
            depword_idxes = [self.stoi.get(tok,self.stoi[DependencyDataset.UNK_WORD]) for tok in tree.words]
            if self.word_dropout:
                depword_idxes = [word_sampler(widx,self.word_dropout) for widx in depword_idxes]

            self.words.append(tree.words)
            self.deps.append(depword_idxes)
            self.heads.append(self.oracle_governors(tree))
            self.labels.append([self.labtoi[lab] for lab in self.oracle_labels(tree)])
             
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
    
    def shuffle_data(self):
        N = len(self.deps)
        order = list(range(N))
        shuffle(order)
        self.deps   = [self.deps[i] for i in order]
        self.heads  = [self.heads[i] for i in order]
        self.labels = [self.labels[i] for i in order]
        self.words  = [self.words[i] for i in order]        

    def order_data(self):
        N           = len(self.deps)
        order       = list(range(N))        
        lengths     = map(len,self.deps)
        order       = [idx for idx, L in sorted(zip(order,lengths),key=lambda x:x[1])]
        self.deps   = [self.deps[idx] for idx in order]
        self.heads  = [self.heads[idx] for idx in order]
        self.labels = [self.labels[idx] for idx in order]
        self.words  = [self.words[idx] for idx in order]  
        
    def make_batches(self, batch_size,shuffle_batches=False,shuffle_data=True,order_by_length=False):
        
        if shuffle_data:  
            self.shuffle_data()
        if order_by_length: #shuffling and ordering is relevant : it change the way ties are resolved and thus batch construction
            self.order_data()

        N = len(self.deps)
        batch_order = list(range(0,N, batch_size))
        if shuffle_batches:
            shuffle(batch_order)
            
        for i in batch_order:
            deps   = self.pad(self.deps[i:i+batch_size])
            heads  = self.pad(self.heads[i:i+batch_size])
            labels = self.pad(self.labels[i:i+batch_size])
            words  = self.words[i:i+batch_size]
            yield (words,deps, heads, labels)

    def pad(self,batch):
        sent_lengths = list(map(len, batch))
        max_len = max(sent_lengths)
        padded_batch = [ ]
        for k, seq in zip(sent_lengths, batch):
            padded = seq + (max_len - k)*[ DependencyDataset.PAD_IDX]
            padded_batch.append(padded)
        return Variable(torch.LongTensor(padded_batch))
                
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
        
    def __len__(self):      
        return len(self.treelist)
    
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
        return [ rev_labels.get(idx,DependencyDataset.PAD_TOKEN) for idx in range(N) ]  
        
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

class MLP(nn.Module):

    def __init__(self,input_size,hidden_size,output_size,dropout=0.0):
        super(MLP, self).__init__()
        self.Wdown    = nn.Linear(input_size,hidden_size)
        self.Wup      = nn.Linear(hidden_size,output_size)
        self.g        = nn.ReLU()
        self.dropout  = nn.Dropout(p=dropout)
        
    def forward(self,input):
        return self.Wup(self.dropout(self.g(self.Wdown(input))))

    
class BiAffine(nn.Module):
    """Biaffine attention layer."""
    def __init__(self, input_dim, output_dim):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim)) #check init
        nn.init.xavier_uniform_(self.U)
        
    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)

class BiAffineParser(nn.Module):
    
    """Biaffine Dependency Parser."""
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 encoder_dropout, #lstm dropout
                 mlp_input,
                 mlp_arc_hidden,
                 mlp_lab_hidden,
                 mlp_dropout,
                 num_labels,
                 device='cuda:1'):
    
        super(BiAffineParser, self).__init__()
        self.device    = torch.device(device)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=DependencyDataset.PAD_IDX).to(self.device)
        self.rnn       = nn.LSTM(embedding_size,mlp_input,1, batch_first=True,dropout=encoder_dropout,bidirectional=True).to(self.device)

        # Arc MLPs
        self.arc_mlp_h = MLP(mlp_input*2, mlp_arc_hidden, mlp_input, mlp_dropout).to(self.device)
        self.arc_mlp_d = MLP(mlp_input*2, mlp_arc_hidden, mlp_input, mlp_dropout).to(self.device)
        # Label MLPs
        self.lab_mlp_h = MLP(mlp_input*2, mlp_lab_hidden, mlp_input, mlp_dropout).to(self.device)
        self.lab_mlp_d = MLP(mlp_input*2, mlp_lab_hidden, mlp_input, mlp_dropout).to(self.device)

        # BiAffine layers
        self.arc_biaffine = BiAffine(mlp_input, 1).to(self.device)
        self.lab_biaffine = BiAffine(mlp_input, num_labels).to(self.device)


    def forward(self,xwords):
        
        """Compute the score matrices for the arcs and labels."""
        #check in the future if adding a mask on padded words is useful
        
        xemb   = self.embedding(xwords)        
        cemb,_ = self.rnn(xemb)

        arc_h = self.arc_mlp_h(cemb)
        arc_d = self.arc_mlp_d(cemb)
        
        lab_h = self.lab_mlp_h(cemb)
        lab_d = self.lab_mlp_d(cemb)
                
        scores_arc = self.arc_biaffine(arc_h, arc_d)
        scores_lab = self.lab_biaffine(lab_h, lab_d)
            
        return scores_arc, scores_lab

    def eval_model(self,dev_set,batch_size):

        loss_fnc   = nn.CrossEntropyLoss(reduction='sum')

        #Note: the accurracy scoring is approximative and cannot be interpreted as an UAS/LAS score !
        
        self.eval()
        dev_batches = dev_set.make_batches(batch_size,shuffle_batches=True,shuffle_data=True,order_by_length=True)
        arc_acc, lab_acc,gloss,ntoks = 0, 0, 0, 0
        overall_size = 0
        
        with torch.no_grad():
            for batch in dev_batches:
                
                words, deps, heads, labels = batch
                deps, heads, labels = deps.to(self.device), heads.to(self.device), labels.to(self.device)

                #preds 
                arc_scores, lab_scores = self.forward(deps)
                
                #get global loss
                #ARC LOSS
                arc_scoresL  = arc_scores.transpose(-1, -2)                             # [batch, sent_len, sent_len]
                arc_scoresL  = arc_scores.contiguous().view(-1, arc_scoresL.size(-1))   # [batch*sent_len, sent_len]
                arc_loss     = loss_fnc(arc_scoresL, heads.view(-1))                    # [batch*sent_len]
            
                #LABEL LOSS
                headsL       = heads.unsqueeze(1).unsqueeze(2)                          # [batch, 1, 1, sent_len]
                headsL       = headsL.expand(-1, lab_scores.size(1), -1, -1)            # [batch, n_labels, 1, sent_len]
                lab_scoresL  = torch.gather(lab_scores, 2, headsL).squeeze(2)           # [batch, n_labels, sent_len]
                lab_scoresL  = lab_scoresL.transpose(-1, -2)                            # [batch, sent_len, n_labels]
                lab_scoresL  = lab_scoresL.contiguous().view(-1, lab_scoresL.size(-1))  # [batch*sent_len, n_labels]
                labelsL      = labels.view(-1)                                          # [batch*sent_len]
                lab_loss     = loss_fnc(lab_scoresL, labelsL)
                
                loss         = arc_loss + lab_loss
                gloss       += loss.item()

                print('DEV',words)

                #greedy arc accurracy (without parsing)
                _, pred = arc_scores.max(dim=-2)
                mask = (heads != DependencyDataset.PAD_IDX).float()
                arc_accurracy = torch.sum((pred == heads).float() * mask, dim=-1)
                arc_acc += torch.sum(arc_accurracy).item()
            
                #greedy label accurracy (without parsing)
                _, pred = lab_scores.max(dim=1)
                pred = torch.gather(pred, 1, heads.unsqueeze(1)).squeeze(1)
                mask = (heads != DependencyDataset.PAD_IDX).float()
                lab_accurracy = torch.sum((pred == labels).float() * mask, dim=-1)
                lab_acc += torch.sum(lab_accurracy).item()
                ntoks += torch.sum(mask).item()

                overall_size += (deps.size(0)*deps.size(1))
                
        print('DL',gloss)
        return gloss/overall_size,arc_acc, lab_acc,ntoks
        
    def train_model(self,train_set,dev_set,epochs,batch_size):
        loss_fnc   = nn.CrossEntropyLoss(reduction='sum')
        optimizer  = torch.optim.Adam(self.parameters(),lr=0.001)
        for e in range(epochs):
            print('----')
            TRAIN_LOSS    =  0
            TRAIN_TOKS    =  0
            BEST_DEV_LOSS =  1000
            train_batches = train_set.make_batches(batch_size,shuffle_batches=True,shuffle_data=True,order_by_length=True)
            overall_size  = 0
            for batch in train_batches:
                self.train()
                optimizer.zero_grad()

                words,deps,heads,labels = batch
                deps, heads, labels = deps.to(self.device), heads.to(self.device), labels.to(self.device)

                print('TRAIN',words)
                
                #FORWARD
                arc_scores, lab_scores = self.forward(deps)
                #ARC LOSS
                arc_scores = arc_scores.transpose(-1, -2)                           # [batch, sent_len, sent_len]
                arc_scores = arc_scores.contiguous().view(-1, arc_scores.size(-1))  # [batch*sent_len, sent_len]
                arc_loss   = loss_fnc(arc_scores, heads.view(-1))                   # [batch*sent_len]

                #LABEL LOSS
                heads      = heads.unsqueeze(1).unsqueeze(2)                        # [batch, 1, 1, sent_len]
                heads      = heads.expand(-1, lab_scores.size(1), -1, -1)           # [batch, n_labels, 1, sent_len]
                lab_scores = torch.gather(lab_scores, 2, heads).squeeze(2)          # [batch, n_labels, sent_len]
                lab_scores = lab_scores.transpose(-1, -2)                           # [batch, sent_len, n_labels]
                lab_scores = lab_scores.contiguous().view(-1, lab_scores.size(-1))  # [batch*sent_len, n_labels]
                labels     = labels.view(-1)                                        # [batch*sent_len]
                lab_loss   = loss_fnc(lab_scores, labels)

                loss       = arc_loss + lab_loss
                TRAIN_LOSS   += loss.item()
                overall_size += (deps.size(0)*deps.size(1)) #bc no masking at training
                
                loss.backward()
                optimizer.step()

               

            print('TL',TRAIN_LOSS)
            
            DEV_LOSS,DEV_ARC_ACC,DEV_LAB_ACC,DEV_TOKS  = self.eval_model(dev_set,batch_size)
            print('Epoch ',e,'train mean loss',TRAIN_LOSS/overall_size,
                             'valid mean loss',DEV_LOSS,
                             'valid arc acc',DEV_ARC_ACC/DEV_TOKS,
                             'valid label acc',DEV_LAB_ACC/DEV_TOKS)

            if DEV_LOSS < BEST_DEV_LOSS: #there is a problem with the validation loss
                torch.save(self,'model.pt')
                BEST_DEV_LOSS = DEV_LOSS
        #torch.load best model
        
    def predict_batch(self,test_set,batch_size):

        test_batches = test_set.make_batches(batch_size,shuffle_batches=False,shuffle_data=False,order_by_length=False) #keep natural order here

        with torch.no_grad():
            
            #softmax = nn.Softmax(dim=1)
            for batch in test_batches:
                self.eval()
                words, deps,heads,labels = batch
                deps, heads, labels = deps.to(self.device), heads.to(self.device), labels.to(self.device)

                SLENGTHS = (deps != DependencyDataset.PAD_IDX).long().sum(-1)
                
                #batch prediction
                arc_scores_batch, lab_scores_batch = self.forward(deps)
                arc_scores_batch, lab_scores_batch = arc_scores_batch.cpu(), lab_scores_batch.cpu()  

                for tokens,length,arc_scores,lab_scores in zip(words,SLENGTHS,arc_scores_batch,lab_scores_batch):
                    # Predict heads
                    probs          = arc_scores.numpy().T
                    mst_heads      = chuliu_edmonds(probs)
                    #mst_heads      = chuliu_edmonds(probs[:,:length])
                    # Predict labels
                    select         = torch.LongTensor(mst_heads).unsqueeze(0).expand(lab_scores.size(0), -1)
                    select         = Variable(select)
                    selected       = torch.gather(lab_scores, 1, select.unsqueeze(1)).squeeze(1)
                    _, mst_labels  = selected.max(dim=0)
                    mst_labels     = mst_labels.data.numpy()
                    edges = [ (head,test_set.itolab[lbl],dep) for (dep,head,lbl) in zip(list(range(length)),mst_heads[:length], mst_labels[:length]) ]
                    dg = DepGraph(edges[1:],wordlist=tokens[1:])
                    print(dg)
                    print()

if __name__ == '__main__':
    
    embedding_size  = 100
    encoder_dropout = 0.0
    mlp_input       = 250
    mlp_arc_hidden  = 500
    mlp_lab_hidden  = 100
    mlp_dropout     = 0.0
    device          = "cuda:1" if torch.cuda.is_available() else "cpu"
    trainset        = DependencyDataset('../spmrl/dev.French.gold.conll',min_vocab_freq=0)
    itos,itolab     = trainset.itos,trainset.itolab
    
    parser          = BiAffineParser(len(itos),embedding_size,encoder_dropout,mlp_input,mlp_arc_hidden,mlp_lab_hidden,mlp_dropout,len(itolab),device)
    parser.train_model(trainset,trainset,60,32)
    parser.predict_batch(trainset,8)
    print('Device used', device)
