import torch
import numpy as np
from deptree import *
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from random import sample,shuffle,random
from collections import Counter,defaultdict

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
            if len(tree) <= 30: 
                self.treelist.append(tree)
            else:
                print('dropped sentence',len(tree))
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
    
    def shuffle(self):
        n = len(self.deps)
        order = list(range(n))
        shuffle(order)
        self.deps   = [self.deps[i] for i in order]
        self.heads  = [self.heads[i] for i in order]
        self.labels = [self.labels[i] for i in order]
        self.words  = [self.words[i] for i in order]
    
    def make_batches(self, batch_size,shuffle_data=True):
        #TODO provide length ordering options and options for not shuffling
        #shuffle causes a bug 
        N = len(self.deps)
        batch_order = list(range(0,N, batch_size))
        if shuffle_data:  
            self.shuffle()
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
        
    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2)
        return S.squeeze(1)


def mst(scores, eps=1e-10):
    """
    Chu-Liu-Edmonds' algorithm for finding minimum spanning arborescence in graphs.
    Calculates the arborescence with node 0 as root.
    :param scores: `scores[i][j]` is the weight of edge from node `j` to node `i`.
    :returns an array containing the head node (node with edge pointing to current node) for each node,
             with head[0] fixed as 0
    """
    scores = scores.T
    length = scores.shape[0]
    scores = scores * (1 - np.eye(length)) # mask all the diagonal elements wih a zero
    heads = np.argmax(scores, axis=1) # THIS MEANS THAT scores[i][j] = score(j -> i)!
    heads[0] = 0 # the root has a self-loop to make it special
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1
    if len(roots) < 1:
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / (head_scores + eps))]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(
            scores[roots, new_heads] / (root_scores + eps))]
        heads[roots] = new_heads
        heads[new_root] = 0

    edges = defaultdict(set) # head -> dep
    vertices = set((0,))
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)),
               np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / (old_scores + eps)
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)

    return heads


def _find_cycle(vertices, edges):
    """
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
    """
    _index = [0]
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []

    def _strongconnect(v):
        _indices[v] = _index[0]
        _lowlinks[v] = _index[0]
        _index[0] += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()
            while True:
                w = _stack.pop()
                _onstack[w] = False
                SCC.add(w)
                if not (w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


    
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
        self.device = torch.device(device)
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

        loss_fnc   = nn.CrossEntropyLoss()

        #Note: the accurracy scoring is approximative and cannot be interpreted as an UAS/LAS score !
        
        self.eval()
        dev_batches = dev_set.make_batches(batch_size)
        arc_acc, lab_acc,gloss,ntoks = 0, 0, 0, 0

        with torch.no_grad():
            for batch in dev_batches:
                words, deps, heads, labels = batch
                deps, heads, labels = deps.to(self.device), heads.to(self.device), labels.to(self.device)
                #preds 
                arc_scores, lab_scores = self.forward(deps)
                
                #get global loss
                #ARC LOSS
                arc_scoresL = arc_scores.transpose(-1, -2)                             # [batch, sent_len, sent_len]
                arc_scoresL = arc_scores.contiguous().view(-1, arc_scoresL.size(-1))   # [batch*sent_len, sent_len]
                arc_loss    = loss_fnc(arc_scoresL, heads.view(-1))                    # [batch*sent_len]
            
                #LABEL LOSS
                headsL       = heads.unsqueeze(1).unsqueeze(2)                          # [batch, 1, 1, sent_len]
                headsL       = headsL.expand(-1, lab_scores.size(1), -1, -1)            # [batch, n_labels, 1, sent_len]
                lab_scoresL  = torch.gather(lab_scores, 2, headsL).squeeze(2)           # [batch, n_labels, sent_len]
                lab_scoresL  = lab_scoresL.transpose(-1, -2)                            # [batch, sent_len, n_labels]
                lab_scoresL  = lab_scoresL.contiguous().view(-1, lab_scoresL.size(-1))  # [batch*sent_len, n_labels]
                labelsL      = labels.view(-1)                                          # [batch*sent_len]
                lab_loss     = loss_fnc(lab_scoresL, labelsL)

                #print(arc_loss.item(),lab_loss.item())
                gloss       += arc_loss.item() + lab_loss.item()
            
                #arc accurracy (without ensuring parsing)
                _, pred = arc_scores.max(dim=-2)
                mask = (heads != DependencyDataset.PAD_IDX).float()
                arc_accurracy = torch.sum((pred == heads).float() * mask, dim=-1)
                arc_acc += torch.sum(arc_accurracy).item()
            
                #labels accurracy (wihtout parsing)
                _, pred = lab_scores.max(dim=1)
                pred = torch.gather(pred, 1, heads.unsqueeze(1)).squeeze(1)
                mask = (heads != DependencyDataset.PAD_IDX).float()
                lab_accurracy = torch.sum((pred == labels).float() * mask, dim=-1)
                lab_acc += torch.sum(lab_accurracy).item()
                ntoks += torch.sum(mask).item()
                
        return gloss,arc_acc, lab_acc,ntoks
        
    def train_model(self,train_set,dev_set,epochs,batch_size):
        loss_fnc   = nn.CrossEntropyLoss()
        optimizer  = torch.optim.Adam(self.parameters())
        for e in range(epochs):
            print('----')
            TRAIN_LOSS    =  0
            TRAIN_TOKS    =  0
            BEST_DEV_LOSS =  1000
            train_batches = train_set.make_batches(batch_size)
            for batch in train_batches:
                self.train()
                words,deps,heads,labels = batch  #POTENTIAL BUG : check that the batches encodings are compliant with the other implementation
                deps, heads, labels = deps.to(self.device), heads.to(self.device), labels.to(self.device)

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
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                TRAIN_TOKS   += torch.sum((heads != DependencyDataset.PAD_IDX).float()).item()
                TRAIN_LOSS   += loss.item()

            DEV_LOSS,DEV_ARC_ACC,DEV_LAB_ACC,DEV_TOKS  = self.eval_model(dev_set,batch_size)
            
            print('Epoch ',e,'train mean loss',TRAIN_LOSS/TRAIN_TOKS,
                             'valid mean loss',DEV_LOSS/DEV_TOKS,
                             'valid arc acc',DEV_ARC_ACC/DEV_TOKS,
                             'valid label acc',DEV_LAB_ACC/DEV_TOKS)

            if DEV_LOSS < BEST_DEV_LOSS: #there is a problem with the validation loss
                torch.save(self,'model.pt')
                BEST_DEV_LOSS = DEV_LOSS
        #torch.load best model
        
    def predict_batch(self,test_set,batch_size):
        #that's semi-batched...
        
        self.eval()
        test_batches = test_set.make_batches(batch_size) #keep natural order here
        for batch in test_batches:
            words, deps,heads,labels = batch  #POTENTIAL BUG : check that the batches encodings are compliant with the other implementation
            deps, heads, labels = deps.to(self.device), heads.to(self.device), labels.to(self.device)

            SLENGTHS = (deps != DependencyDataset.PAD_IDX).long().sum(-1)
            #batch prediction
            arc_scores_batch, lab_scores_batch = self.forward(deps)
            arc_scores_batch, lab_scores_batch = arc_scores_batch.cpu(), lab_scores_batch.cpu()  
            
            for tokens,length,arc_scores,lab_scores in zip(words,SLENGTHS,arc_scores_batch, lab_scores_batch):
                # Predict heads
                scores     = arc_scores.data.numpy()
                mst_heads  = mst(scores)
                # Predict labels
                select         = torch.LongTensor(mst_heads).unsqueeze(0).expand(lab_scores.size(0), -1)
                select         = Variable(select)
                selected       = torch.gather(lab_scores, 1, select.unsqueeze(1)).squeeze(1)
                _, mst_labels  = selected.max(dim=0)
                mst_labels     = mst_labels.data.numpy()
                #print(tokens)
                #print(mst_heads[:length], mst_labels[:length])
                edges = [ (head,test_set.itolab[lbl],dep) for (dep,head,lbl) in zip(list(range(length)),mst_heads[:length], mst_labels[:length]) ]
                dg = DepGraph(edges[1:],wordlist=tokens[1:])
                print(dg)
                print()

            
if __name__ == '__main__':
    
    embedding_size  = 100
    encoder_dropout = 0.0
    mlp_input       = 100
    mlp_arc_hidden  = 50
    mlp_lab_hidden  = 50
    mlp_dropout     = 0.0
    device          = "cuda:1" if torch.cuda.is_available() else "cpu"
    trainset    = DependencyDataset('../spmrl/example.txt',min_vocab_freq=0)
    itos,itolab = trainset.itos,trainset.itolab
    
    parser      = BiAffineParser(len(itos),embedding_size,encoder_dropout,mlp_input,mlp_arc_hidden,mlp_lab_hidden,mlp_dropout,len(itolab),device)
    parser.train_model(trainset,trainset,60,4)
    parser.predict_batch(trainset,4)
    print('Device used', device)
