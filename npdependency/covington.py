import torch
import torch.nn as nn
from tqdm import tqdm
from random import sample
from bpe_lexer import *
from deptree   import *

class CovingtonParser(nn.Module):
    
    NO_ARC    = 'N'
    LEFT_ARC  = 'L'
    RIGHT_ARC = 'R'
    SHIFT     = 'S'
 
    def __init__(self,word_embedding_size,lstm_hidden_size,hidden_size,dep_labels,dropout=0.0):

        super(CovingtonParser, self).__init__()
        self.code_actions(dep_labels)
        self.allocate(word_embedding_size,lstm_hidden_size,hidden_size,dropout) 
        
    def allocate(self,word_embedding_size,lstm_hidden_size,hidden_size,dropout):

        nactions            = len(self.itoa)
        self.Wup            = nn.Linear(hidden_size,nactions)
        self.Wbot           = nn.Linear(lstm_hidden_size*2*6,hidden_size)
        self.lstm           = nn.LSTM(word_embedding_size,lstm_hidden_size,1,bidirectional=True)
        self.softmax        = nn.LogSoftmax(dim=0)
        self.tanh           = nn.Tanh()
        self.null_vec       = torch.zeros(lstm_hidden_size*2) # x2 because bi-lstm
        self.embedding_size = word_embedding_size
        self.lstm_size      = lstm_hidden_size
        self.hidden_size    = hidden_size
        self.dropout        = nn.Dropout(p=dropout)

    def code_actions(self,dep_labels,nolabel='-'):

        self.itoa = [(CovingtonParser.SHIFT,nolabel),(CovingtonParser.NO_ARC,nolabel)] \
          + [ (CovingtonParser.LEFT_ARC,lbl)  for lbl in dep_labels ] \
          + [ (CovingtonParser.RIGHT_ARC,lbl) for lbl in dep_labels ]
        self.atoi = dict( [ (A,idx) for (idx,A) in enumerate(self.itoa)])

    def save(self,model_prefix):
        torch.save({'embedding_size'  : self.embedding_size,\
                    'lstm_hidden_size': self.lstm_size,\
                    'hidden_size'     : self.hidden_size,\
                    'dropout'         : self.dropout.p,\
                    'params'          : self.state_dict()},\
                    model_prefix+'.parser.params')
        torch.save(lexer.state_dict(),model_prefix+'.lexer.params')
        codes = open(model_prefix+'.codes','w')
        for action,label in self.itoa:
            print('%s\t%s'%(action,label),file=codes)
        codes.close( )
        print('  => params saved.')

    @staticmethod
    def load(prefix_path): 

        codes      = open(prefix_path+'.codes')
        itoa       = [ tuple(line.split()) for line in codes ]
        atoi       = dict( [ (A,idx) for (idx,A) in enumerate(itoa)])
        deplabels  = set([lbl for a,lbl in itoa if lbl != '-']) 
        codes.close() 

        matrix_reloaded     = torch.load(prefix_path+'.parser.params')
        hidden_size         = matrix_reloaded['hidden_size']
        lstm_hidden_size    = matrix_reloaded['lstm_hidden_size']
        word_embedding_size = matrix_reloaded['embedding_size']
        dropout             = matrix_reloaded['dropout']
        model               = CovingtonParser(word_embedding_size,lstm_hidden_size,hidden_size,deplabels,dropout)
        model.load_state_dict(matrix_reloaded['params'])
        model.itoa = itoa
        model.atoi = atoi
        return model 
    
    def score_actions(self,xembeddings,S1,S2,B,graph):
        """
        Scores all action and returns the softmaxed vector
        Args: 
           xembeddings (tensor): a tensor of input word embeddings
           S1            (list): the first stack, a list of int
           S2            (list): the second stack, a list of int
           B             (list): the buffer, a list of int
           graph      (DepTree): a Dependency Tree object
        """
        X1 = xembeddings[S1[-1]] if S1 else self.null_vec
        X2 = xembeddings[S1[-2]] if len(S1) > 1 else self.null_vec

        X3 = xembeddings[S2[0]]  if S2 else self.null_vec
        X4 = xembeddings[S2[-1]] if S2 else self.null_vec

        X5 = xembeddings[B[0]]   if B  else self.null_vec
        X6 = xembeddings[B[1]]   if len(B) > 1  else self.null_vec

        xinput = self.dropout(torch.cat([X1,X2,X3,X4,X5,X6]))
        h = self.tanh(self.Wbot(xinput))
        return self.softmax(self.action_mask(self.Wup(h),S1,S2,B,graph))

    def action_mask(self,xinput,S1,S2,B,graph):
        """
        Takes a graph as input and returns a masked vector where illegal actions are nullified
        """
        mask_val = {CovingtonParser.SHIFT:0.0,CovingtonParser.NO_ARC:0.0,CovingtonParser.LEFT_ARC:0.0,CovingtonParser.RIGHT_ARC:0.0}
        
        if not S1:
            mask_val[CovingtonParser.NO_ARC]    = -float('Inf')
            mask_val[CovingtonParser.LEFT_ARC]  = -float('Inf')
            mask_val[CovingtonParser.RIGHT_ARC] = -float('Inf')
        if not B:
            mask_val[CovingtonParser.SHIFT]     = -float('Inf')
            mask_val[CovingtonParser.LEFT_ARC]  = -float('Inf')
            mask_val[CovingtonParser.RIGHT_ARC] = -float('Inf')
        if S1 and B:
            i = S1[-1]
            j = B[0]
            if  mask_val[CovingtonParser.LEFT_ARC] > -float('Inf'):
                if i == -1:# <-> i is artificial root
                    mask_val[CovingtonParser.LEFT_ARC]  = -float('Inf')
                elif graph.is_dag_add(j,i):
                    mask_val[CovingtonParser.LEFT_ARC]  = -float('Inf')
                elif  graph.is_cyclic_add(j,i):
                    mask_val[CovingtonParser.LEFT_ARC]  = -float('Inf')
            if  mask_val[CovingtonParser.RIGHT_ARC] > -float('Inf'):
                if graph.is_dag_add(i,j):
                    mask_val[CovingtonParser.RIGHT_ARC] = -float('Inf')
                elif graph.is_cyclic_add(i,j):
                    mask_val[CovingtonParser.RIGHT_ARC] = -float('Inf')

        mask = torch.tensor([ mask_val[action]  for (action,label) in self.itoa ])
        return mask + xinput
 
    def forward(self,xembeddings,K=8):  
        """
        This method performs parsing with a beam of size K
        """
        #run a bilstm on input first
        xembeddings   = self.dropout(xembeddings)
        lembeddings,_ = self.lstm(xembeddings.unsqueeze(dim=0))
        lembeddings   = lembeddings.squeeze(dim=0) 
                            
        Beam      = [ (self.init_config(lembeddings.size(0)),0.0)]
        successes = [ ]
        while Beam:
            nextBeam = [ ]
            for (config,prefix_score) in Beam:
                S1,S2,B,Arcs = config
                if not B:
                    successes.append((config,prefix_score))
                else:
                    ascores = self.score_actions(lembeddings,S1,S2,B,Arcs)
                    nextBeam.extend( (config,action,prefix_score+local_score) for (action,local_score) in zip( self.itoa , ascores) if local_score > -float('Inf'))
            nextBeam.sort(reverse=True,key=lambda x:x[2])
            nextBeam = nextBeam[:K]
            Beam = [ ( self.exec_action(action,config), score) for (config,action,score) in nextBeam ]

        if not successes:
            return [ ]
        
        successes.sort(reverse=True,key=lambda x:x[1])
        (S1,S2,B,Graph), score = successes[0]
        return Graph
    
    def parse_corpus(self,bpe_dataset,sentlist,lexer,K=8,multilingual=True): 
        """
        Parses a list of raw sentences and yields a sequence of dep trees
        Args:
             bpe_dataset (DatasetBPE): a Dataset with BPE encoded sentences
             sentlist          (list): a list of sentences. A sentence is a list of strings
             lexer         (LexerBPE): a lexer mapping BPE embeddings to word embeddings
        KwArgs:
            K                   (int): the beam size
        """
        assert ( len(sentlist) == len(bpe_dataset))
        with torch.no_grad():
            self.eval()
            for idx in tqdm(range(len(sentlist))):
                if multilingual:
                    bpe_toks  = lexer.encode2bpe(' '.join(train_trees[idx].words))
                else:
                    bpe_toks  = bpe_trainset[idx]
                xembeddings = lexer.forward(' '.join(bpe_toks) )
                #print(sentlist[idx])
                deptree = self.forward(xembeddings,K)      
                deptree.words = sentlist[idx]
                yield deptree
                
    def train_model(self,bpe_trainset,train_trees,bpe_validset,valid_trees,lexer,epochs,learning_rate=0.01,modelname='xlm',multilingual=True):
        """
        Args:
            bpe_trainset(DatasetBPE): a Dataset with BPE encoded sentences
            train_trees       (list): a list of DepTree objects
            bpe_validset(DatasetBPE): a Dataset with BPE encoded sentences
            valid_trees       (list): a list of DepTree objects
            lexer         (LexerBPE): a lexer mapping BPE embeddings to word embeddings
            epochs             (int): number of epochs
            learning_rate      (int): a learning rate
        """
        loss_fn = nn.NLLLoss(reduction='sum') 
        optimizer = optim.Adagrad(list(self.parameters())+list(lexer.parameters()),lr=learning_rate)
        #print(len(train_trees), len(bpe_trainset) )
        assert ( len(train_trees) == len(bpe_trainset) )
        idxes = list(range(len(train_trees)))
        bestNLL = 10000000000000000
        for epoch in range(epochs):
            self.train()
            L = 0
            N = 0
            try:
                for idx in tqdm(sample(idxes,len(idxes))):
                    refD          = CovingtonParser.oracle_derivation( train_trees[idx] )
                    if multilingual:
                        bpe_toks  = lexer.encode2bpe(' '.join(train_trees[idx].words))
                    else:
                        bpe_toks  = bpe_trainset[idx]
                    xembeddings   = lexer.forward(' '.join(bpe_toks) )
                    lembeddings,_ = self.lstm(xembeddings.unsqueeze(dim = 0)) 
                    lembeddings   = lembeddings.squeeze(dim=0) 
                    config        = self.init_config(len(train_trees[idx].words))
                    optimizer.zero_grad() 
                    for (act_type,label) in refD:
                        S1,S2,B,Arcs = config
                        output       = self.score_actions(lembeddings,S1,S2,B,Arcs)
                        reference    = torch.tensor([self.atoi[(act_type,label)]])
                        loss         = loss_fn(output.unsqueeze(dim=0),reference)
                        loss.backward(retain_graph=True)
                        L += loss.item() 
                        config    = self.exec_action( (act_type,label), config)
                    optimizer.step() 
                    N += len(refD)
                validNLL = self.valid_model(bpe_validset,valid_trees,lexer,multilingual)
                if validNLL < bestNLL:
                    bestNLL = validNLL  
                    self.save(modelname) 
                print('\nepoch %d'%(epoch,),'train loss (avg NLL) = %f'%(L/N,),'valid loss (avg NLL) = %f'%(validNLL,),flush=True) 
            except KeyboardInterrupt:
                print('Caught SIGINT signal. aborting training immediately.')
                return None 
            
                
    def valid_model(self,bpe_dataset,ref_trees,lexer,multilingual):
        """
        Performs the validation of the model on derivation sequences
        Args:
            bpe_dataset (DatasetBPE): a Dataset with BPE encoded sentences
            ref_trees         (list): a list of DepTree objects
            lexer         (LexerBPE): a lexer mapping BPE embeddings to word embeddings
            epochs             (int): number of epochs
            learning_rate      (int): a learning rate
        """
        L = 0
        N = 0
        with torch.no_grad():
            self.eval()
            loss_fn = nn.NLLLoss(reduction='sum') 
            assert ( len(ref_trees) == len(bpe_dataset) )
        
            idxes = list(range(len(ref_trees)))        
            for idx in tqdm(sample(idxes,len(idxes))):
                refD        = CovingtonParser.oracle_derivation( ref_trees[idx] )
                if multilingual:
                    bpe_toks  = lexer.encode2bpe(' '.join(train_trees[idx].words))
                else:
                    bpe_toks  = bpe_trainset[idx]
                xembeddings = lexer.forward(' '.join(bpe_toks) )
                lembeddings,_ = self.lstm(xembeddings.unsqueeze(dim=0))
                lembeddings = lembeddings.squeeze(dim=0) 
                config      = self.init_config(len(ref_trees[idx].words))
                for (act_type,label) in refD:
                    S1,S2,B,Arcs = config
                    output    = self.score_actions(lembeddings,S1,S2,B,Arcs)
                    reference = torch.tensor([self.atoi[(act_type,label)]])
                    loss      = loss_fn(output.unsqueeze(dim=0),reference)
                    L += loss.item()
                    config    = self.exec_action( (act_type,label), config)
                N += len(refD)
                
        return L/N

            
    def exec_action(self,action,configuration):
        
        act_type,label = action
        S1,S2,B,Arcs   = configuration
        if act_type == CovingtonParser.NO_ARC:
            return self.no_arc(S1,S2,B,Arcs)
        elif act_type == CovingtonParser.SHIFT:
            return self.shift(S1,S2,B,Arcs)
        elif act_type == CovingtonParser.LEFT_ARC:
            return self.left_arc(label,S1,S2,B,Arcs)
        elif  act_type == CovingtonParser.RIGHT_ARC:
            return self.right_arc(label,S1,S2,B,Arcs)
        
    def init_config(self,N):
        """
        Returns the parser init configuration
        """
        return ([-1],[ ],list(range(N)),DepGraph([ ]))
        
    def left_arc(self,label,S1,S2,Buff,graph):
        """
        Performs a left arc action on a configuration and returns the next configuration
        """
        dep = S1[-1]
        S1  = S1[:-1] 
        S2  = [dep] + S2
        gov = Buff[0]
        graph = graph.fastcopy()
        graph.add_arc(gov,label,dep)
        return (S1,S2,Buff,graph) #TODO : perform graph copy

    def right_arc(self,label,S1,S2,Buff,graph):
        """
        Performs a right arc action on a configuration and returns the next configuration
        """
        gov = S1[-1]
        S1  = S1[:-1] 
        S2  = [gov] + S2
        dep = Buff[0]
        graph = graph.fastcopy()
        graph.add_arc(gov,label,dep)
        return (S1,S2,Buff,graph) #TODO : perform graph copy

    def no_arc(self,S1,S2,Buff,graph):
        """
        Performs a no arc action on a configuration and returns the next configuration
        """
        node = S1[-1]
        S1   = S1[:-1] 
        S2   = [node] + S2 
        return (S1,S2,Buff,graph) #TODO : perform graph copy

    def shift(self,S1,S2,Buff,graph): 
        """
        Performs a shift action on a configuration and returns the next configuration
        """
        S1  = S1 + S2 + [ Buff[0] ]
        return (S1,[ ],Buff[1:],graph) #TODO : perform graph copy
    
    @staticmethod
    def oracle_derivation(oracle_tree,nolabel='-'): 
        """
        Args:
           oracle_tree (DepGraph):
        Returns:
            a list. The derivation
        """
        D = [ ]
        for nodeJ in range(0,len(oracle_tree)):
            na_tmp = [ ]
            for nodeI in reversed(range(-1,nodeJ)):
                left_dep  = oracle_tree.get_arc(nodeJ,nodeI)
                right_dep = oracle_tree.get_arc(nodeI,nodeJ)
                if left_dep:
                    g,lbl,d = left_dep
                    D.extend(na_tmp)
                    D.append( (CovingtonParser.LEFT_ARC,lbl) )
                    na_tmp = [ ]
                elif right_dep:
                    g,lbl,d = right_dep
                    D.extend(na_tmp)
                    D.append( (CovingtonParser.RIGHT_ARC,lbl) )
                    na_tmp = [ ]
                else:
                    na_tmp.append(  (CovingtonParser.NO_ARC,nolabel) )
            D.append( (CovingtonParser.SHIFT,nolabel) )
        return D

if __name__ == "__main__":
    
    src_train   = 'spmrl/train.French.gold.conll'
    #src_train   = 'spmrl/example.txt'
    src_valid   = 'spmrl/dev.French.gold.conll'
    src_test   = 'spmrl/test.French.gold.conll'

    modelname  =  'xlm.multilingual' 
    
    def read_graphlist( src_file ):
        
        istream = open( src_file ) 
        graphList   = [ ]
        labels      = set() 
        graph       = DepGraph.read_tree( istream )
        while not graph is None:
            graphList.append( graph )
            labels.update( graph.get_all_labels() )
            graph   = DepGraph.read_tree( istream )
        istream.close()
        return labels,graphList

    labels,train_trees = read_graphlist( src_train )
    _,valid_trees      = read_graphlist( src_valid ) 
    _,test_trees       = read_graphlist( src_test ) 

    #vocabulary = set()
    #for graph in train_trees:
    #    vocabulary.update(graph.words)
    #lexer = DefaultLexer(256,list(vocabulary))
    #parser  = CovingtonParser(256,256,labels) 
    #parser.train_model([graph.words for graph in train_trees],train_trees,[graph.words for graph in valid_trees],valid_trees,lexer,4,learning_rate=0.001,modelname=modelname)

    #out = open(modelname+'.test.conll','w')
    #for g in parser.parse_corpus([ graph.words for graph in test_trees ],[ graph.words for graph in test_trees ],lexer,K=32):
    #    print(g,file=out,flush=True)
    #    print('',file=out)
    #out.close()

    #exit(0) 
    
    bpe_trainset = DatasetBPE([ ' '.join(graph.words) for graph in train_trees],modelname + '.train-spmrl')
    bpe_validset = DatasetBPE([ ' '.join(graph.words) for graph in valid_trees],modelname + '.dev-spmrl')  
    bpe_testset  = DatasetBPE([ ' '.join(graph.words) for graph in test_trees],modelname + '.test-spmrl')  

    #lexer   = SelectiveBPELexer('bert-base-lowercase/best-valid_fr_mlm_ppl.pth',768)
    lexer   = MultilingualLexer( ) 
    parser  = CovingtonParser(768,512,256,labels,dropout=0.3)   
    parser.train_model(bpe_trainset,train_trees,bpe_validset,valid_trees,lexer,15,learning_rate=0.01,modelname=modelname,multilingual=True)
  
    #lexer  = SelectiveBPELexer('frwiki_embed1024_layers12_heads16/model-002.pth',1024)
    #parser = CovingtonParser.load(modelname)
    out = open(modelname+'.test.conll','w')
    for g in parser.parse_corpus(bpe_testset,[ graph.words for graph in test_trees ],lexer,K=32,multilingual=True):
        print(g,file=out,flush=True)
        print('',file=out)          
    out.close()


