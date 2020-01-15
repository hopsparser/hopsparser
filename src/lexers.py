import torch
import fasttext
from torch import nn
from graph_parser2 import DependencyDataset,DepGraph
from collections import Counter,defaultdict
from random import random

def word_sampler(word_idx,unk_idx,dropout):
    return unk_idx if random() < dropout else word_idx
    
def make_vocab(treelist,threshold):
    """
    Extracts the set of tokens found in the data and orders it 
    """
    vocab = Counter([ word  for tree in treelist for word in tree.words ])
    vocab = set([tok for (tok,counts) in vocab.most_common() if counts > threshold])
    vocab.update([DependencyDataset.UNK_WORD])
    
    itos = [DependencyDataset.PAD_TOKEN] +list(vocab)
    return itos

class DefaultLexer(nn.Module):
    """
    This is the basic lexer wrapping an embedding layer.
    """
    def __init__(self,itos,embedding_size,word_dropout):
        
        super(DefaultLexer, self).__init__()
        self.embedding      = nn.Embedding(len(itos), embedding_size, padding_idx=DependencyDataset.PAD_IDX)
        self.embedding_size = embedding_size #thats the interface property
        self.itos           = itos
        self.stoi           = {token:idx for idx,token in enumerate(self.itos)}
        self.word_dropout   = word_dropout
        self._dpout         = 0 
        
    def train_mode(self):
        self._dpout = self.word_dropout
    def eval_mode(self):
        self._dpout = 0
        
    def forward(self,word_sequences):
        """
        Takes words sequences codes as integer sequences and returns the embeddings 
        """
        return self.embedding(word_sequences)

    def tokenize(self,tok_sequence):
        """
        This maps word tokens to integer indexes.
        Args:
           tok_sequence: a sequence of strings
        Returns:
           a list of integers
        """
        word_idxes     = [self.stoi.get(token,self.stoi[DependencyDataset.UNK_WORD]) for token in tok_sequence]
        if self._dpout > 0:
            print('@')
            word_idxes = [word_sampler(widx,self.stoi[DependencyDataset.UNK_WORD],self._dpout) for widx in word_idxes]
        return word_idxes

class FastTextLexer(nn.Module):
    """
    This is the a lexer that uses fastText embeddings.
    FastText models are large in memory. 
    So we intersect the model with the vocabulary found in the data to
    reduce memory usage. 
    """
    def __init__(self,itos,dropout,ft_modelfile='cc.fr.300.bin'):

        super(FastTextLexer,self).__init__()
        FT = fasttext.load_model(ft_modelfile)
        self.embedding_size = 300             #thats the interface property

        ematrix = []
        for word in itos:
            ematrix.append(torch.from_numpy(FT[word]))
        ematrix = torch.stack(ematrix)
        self.embedding = nn.Embedding.from_pretrained(ematrix,freeze=True,padding_idx=DependencyDataset.PAD_IDX)
        self.word_dropout   = dropout
        self.itos      = itos
        self.stoi      = {token:idx for idx,token in enumerate(self.itos)}
        self._dpout    = 0 
        
    def train_mode(self):
        self._dpout = self.word_dropout
        print('train',self._dpout)
    def eval_mode(self):
        self._dpout = 0
        print('eval',self._dpout)
        
    def tokenize(self,tok_sequence,word_dropout=0.0):
        """
        This maps word tokens to integer indexes.
        Args:
           tok_sequence: a sequence of strings
        Returns:
           a list of integers
        """
        print('Call FT',self._dpout)
        word_idxes     = [self.stoi.get(token,self.stoi[DependencyDataset.UNK_WORD]) for token in tok_sequence]
        if self._dpout > 0:
            print('*')
            word_idxes = [word_sampler(widx,self.stoi[DependencyDataset.UNK_WORD],word_dropout) for widx in word_idxes]
        return word_idxes
        
    def forward(self,word_sequences):
        """
        Takes words sequences codes as integer sequences and returns the embeddings 
        """
        return self.embedding(word_sequences)

class FlauBertBaseLexer(nn.Module):
    """
    This Lexer performs tokenization and embedding mapping with BERT
    style models. (uses Flaubert / XLM).
    !!! PERFORMS LOWERCASING !!!
    """
    def __init__(self,bert_modelfile="xlm_bert_fra_base_lower"): 

        self.embedding_size = 1024 #thats the interface property
        self.bert,_         = XLMModel.from_pretrained(modelname, output_loading_info=True)
        self.bert_tokenizer = XLMTokenizer.from_pretrained(modelname,\
                                                           do_lowercase_and_remove_accent=False,\
                                                           unk_token=DependencyDataset.UNK_WORD,\
                                                           pad_token=DependencyDataset.PAD_TOKEN)

    def train_mode(self):
        pass
    def eval_mode(self):
        pass
    
    def forward(self,word_idxes):
        """
        Takes words sequences codes as integer sequences and returns
        the embeddings from the last (top) BERT layer.
        """
        return self.bert(word_idxes)[0]
        
    def tokenize(self,tok_sequence,word_dropout=0.0):
        """
        This maps word tokens to integer indexes.
        When a word decomposes as multiple BPE units, we keep only the
        first (!) 
        Args:
           tok_sequence: a sequence of strings
        Returns:
           a list of integers
        """
        word_idxes  = [self.bert.encode(token.lower())[0] for token in tok_sequence]
        if self.word_dropout:
            word_idxes = [word_sampler(widx,word_dropout) for widx in word_idxes]
        return word_idxes


    
