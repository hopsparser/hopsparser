import torch
import fasttext
from torch import nn
from graph_parser2 import DependencyDataset,DepGraph
from transformers  import AutoConfig, AutoModel, AutoTokenizer
from collections   import Counter,defaultdict
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
    
    itos = [DependencyDataset.PAD_TOKEN] + list(vocab)
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
            word_idxes = [word_sampler(widx,self.stoi[DependencyDataset.UNK_WORD],self._dpout) for widx in word_idxes]
        return word_idxes


# class FastTextLexer(nn.Module):
#     """
#     This is the a lexer that uses fastText embeddings.
#     FastText models are large in memory.
#     So we intersect the model with the vocabulary found in the data to
#     reduce memory usage.
#     """
#     def __init__(self,itos,dropout,ft_modelfile='cc.fr.300.bin'):
#
#         super(FastTextLexer,self).__init__()
#         FT = fasttext.load_model(ft_modelfile)
#         self.embedding_size = 300             #thats the interface property
#
#         ematrix = []
#         for word in itos:
#             ematrix.append(torch.from_numpy(FT[word]))
#         ematrix = torch.stack(ematrix)
#         self.embedding = nn.Embedding.from_pretrained(ematrix,freeze=False,padding_idx=DependencyDataset.PAD_IDX)
#         self.word_dropout   = dropout
#         self.itos      = itos
#         self.stoi      = {token:idx for idx,token in enumerate(self.itos)}
#         self._dpout    = 0
#
#     def train_mode(self):
#         self._dpout = self.word_dropout
#     def eval_mode(self):
#         self._dpout = 0
#
#     def tokenize(self,tok_sequence,word_dropout=0.0):
#         """
#         This maps word tokens to integer indexes.
#         Args:
#            tok_sequence: a sequence of strings
#         Returns:
#            a list of integers
#         """
#         word_idxes     = [self.stoi.get(token,self.stoi[DependencyDataset.UNK_WORD]) for token in tok_sequence]
#         if self._dpout > 0:
#             word_idxes = [word_sampler(widx,self.stoi[DependencyDataset.UNK_WORD],word_dropout) for widx in word_idxes]
#         return word_idxes
#
#     def forward(self,word_sequences):
#         """
#         Takes words sequences codes as integer sequences and returns the embeddings
#         """
#         return self.embedding(word_sequences)

class BertBaseLexer(nn.Module):
    """
    This Lexer performs tokenization and embedding mapping with BERT
    style models. It concatenates a standard embedding with a Flaubert
    embedding (uses Flaubert).
    """
    def __init__(self,default_itos,default_embedding_size,word_dropout,cased=False,bert_modelfile="flaubert/flaubert_base_uncased",BERT_SIZE=768):

        super(BertBaseLexer,self).__init__()
        self._embedding_size        = default_embedding_size
        self.itos                   = default_itos
        self.stoi                   = {token:idx for idx,token in enumerate(self.itos)}
        
        self.embedding              = nn.Embedding(len(self.itos), default_embedding_size, padding_idx=DependencyDataset.PAD_IDX)

        bert_config                 = AutoConfig.from_pretrained(bert_modelfile, output_hidden_states=True)
        self.bert                   = AutoModel.from_pretrained(bert_modelfile, config=bert_config)
        self.bert_tokenizer         = AutoTokenizer.from_pretrained(bert_modelfile,
                                                                    additional_special_tokens=[DependencyDataset.PAD_TOKEN],
                                                                    #padding=True,
                                                                    unk_token=DependencyDataset.UNK_WORD)#,
                                                                    #pad_token=DependencyDataset.PAD_TOKEN)
                                                                       
        print('***',self.bert_tokenizer.pad_token,self.bert_tokenizer.pad_token_id)
        self.BERT_PAD_IDX = self.bert_tokenizer.pad_token_id
        self.BERT_SIZE    = BERT_SIZE                
        #assert(self.bert_tokenizer.pad_token == DependencyDataset.PAD_TOKEN)
        #assert(self.bert_tokenizer.unk_token == DependencyDataset.UNK_WORD)
        
        self.bert_tokenizer.add_tokens([DepGraph.ROOT_TOKEN])
        self.bert.resize_token_embeddings(len(self.bert_tokenizer))
        
        self.word_dropout           = word_dropout
        self._dpout                 = 0
        self.cased                  = cased
        
    @property
    def embedding_size(self):
        return self._embedding_size + self.BERT_SIZE
    
    @embedding_size.setter
    def embedding_size(self,value):
        self._embedding_size = value + self.BERT_SIZE
    
    def train_mode(self):
         self._dpout = self.word_dropout
         self.bert.train()
         
    def eval_mode(self):
        self._dpout = 0
        self.bert.eval()
         
    def forward(self,coupled_sequences):
        """
        Takes words sequences codes as integer sequences and returns
        the embeddings from the last (top) BERT layer.
        """
        word_idxes,bert_idxes = coupled_sequences          
        #bertE                = self.bert(bert_idxes)[0]  
        bert_layers           = self.bert(bert_idxes)[-1]
        bertE                 = torch.mean(torch.stack(bert_layers[4:8]),0) #4th to 8th layers are said to encode syntax
        wordE                 = self.embedding(word_idxes)
        return torch.cat( (wordE,bertE) ,dim=2)

    def tokenize(self,tok_sequence,word_dropout=0.0):
        """
        This maps word tokens to integer indexes.
        When a word decomposes as multiple BPE units, we keep only the first (!) 
        Args:
           tok_sequence: a sequence of strings
        Returns:
           a list of integers 
        """
        word_idxes  = [self.stoi.get(token,self.stoi[DependencyDataset.UNK_WORD]) for token in tok_sequence]
        if self.cased:
            bert_idxes  = [self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(token))[0] for token in tok_sequence]
        else:
            bert_idxes  = [self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(token.lower()))[0] for token in tok_sequence]

        if self._dpout:
            word_idxes = [word_sampler(widx,self.stoi[DependencyDataset.UNK_WORD],self._dpout) for widx in word_idxes]

        #ensure that first index is <root> and not an <unk>
        word_idxes[0] = self.stoi[DependencyDataset.UNK_WORD]
        bert_idxes[0] = self.bert_tokenizer.convert_tokens_to_ids(DepGraph.ROOT_TOKEN)
        return (word_idxes,bert_idxes)


    
