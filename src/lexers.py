import torch
import fasttext
from torch import nn
from graph_parser2 import DependencyDataset,DepGraph

class DefaultLexer(nn.Module):
    """
    This is the basic lexer wrapping an embedding layer.
    """
    def __init__(self,vocab_size,embedding_size):
        
        super(DefaultLexer, self).__init__()
        self.embedding      = nn.Embedding(vocab_size, embedding_size, padding_idx=DependencyDataset.PAD_IDX)
        self.embedding_size = embedding_size #thats the interface property

    def forward(self,word_sequences):
        """
        Takes words sequences codes as integer sequences and returns the embeddings 
        """
        return self.embedding(word_sequences)
 
class FastTextLexer(nn.Module):
    """
    This is the a lexer that uses fastText embeddings.
    FastText models are large in memory. 
    So we intersect the model with the vocabulary found in the data to
    reduce memory usage. 
    """
    def __init__(self,itos,ft_modelfile='cc.fr.300.bin',dropout = 0.0):

        super(FastTextLexer, self).__init__()

        FT = fasttext.load_model(ft_modelfile)
        self.embedding_size = 300 #thats the interface property

        ematrix = []
        for word in itos:
            ematrix.append(torch.from_numpy(FT[word]))
        ematrix = torch.stack(ematrix)
        self.embedding = nn.Embedding.from_pretrained(ematrix,freeze=True,padding_idx=DependencyDataset.PAD_IDX)
        self.dropout   = nn.Dropout(p=dropout)

        
    def forward(self,word_sequences):
        """
        Takes words sequences codes as integer sequences and returns the embeddings 
        """
        return self.dropout(self.embedding(word_sequences))

    @staticmethod
    def update_vocab(filename,vocab=None):
        """
        May extract vocab from treebanks out of the regular encoding context.
        This is used here because fasttext is built for managing <unk> words
        """
       
        vocab = set()  if vocab is None else set(vocab)
        istream       = open(filename)
        tree = DepGraph.read_tree(istream) 
        while tree:
            vocab.update(tree.words)
            tree = DepGraph.read_tree(istream)
        istream.close()
        vocab.update([DependencyDataset.UNK_WORD])
        itos = [DependencyDataset.PAD_TOKEN] +list(vocab)
        return itos
