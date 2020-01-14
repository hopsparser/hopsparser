from torch import nn
from graph_parser2 import DependencyDataset

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

    pass
