import os
import torch
import torch.nn as nn
from transformers import *

#import torch.optim as optim
#from XLM.src.utils import AttrDict
#from XLM.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
#from XLM.src.model.transformer import TransformerModel


codes = "bert-base-lowercase/BPE/codes"
fastbpe = os.path.join(os.getcwd(), 'XLM/tools/fastBPE/fast')

class DatasetBPE:
      """
      This is a dataset for yielding BPE tokenized sentences
      """
      def __init__(self,sent_list,dataset_id):
            """
            Args:
               sentlist   (list): a list of dataset sentences
               dataset_id  (str): a unique ID specific to this datasetfile
            """
            self.sentences = self.to_bpe(sent_list,dataset_id)
            self.reset() 
      
      def to_bpe(self,sent_list,dataset_id):
            # write sentences to tmp file
            ostream = open('/tmp/%s'%(dataset_id,), 'w')
            for s in sent_list:
                  s = s.replace('-RRB-',')').replace('-LRB-','(').lower()
                  print(s,file=ostream) #rebrackets and lowers the full dataset
            #print(file=ostream) 
            ostream.close() 
            # apply bpe to tmp file
            print('%s applybpe /tmp/%s.bpe /tmp/%s %s'% (fastbpe,dataset_id,dataset_id,codes))    
            os.system('%s applybpe /tmp/%s.bpe /tmp/%s %s'% (fastbpe,dataset_id,dataset_id,codes)) 
    
            # load bpe-ized sentences 
            sentences_bpe = [ ]
            istream = open('/tmp/%s.bpe'%(dataset_id))
            for line in istream:
                  sentences_bpe.append(line.rstrip())
            istream.close()
            return sentences_bpe

      def __getitem__(self,idx): 
        return self.sentences[idx]
          
      def reset(self):
        self.idx = -1
        
      def __len__(self):
        return len(self.sentences)
        
      def __iter__(self):
        return self

      def __next__(self):
        if self.idx < len(self.sentences)-1:
          self.idx += 1
          return self.sentences[self.idx]
        raise StopIteration

  
class DefaultLexer(nn.Module):
      """
      That's a default lexer with simple embeddings
      """
      UNK_TOKEN = '<unk>'
      
      def __init__(self,word_embedding_size,vocabulary):

        super(DefaultLexer, self).__init__()

        self.code_symbols(vocabulary)
        self.allocate(word_embedding_size)
        
      def code_symbols(self,vocabulary):
        vocabulary.append(DefaultLexer.UNK_TOKEN)
        self.itos = list(vocabulary)
        self.stoi = dict([(word,idx) for idx,word in enumerate(vocabulary)])
        
      def allocate(self,word_embedding_size):
        self.embedding = nn.Embedding(len(self.itos),word_embedding_size)


      def forward(self,word_sequence):
        xinput = torch.LongTensor( [self.stoi.get(elt,self.stoi[DefaultLexer.UNK_TOKEN]) for elt in word_sequence] )
        return self.embedding(xinput)

class MultilingualLexer(nn.Module):
    """
    This returns embeddings from multilingual BERT
    """
    def __init__(self):

        super(MultilingualLexer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.transformer = BertModel.from_pretrained('bert-base-multilingual-uncased')

    def encode2bpe(self,text):
        """
        Turns a normal string into a bpe string
        """
        return self.tokenizer.tokenize(text)
        
    def forward(self,bpe_sequence):
        """
        Generates an embedding sequence for the bpe encoded sentence
        """
        bpe_sequence = bpe_sequence.split()
        bpe_sequence = [bpe_tok for bpe_tok in bpe_sequence if not bpe_tok.startswith('##')]
        tok_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(bpe_sequence)])
        hidden,attention = self.transformer(tok_tensor)
        hidden = hidden.squeeze(dim=0)
        hidden = hidden.detach()
        return hidden
        
class SelectiveBPELexer(nn.Module):
    """
    This class selects one BPE to be the the word vector.
    """    
    def __init__(self,model_path,bpe_embedding_size):
        """
        Args:
        model_path (string): path to model 
        """
        super(SelectiveBPELexer, self).__init__()
        self.load_transformer(model_path)
        
    def load_transformer(self,path):
      """
      Loads the transformer model from path
      """
      def fix_state_dict(state_dict):
        # create a new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

   
      reloaded = torch.load(path,map_location=torch.device('cpu'))
      params = AttrDict(reloaded['params']) 
      self.dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
      params.n_words    = len(self.dico)
      params.bos_index  = self.dico.index(BOS_WORD)
      params.eos_index  = self.dico.index(EOS_WORD)
      params.pad_index  = self.dico.index(PAD_WORD)
      params.unk_index  = self.dico.index(UNK_WORD)
      params.mask_index = self.dico.index(MASK_WORD)
      # build model / reload weights
      self.transformer = TransformerModel(params, self.dico, True, True)
      self.transformer.eval()
      #self.transformer.load_state_dict(fix_state_dict(reloaded['model']))          
      self.transformer.load_state_dict(reloaded['model'])          

    def forward(self,bpe_string):  
      """
      Aggregates a BPE to yield a word sequence encoding.
      Works for a single sentence. Does not support batching.
      Args:
          bpe_string (list): a list of strings, the BPE
          Returns:
             a tensor with n rows. Each row is the embedding of a word in a sentence w_1 ... w_N
      """
      bpe_sequence = bpe_string.split()
      sidxes       = torch.LongTensor([self.dico.index(w) for w in bpe_sequence]).unsqueeze(dim=1)
      L            = torch.LongTensor([len(bpe_sequence)])
      bpe_tensor   = self.transformer('fwd',x=sidxes,lengths= L,langs=None, causal=False).contiguous()
      bpe_tensor   = bpe_tensor.detach() #prevents backprop into the transformer
      bpe_tensor   = bpe_tensor.squeeze() if bpe_tensor.dim() > 2 else bpe_tensor
      bpe_tensor   = bpe_tensor.unsqueeze(dim=0) if bpe_tensor.dim() < 2 else bpe_tensor

      emb_buffer    = [ ]
      word_sequence = [ ]
      for (bpe_tok,bpe_vec) in zip(bpe_sequence,bpe_tensor):
        emb_buffer.append(bpe_vec) #if crash here, check the bpe_embeddings_size of the model
        if not bpe_tok.endswith('@@'):
          word_sequence.append(emb_buffer[0]) #first bpe only !
          emb_buffer.clear()
      return torch.stack(word_sequence)

class AveragingBPELexer(nn.Module):
      """
      This class merges BPE vectors into word vectors.
      This is a bag of words method with dimensionality reduction
      """
      def __init__(self,model_path,word_embedding_size,bpe_embedding_size):
            """
            Args:
               model_path (string): path to model 
            """
            super(AveragingBPELexer, self).__init__()
            self.load_transformer(model_path)
            self.allocate(word_embedding_size,bpe_embedding_size)

      @staticmethod
      def load(lexer_path,transformer_path):
            model = LexerBPE(transformer_path,256,1024)
            model.load_state_dict(torch.load(lexer_path+'.lexer.params'))
            return model 
        
      def load_transformer(self,path):
          reloaded = torch.load(path,map_location=torch.device('cpu'))
          params = AttrDict(reloaded['params']) 
          self.dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
          params.n_words    = len(self.dico)
          params.bos_index  = self.dico.index(BOS_WORD)
          params.eos_index  = self.dico.index(EOS_WORD)
          params.pad_index  = self.dico.index(PAD_WORD)
          params.unk_index  = self.dico.index(UNK_WORD)
          params.mask_index = self.dico.index(MASK_WORD)
          # build model / reload weights
          self.transformer = TransformerModel(params, self.dico, True, True)
          self.transformer.eval()
          self.transformer.load_state_dict(reloaded['model'])          

      def allocate(self,word_embedding_size,bpe_embedding_size):

          self.W = nn.Linear(bpe_embedding_size,word_embedding_size)
          self.tanh = nn.Tanh()
    
      def forward(self,bpe_string):  
          """
          Aggregates a BPE to yield a word sequence encoding.
          Works for a single sentence. Does not support batching.
          Args:
             bpe_string (list): a list of strings, the BPE
          Returns:
             a tensor with n rows. Each row is the embedding of a word in a sentence w_1 ... w_N
          """
          bpe_sequence = bpe_string.split()
          sidxes       = torch.LongTensor([self.dico.index(w) for w in bpe_sequence]).unsqueeze(dim=1)
          L            = torch.LongTensor([len(bpe_sequence)])
          bpe_tensor   = self.transformer('fwd',x=sidxes,lengths= L,langs=None, causal=False).contiguous()
          bpe_tensor   = bpe_tensor.detach () #prevents backprop into the transformer
          bpe_tensor   = bpe_tensor.squeeze()        if bpe_tensor.dim() > 2 else bpe_tensor 
          bpe_tensor   = bpe_tensor.unsqueeze(dim=0) if bpe_tensor.dim() < 2 else bpe_tensor 

          emb_buffer    = [ ]
          word_sequence = [ ]
          for (bpe_tok,bpe_vec) in zip(bpe_sequence,bpe_tensor):
                emb_buffer.append(self.tanh(self.W(bpe_vec))) #if crash here, check the bpe_embeddings_size of the model
                if not bpe_tok.endswith('@@'):
                  word_sequence.append(torch.stack(emb_buffer).sum(dim=0))
                  #word_sequence.append(emb_buffer[0]) #first bpe only !
                  emb_buffer.clear()
          return torch.stack(word_sequence)
