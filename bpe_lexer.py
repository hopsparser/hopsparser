import os
import torch
import torch.nn as nn
import torch.optim as optim

from XLM.src.utils import AttrDict
from XLM.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from XLM.src.model.transformer import TransformerModel

codes = "frwiki_embed1024_layers12_heads16/BPE/codes"
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
                  print(s,file=ostream)
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

  
class LexerBPE(nn.Module):
      """
      This class merges BPE vectors into word vectors.
      This is a bag of words method with dimensionality reduction
      """
      def __init__(self,model_path,word_embedding_size,bpe_embedding_size):
            """
            Args:
               model_path (string): path to model 
            """
            super(LexerBPE, self).__init__()
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
          bpe_tensor   = bpe_tensor.detach() #prevents backprop into the transformer
          bpe_tensor   = bpe_tensor.squeeze() if bpe_tensor.dim() > 2 else bpe_tensor
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

      
if __name__ == '__main__':
    
    dataset = DatasetBPE('/Users/bcrabbe/Desktop/MCVF_CORPUS/script/example.txt','train-01')
     
    print(dataset.sentences)
    lexer = LexerBPE('frwiki_embed1024_layers12_heads16/model-002.pth',256,1024)
    for seq in dataset:
        embeddings = lexer(seq)
        print(embeddings.size()) 
        print(embeddings)
