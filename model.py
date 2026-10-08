import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    
    def __init__(self,vocab_size:int,embedding_dim:int):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        
    def forward(self,x):
        return self.embedding(x) * (math.sqrt(self.embedding_dim))
    
class PositionalEncoding(nn.Module):
    def __init__(self,embedding_size:int,seq_len:int,dropout:float)->None:
        '''Seq_len is to get the max sentence size which can be used for furture purpose such as padding of sentence etc.'''
        self.embedding_size=embedding_size
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        
        #We will create a matrix of embedding_dim x Sq_len. 
        # Imp:If it needs to move with the model (e.g., to GPU via .cuda()):
        #  Use PyTorch's register_buffer so it behaves correctly without being treated as a trainable parameter
        #and will be saved along with the state of the model
        matrix=torch.zeros([self.embedding_dim,self.seq_len])
        
        pos_vector=torch.arange(0,self.seq_len,dtype=torch.float).unsqueeze(1)
        value_term=torch.exp(torch.arange(0,seq_len,2).float() * (-math.log(10000.0)/self.embedding_size))
        pos_vector[:,0::2]=torch.sin(pos_vector*value_term)
        pos_vector[:,1::2]=torch.cos(pos_vector*value_term)
        
        pos_vector=pos_vector.unsqueeze(0)#(1,seq_len,embedding_dim)
        
        self.register_buffer('position_encoding',pos_vector)
        
    def forward(self,x):
        x=x+(self.pos_vector[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    
    def __init__(self,eps):
        #eps is added to standard_dev under root==(root((sigma)^2+eps))
        super().__init__()
        self.eps=eps    
        self.gamma=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.zeros(1))   
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.gamma*(x-mean)/(std+self.eps) + self.beta


class ff_nn(nn.Module):
    def __init__(self,input_dim,output_dim,dropout):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.linear1=nn.Linear(input_dim,output_dim)
        self.dropout=nn.dropout(dropout)
        self.linear2=nn.Linear(output_dim,input_dim)
    
    def forward(self,x):
        #inp: batch x seq_len x embedding_dim -> batch_seq x len x output_dim -> batch x seq_len x embedding_dim
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

        
        
        

        
        
        
        
        
        
        
        
        
        
        
        

