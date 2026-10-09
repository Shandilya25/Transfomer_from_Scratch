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

def MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dim,head,dropout):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.heads=head
        self.dropout=dropout
        
        assert self.embedding_dim%self.heads==0,"Cannot divide the vector into given number of heads"
        
        self.embedding_div=self.embedding_dim/self.heads
        self.w_k=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.w_q=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.w_v=nn.Linear(self.embedding_dim,self.embedding_dim)
        
        self.final_w=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def find_attention(query,key,value,mask,dropout):
        
        d_k=query.shape[-1]
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        
        if mask:
            attention_scores=attention_scores.masked_fill_(mask==0,-1e9)
            
        if dropout:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ value),attention_scores

            
        
        
    def forward(self,q,k,v,mask):
        query=self.w_q(q) #-->Dimensions are(batch,Seq_len,embedding_dim) Same for both output and input
        key=self.w_k(k)   #-->Dimensions are(batch,Seq_len,embedding_dim)
        value=self.w_k(v) #-->Dimensions are(batch,Seq_len,embedding_dim)
        
        
        #Here the creation of these vectors are important because of the way we 
        # are converting or reshaping the given q,k,v vectors so that each vector into
        # a head of MH properly aligns with the inpput requirements and especially the usage 
        # of q.view(dims).transpose
        
        query=query.view(query.shape[0],query.shape[1],self.heads,self.embedding_div).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.heads,self.embedding_div).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.heads,self.embedding_div).transpose(1,2)
        
        x=MultiHeadAttention.find_attention(query,key,value,mask,self.dropout)
        # x=x.transpose(1,2)
        # x=x.view(x.shape[0],x.shape[1],self.head*self.embedding_div)
        
        ##The above method is wrong because operations like transpose,permute makes the 
        ## memory non contiguous because the strides are not looked-over so before making a 
        #  .view() method u have to use .contiguous methods
        # transpose() / permute(): Do not copy memory. They simply swap stride numbers. That is what causes non-contiguity.
        
        x=x.transpose(1,2).contiguous().view(x.dim[0],-1,self.heads*self.embedding_div)
        
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self,dropout):
        super().__init__() 
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
        
    def forward(self,x,prev_layer):
        return x+self.dropout(prev_layer(self.norm(x)))
        

        
class Encoder(nn.Module):
    def __init__(self,multi_head,norm,residual_connection,mlp):
        super().__init__()
        
        
        
        
        
        
        
        
        
        
        

