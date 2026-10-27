import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,dataset,src_tokenizer,tgt_tokenizer,src_language,tgt_language,seq_len):
        super().__init__()
        self.dataset=dataset
        self.seq_len=seq_len
        self.src_lang=src_language
        self.tgt_lang=tgt_language
        self.src_tokenizer=src_tokenizer
        self.tgt_tokenizer=tgt_tokenizer
        self.sos_token=torch.tensor([src_tokenizer.token_to_id(['[SOS]'])],dtype=torch.int64)
        self.eos_token=torch.tensor([src_tokenizer.token_to_id(['[EOS]'])],dtype=torch.int64)
        self.pad_token=torch.tensor([src_tokenizer.token_to_id(['[PAD]'])],dtype=torch.int64)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        
        src_target_pair=self.dataset[index]
        src_text=src_target_pair['translation'][self.src_lang]
        tgt_text=src_target_pair['translation'][self.tgt_lang]
        
        src_tokens=self.src_tokenizer.encode(src_text).ids
        tgt_tokens=self.tgt_tokenizer.encode(tgt_text).ids
        
        src_padding_tokens=self.seq_len-len(src_tokens)-2 # to make it up for sos,eos
        tgt_padding_tokens=self.seq_len-len(tgt_tokens)-1 # we dont add sos token to decoder
        
        if src_padding_tokens<0 or tgt_padding_tokens<0:
            raise ValueError('Sentence is long')
        
        encoder_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(src_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*src_padding_tokens,dtype=torch.int64)
                
            ]
        )
        
        decoder_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(tgt_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*tgt_padding_tokens,dtype=torch.int64)
            ]
        )
        
        label=torch.cat([
            torch.tensor(tgt_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*tgt_padding_tokens,dtype=torch.int64)
        ])
        
        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len
        
        return {
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "encoder_mask":(encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask":(decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), #(1,seq_len)&(1,seq_len,seq_len)   
            "src_text":src_text,
            "tgt_text":tgt_text,
            "label":label                    
        }
        
def casual_mask(seq_len):
    mask=torch.triu(torch.ones(1,seq_len,seq_len),diagonal=1).type(torch.int64)
    return mask==0
    