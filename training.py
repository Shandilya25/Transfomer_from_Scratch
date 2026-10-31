import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import warnings

from pathlib import Path
from dataset import BilingualDataset,casual_mask
from model import build_transformer
from config import get_configuration,get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
import tqdm

def get_all_sentences(dataset,language):
    for element in dataset:
        yield element['translation'][language]

def get_build_tokenizer(config,dataset,language):
    tokenizer_path=Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_trainer=Whitespace()#splitting by white space.
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset,language),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):

    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    tokenizer_src = get_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def greedy_decoding(model,source,source_mask,src_tokenizer,tgt_tokenizer,max_len,devices):
    sos_ids=tgt_tokenizer.token_to_id('[SOS]')
    eos_ids=src_tokenizer.token_to_ids('[EOS]')
    
    encoder_output=model.encode(source,source_mask)
    decoder_input=torch.empty(1,1).fill(sos_ids).type_as(source).to(device)

    while True:
        if decoder_input.size(1)==max_len:
            break
    
        decoder_mask=casual_mask(decoder_input.size(1)).type_as(source_mask).to(devices)
        
        out=model.decode(encoder_output,source_mask,decoder_input,decoder_mask)
        
        prob=model.project(out[:,-1])
        
        _,next_word=torch.max(prob,dim=1)
        decoder_input=torch.cat([decoder_input,torch.empty(1,1).type_as(source).fill_(next_word.item()).to(devices)],dim=1)
        
        if next_word==eos_ids:
            break
        
        return decoder_input.squeeze(0)
        

def run_validate(model,validation_dataset,src_tokenizer,tgt_tokenizer,max_len,device,msg,global_state,writer,num_examples=3):
    model.eval()
    cnt=0
    
    source_texts=[]
    expected_texts=[]
    predicted_texts=[]
    
    control_window=80
    
    with torch.no_grad:
        for batch in validation_dataset:
            count+=1
            encoder_input=batch['encoder_input'].to(device)
            encoder_mask=batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0)==1, "Batch size should be 1 for validation"
            
            model_out=greedy_decoding(model,encoder_input,encoder_mask,src_tokenizer,tgt_tokenizer,max_len,device)
            
            source_text=batch['src_text'][0]
            target_text=batch=['tgt_text'][0]
            model_out_text=tgt_tokenizer.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected_texts.append(target_text)
            predicted_texts.append(model_out_text)
            
            print_msg(f'SOURCE:{source_text}')
            print_msg(f'TARGET:{target_text}')
            print_msg(f'PREDICTED:{model_out_text}')
            
            if count==num_examples:
                break
                



def get_model(config,src_vocab_len,tgt_vocab_len):
    model=build_transformer(src_vocab_len,tgt_vocab_len,config['seq_len'],config['seq_len'],config['embedding_size'])
    return model

def train_model():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USing device:{device}")
    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)
    
    train_dataloader,val_dataloader,src_tokenizer,tgt_tokenizer=get_dataset(config)
    model=get_model(config,src_tokenizer.get_vocab_size(),tgt_tokenizer.get_vocab_size()).to(device)
    
    writer=SummaryWriter(config["experiment_name"])
    
    optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"],eps=1e-9)
    
    zeroth_epoch=0
    global_step=0
    
    if config["preload"]:
        model_filename=get_weights_file_path(config,config['preload'])
        print(f'Preloading model {model_filename}')
        state=torch.load(model_filename)
        initial_epoch=state['epoch']
        optimizer.load_state_dictionary(state['optimizer_state_dict'])
        global_step=state['global_step']
        
    loss_function=nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'),label_smoothing=0.1).to(device)
    
    for epoch in range(zeroth_epoch,config['num_epochs']):
        model.train()
        batch_iterator=tqdm(train_dataloader)
        for batch in batch_iterator:
            
            encoder_input=batch["encoder_input"].to(device)
            decoder_input=batch["decoder_input"].to(device)
            encoder_mask=batch["encoder_mask"].to(device)
            decoder_mask=batch["decoder_mask"].to(device)
            
            encoder_op=model.encode(encoder_input,encoder_mask)
            decoder_op=model.decode(encoder_op,encoder_mask,decoder_mask,decoder_mask)
            projected_op=model.project(decoder_op)
            
            
            label=batch['label'].to(device)
            
            loss=loss_fn(projected_op.view(-1,tgt_tokenizer.get_vocab_size()),label.view(-1))
            batch_iterator=set_postfix({f"loss":f"{loss.item():6.3f}"})
            
            writer.add_scale("train loss",loss.item(),global_step)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step==1
            
        model_filename=get_weights_file_path(config,f"{epoch:02d}")
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'global_step':global_step
        },model_filename)
        
if __name__=="__main__":
    warnings.filterwarnings('ignore')
    config=get_configuration()
    train_model(config)
    
            
                