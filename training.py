import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

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
                