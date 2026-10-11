import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BillingualDataset, casual_mask
from config import *
from transformer import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pathlib import Path

def get_all_sentence(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=config['vocab_size'], special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentence(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_datasets(config):
    print(f'{config["lang_src"]}-{config["lang_tgt"]}')
    ds_raw = load_dataset('Helsinki-NLP/opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


    train_ds = BillingualDataset(train_ds_raw,
                                tokenizer_src,
                                tokenizer_tgt,
                                config['lang_src'],
                                config['lang_tgt'],
                                config['seq_len'])

    
    val_ds = BillingualDataset(val_ds_raw,
                                tokenizer_src,
                                tokenizer_tgt,
                                config['lang_src'],
                                config['lang_tgt'],
                                config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, shuffle=True, batch_size=1)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, 
                              config['seq_len'], config['seq_len'],
                              config['d_model'])

    return model

def train_model(config):
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloder, val_dataloader, tokenizer_src, tokenizer_tgt = get_datasets(config)
    model = get_model(config, tokenizer_src.get_vocab_size(),
                       tokenizer_tgt.get_vocab_size())

    writer = SummaryWriter(config['expirement_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    epoch_intial = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading... from {model_filename}")
        state = torch.load(model_filename)
        epoch_intial = state['epoch']
        optimizer.load_state_dict(state['optim_state_dict'])
        global_step = state['global_step']

    #learn label smoothing
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'))

    for epoch in range(epoch_intial, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloder, desc=f'Processing epoch{epoch}')

        for batch in batch_iterator:
            enc_inp = batch['encoder_input']
            dec_inp = batch['decoder_input']
            enc_mask = batch['encoder_mask']
            dec_mask = batch['decoder_mask']

            #Run tensor thru transformer and then calc loss and backprop
            enc_out = model.encode(enc_inp, enc_mask) # (B, _nseq, d_model)
            dec_out = model.decode(enc_out, enc_mask, dec_inp, dec_mask) #(B, _nseq, d_model)
            proj_out = model.project(dec_out) #(B, _nseq, _n_tgt_vocab)

            label = batch['label'] #(B, _nseq)

            #(B, _nseq, _n_tgt_vocab) --> (B*_nseq, _n_tgt_vocab)
            loss = loss_fn(proj_out.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss":f"{loss.item()}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()


            optimizer.step()
            optimizer.zero_grad()

            global_step+=1

        model_filename = get_weights_file_path(config, f'{epoch}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    config = get_config()
    train_model(config)