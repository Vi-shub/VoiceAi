import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.utils.data import random_split,DataLoader
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
import os

from config import get_config, get_weights_file_path

from pathlib import Path
from datasets import Dataset
import pickle
import warnings

from dataset import SummaryDataset,causal_mask
from model import build_transformer


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)
    

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # try:
    #     # get the console window width
    #     with os.popen('stty size', 'r') as console:
    #         _, console_width = console.read().split()
    #         console_width = int(console_width)
    # except:
    #     # If we can't get the console width, use 80 as default
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break


def get_all_sentences(ds, article_or_summary):
    for item in ds:
        yield item['entry'][article_or_summary]


def get_or_build_tokenizer(config, ds, article_or_summary):
    tokenizer_path = Path(config['tokenizer_file'].format(article_or_summary))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds,article_or_summary), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    with open('dataset3.pkl', 'rb') as fp:
        data = pickle.load(fp)
        
    ds_raw = Dataset.from_list(data)

    #Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #keep 90% for training, rest for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = SummaryDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],config['seq_len'])
    val_ds = SummaryDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'],config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    cnt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['entry'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['entry'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))
        print(cnt)
        cnt+=1


    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    #Define the device
    torch.cuda.empty_cache()
    device = torch.device('cuda')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok= True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model.to(device)
    #tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            # model.train()

            encoder_input = batch['encoder_input'].to(device)  #(B,seq_len)
            decoder_input = batch['decoder_input'].to(device)  #(B,seq_len) 
            encoder_mask = batch['encoder_mask'].to(device)  #(B,1,1,seq_len) 
            decoder_mask = batch['decoder_mask'].to(device)  #(B,1,seq_len,seq_len) 

            #Run the tensors through the transformer
            encoder_output = model.encode(encoder_input,encoder_mask)   #(B, seq_len, d_model)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input, decoder_mask)  #(B, seq_len, d_model)
            proj_output = model.project(decoder_output)     #(B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) #(B, seq_len) 

            #(B, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size) 
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"Loss: ": f"{loss.item():6.3f}"})

            #Log the loss
            writer.add_scalar('Train loss', loss.item(), global_step)
            writer.flush()

            #Backprop the loss
            loss.backward()

            #Update the weights
            optimizer.step()
            optimizer.zero_grad()

            # run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg))

            global_step+=1
        run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg))

        #Save the model at the end of each epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)