# python3 train_textworld2.py --data=../../tw_data --gamefile ../../tw_data


import torch
from torch import nn
from torch import optim

import numpy as np

import textworld
from transformers import BartConfig, T5Config
from transformers import BartTokenizerFast, T5TokenizerFast
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import AdamW
from transformers.models.bart.modeling_bart import BartEncoder

import argparse
import os
from tqdm import tqdm
import copy
import json
import logging
import random
import glob

from metrics.tw_metrics import consistencyCheck
from data.textworld.tw_dataloader import TWDataset, TWFullDataLoader


def eval_checkpoint(
    args, i, model, dev_dataloader, save_path=None, best_val_loss=float("inf"),
):
    model.eval()
    stdout_message = [f"EPOCH {i}"]
    avg_val_loss = 0
    n_val = 0
    print("Evaluating lang model")
    with torch.no_grad():
        tot_val_loss = 0
        for j, (inputs, lang_tgts, init_state, tgt_state, game_ids, entities) in enumerate(tqdm(dev_dataloader)):
            return_dict = model(
                input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=lang_tgts['input_ids'], return_dict=True,
            )
            lang_loss = return_dict.loss
            tot_val_loss += lang_loss.item()
            n_val += len(inputs['input_ids'])
    avg_val_loss = tot_val_loss / n_val
    stdout_message.append(f"OVERALL: avg val loss - {avg_val_loss}")
    print("; ".join(stdout_message))

    # save checkpoints
    new_best_loss = avg_val_loss <= best_val_loss
    if save_path is not None:
        if new_best_loss:
            print("NEW BEST MODEL")
            model.epoch = i
            torch.save(model.state_dict(), save_path)
        else:
            print(f"model val loss went up")
    return avg_val_loss, new_best_loss


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='bart', choices=['bart', 't5'])
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--eval_batchsize', type=int, default=32)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--eval_only', action='store_true', default=False)
parser.add_argument('--gamefile', type=str, required=False)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--no_pretrain', action='store_true', default=False)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_data_size', type=int, default=4000)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--local_files_only', action='store_true', default=False, help="use pretrained checkpoints saved in local directories")
args = parser.parse_args()

arch = args.arch
pretrain = not args.no_pretrain
batchsize = args.batchsize
eval_batchsize = args.eval_batchsize
max_seq_len = args.max_seq_len
inform7_game = None
# maybe inexact (grammars between worlds might be different(?)) but more efficient
for fn in glob.glob(os.path.join(args.gamefile, 'train/*.ulx')):
    env = textworld.start(fn)
    game_state = env.reset()
    game_kb = game_state['game'].kb.inform7_predicates
    inform7_game = env._inform7
    break

# seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

from transformers import PreTrainedTokenizerBase, BatchEncoding
import torch

class WordTokenizer(PreTrainedTokenizerBase):
    def __init__(self, **kwargs):
        self.itos = ["OOV", "<EOS>", "<BOS>", "<PAD>"]
        self.stoi = {x : i for i, x in enumerate(self.itos)}
        self._pad_token = self.stoi["<PAD>"]
        self.model_max_length = 200
        pass

    def tokenize(self, text, **kwargs):
        split = text.replace(".", "").replace(",", "").split(" ")
        return split
    def convert_tokens_to_ids(self, split):
        result = []
        for x in split:
            if x not in self.stoi and len(self.itos) < 10000:
                self.stoi[x] = len(self.stoi)
                self.itos.append(x)
            result.append(self.stoi.get(x, 0))
        return torch.LongTensor(result[:self.model_max_length])
    def batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        converted = [self.convert_tokens_to_ids(self.tokenize(x)) for x in batch_text_or_text_pairs]
        max_size = max([x.size()[0] for x in converted])
        for i in range(len(converted)):
            converted[i] = torch.cat([converted[i], self._pad_token + torch.zeros(max_size - converted[i].size()[0])], dim=0)
        converted = torch.stack(converted, dim=0)
        return BatchEncoding({'input_ids' : converted, 'attention_mask' : (converted != self._pad_token).long()})
# get arch-specific settings and tokenizers
if arch == 'bart':
    model_class = BartForConditionalGeneration
    config_class = BartConfig
    model_fp = 'facebook/bart-base'
    tokenizer = WordTokenizer() #BartTokenizerFast.from_pretrained(model_fp, local_files_only=args.local_files_only)
elif arch == 't5':
    model_class = T5ForConditionalGeneration
    config_class = T5Config
    model_fp = 't5-base'
    tokenizer = T5TokenizerFast.from_pretrained(model_fp, local_files_only=args.local_files_only)
else:
    raise NotImplementedError()


transformer = torch.nn.TransformerEncoder(encoder_layer=torch.nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=8)
embedding = torch.nn.Embedding(num_embeddings=10000, embedding_dim=512)
output = torch.nn.Linear(512, 1, bias=False)

def parameters():
    for x in [transformer, embedding, output]:
        for y in x.parameters():
            yield y


#model.to(args.device)

# load optimizer
all_parameters = [p for p in parameters() if p.requires_grad]
optimizer = AdamW(all_parameters, lr=args.lr)



# load data
dev_dataset = TWDataset(
    args.data, tokenizer, 'dev', max_seq_len, max_data_size=500, inform7_game=inform7_game,
)
dataset = TWDataset(
    args.data, tokenizer, 'train', max_seq_len, max_data_size=4000, inform7_game=inform7_game,
)
train_dataloader = TWFullDataLoader(dataset, args.gamefile, tokenizer, batchsize, device=args.device)
dev_dataloader = TWFullDataLoader(dev_dataset, args.gamefile, tokenizer, eval_batchsize, device=args.device)
print(f"Loaded data: {len(dataset)} train examples, {len(dev_dataset)} dev examples")

output_json_fn = None
if args.eval_only:
    output_json_fn = f"{save_path[:-2]+f'{args.num_samples}_samples.jsonl'}"
    print(f"Saving predictions to {output_json_fn}")

# Initial eval
print("Initial eval")
avg_val_loss = 0
#results = eval_checkpoint(args, "INIT", model, dev_dataloader)
#avg_val_loss += results[0]
#print(f"CONSISTENCY: loss - {results[0]}")
best_loss_epoch = -1
best_val_loss = avg_val_loss

if args.eval_only:
    exit()


print("Start training")
num_updates = 0
best_update = 0
for i in range(args.epochs):
    if i - best_loss_epoch > args.patience: break
    #model.train()
    lang_train_losses = []

    for j, (inputs, lang_tgts, init_state, tgt_state, game_ids, entities) in enumerate(train_dataloader):
        optimizer.zero_grad()
        print(inputs)
        print(lang_tgts)
        print(init_state)
        print(tgt_state)
        print(game_ids)
        print(entities)
        quit()



        return_dict = model(
            input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=lang_tgts['input_ids'], return_dict=True,
        )
        lang_loss, dec_output, encoder_hidden = return_dict.loss, return_dict.logits, return_dict.encoder_last_hidden_state
        # encoder_outputs = (encoder_hidden,)
        lang_train_losses.append(lang_loss.item())
        lang_loss.backward()
        optimizer.step()
        num_updates += 1
        if j%100 == 0:
            print(f"epoch {i}, batch {j}, loss: {lang_loss.item()}", flush=True)
#    avg_val_loss, new_best_loss = eval_checkpoint(
#        args, i, model, dev_dataloader, save_path=save_path, best_val_loss=best_val_loss,
#    )
    if new_best_loss:
        best_val_loss = avg_val_loss
        best_loss_epoch = i