# python3 train_textworld2.py --data=../../tw_data --gamefile ../../tw_data

# python3 train_textworld2.py --data=../../tw_data/simple_traces/ --gamefile ../../tw_data/simple_games/

# ~/python-py37-mhahn train_textworld2.py --data=/juice/scr/mhahn/CODE/FORM-MEANING/DATA/tw_data/simple_traces --gamefile /juice/scr/mhahn/CODE/FORM-MEANING/DATA/tw_data/simple_games --device=cuda

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

from localizer.tw_localizer import TWLocalizer
from metrics.tw_metrics import get_em, get_confusion_matrix
from data.textworld.parse_tw import (
    translate_inv_items_to_str, translate_inv_str_to_items,
)
from data.textworld.utils import (
    EntitySet, control_mention_to_tgt_simple, control_mention_to_tgt_with_rooms_simple,
    load_possible_pairs, load_negative_tgts,
)
from data.textworld.tw_dataloader import (
    TWDataset, TWEntitySetDataset, TWFullDataLoader, TWEntitySetDataLoader,
)

from probe_models import (
    ProbeLinearModel, ProbeConditionalGenerationModel, ProbeLanguageEncoder, encode_target_states,
    get_probe_model, get_state_encoder, get_lang_model,
)
from itertools import chain, combinations


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
parser.add_argument('--lm_save_path', type=str, default="SAVE/")
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--metric', type=str, choices=['em', 'loss'], help='which metric to use on dev set', default='em')
parser.add_argument('--probe_save_path', type=str, default=None)
parser.add_argument('--probe_layer', type=int, default=-1, help="which layer of the model to probe")
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--no_pretrain', action='store_true', default=False)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--probe_type', type=str, choices=['3linear_classify', 'linear_classify', 'linear_retrieve', 'decoder'], default='decoder')
parser.add_argument('--encode_tgt_state', type=str, default=False, choices=[False, 'NL.bart', 'NL.t5'], help="how to encode the state before probing")
parser.add_argument('--train_data_size', type=int, default=4000)
parser.add_argument('--tgt_agg_method', type=str, choices=['sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'], default='avg', help="how to aggregate across tokens of target, if `encode_tgt_state` is set True")
parser.add_argument('--probe_agg_method', type=str, choices=[None, 'sum', 'avg', 'first', 'last', 'lin_attn', 'ffn_attn', 'self_attn'], default=None, help="how to aggregate across tokens")
parser.add_argument('--probe_attn_dim', type=int, default=None, help="what dimensions to compress sequence tokens to")
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--control_input', default=False, action='store_true', help='control inputs to tokenized entity pair')
parser.add_argument('--local_files_only', action='store_true', default=False, help="use pretrained checkpoints saved in local directories")
parser.add_argument('--probe_target', type=str, default='final.belief_facts', choices=list(chain(*[[
    f'{init_final}.full_facts', f'{init_final}.full_belief_facts', f'{init_final}.belief_facts',
    f'{init_final}.belief_facts_single', f'{init_final}.belief_facts_pair',
    f'{init_final}.full_belief_facts_single', f'{init_final}.full_belief_facts_pair',
    f'{init_final}.belief_facts_single.control', f'{init_final}.full_belief_facts_single.control', f'{init_final}.belief_facts_single.control_with_rooms', f'{init_final}.full_belief_facts_single.control_with_rooms',
    f'{init_final}.belief_facts_pair.control', f'{init_final}.full_belief_facts_pair.control', f'{init_final}.belief_facts_pair.control_with_rooms', f'{init_final}.full_belief_facts_pair.control_with_rooms',
] for init_final in ['init', 'final']])))
parser.add_argument('--localizer_type', type=str, default='all',
    choices=['all'] + [f'belief_facts_{sides}_{agg}' for sides in ['single', 'pair'] for agg in ['all', 'first', 'last']],
    help="which encoded tokens of the input to probe."
    "Set to `all`, `belief_facts_{single|pair}_{all|first|last}`")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--ents_to_states_file', type=str, default=None, help='Filepath to precomputed state vectors')
args = parser.parse_args()

arch = args.arch
pretrain = not args.no_pretrain
batchsize = args.batchsize
control_input = args.control_input
eval_batchsize = args.eval_batchsize
max_seq_len = args.max_seq_len
lm_save_path = args.lm_save_path
localizer_type = args.localizer_type
probe_target = args.probe_target.split('.')
probe_type = args.probe_type
retrieve = probe_type.endswith('retrieve')
classify = probe_type.endswith('classify')
assert not (retrieve and classify)
encode_tgt_state = args.encode_tgt_state
tgt_agg_method = args.tgt_agg_method
probe_agg_method = args.probe_agg_method
probe_attn_dim = args.probe_attn_dim
probe_save_path = args.probe_save_path
train_data_size = args.train_data_size
game_kb = None

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
        self.itos = ["OOV", "<EOS>", "<BOS>", "<PAD>", "<SEP>"]
        self.stoi = {x : i for i, x in enumerate(self.itos)}
        self._pad_token = self.stoi["<PAD>"]
        self.model_max_length = 200
        self._separator = self.stoi["<SEP>"]
        pass

    def tokenize(self, text, **kwargs):
        split = text.replace(".", "").replace(",", "").lower().split(" ")
        return split
    def convert_tokens_to_ids(self, split):
        result = []
        for x in split:
            x = x.lower()
            if x not in self.stoi and len(self.itos) < 10000:
                self.stoi[x] = len(self.stoi)
                self.itos.append(x)
            result.append(self.stoi.get(x, 0))
        return torch.LongTensor(result[:self.model_max_length])
    def towords_batch(self, tensor):
        return [self.towords(x) for x in tensor]
    def towords(self, tensor):
        tensor = tensor.long().cpu().numpy().tolist()
        return " ".join([self.itos[x] for x in tensor])
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

def to_device(x):
 if args.device == "cuda":
  return x.cuda()
 return x

transformer = to_device(torch.nn.TransformerEncoder(encoder_layer=torch.nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=4))
embedding = to_device(torch.nn.Embedding(num_embeddings=10000, embedding_dim=64))
output = to_device(torch.nn.Linear(64, 1, bias=False))

def parameters():
    for x in [transformer, embedding, output]:
        for y in x.parameters():
            yield y


#model.to(args.device)

# load optimizer
all_parameters = [p for p in parameters() if p.requires_grad]
#print(all_parameters)
#quit()
optimizer = torch.optim.Adam(all_parameters, lr=args.lr)


print("Loaded model")

print(f"Saving probe checkpoints to {probe_save_path}")
output_json_fn = None
if args.eval_only:
    output_json_dir = os.path.split(probe_save_path)[0]
    if not os.path.exists(output_json_dir): os.makedirs(output_json_dir)
    output_json_fn = os.path.join(output_json_dir, f"{os.path.split(probe_save_path)[-1].replace('.p', '.jsonl')}")
    print(f"Saving predictions to {output_json_fn}")

DEBUG = True
if DEBUG:
    max_data_size = [2,2]
else:
    max_data_size = [500,train_data_size]

state_key = probe_target[1].replace('_single', '').replace('_pair', '')
tgt_state_key = probe_target[0]+'_states'
possible_pairs = None
ent_set_size = 2
if probe_target[1].endswith('_pair'):
    ent_set_size = 2
    possible_pairs = load_possible_pairs(pair_out_file=os.path.join(args.data, 'entity_pairs.json'))
    assert possible_pairs is not None
if probe_target[1].endswith('_single'): ent_set_size = 1
neg_facts_fn = args.ents_to_states_file

precomputed_negs = None
control = probe_target[2] if len(probe_target)>2 else False
# load data
dev_dataset = TWEntitySetDataset(
    args.data, tokenizer, 'dev', max_seq_len=max_seq_len, ent_set_size=ent_set_size, control=control,
    gamefile=args.gamefile, state_key=state_key, tgt_state_key=tgt_state_key, max_data_size=max_data_size[0],
    inform7_game=inform7_game, possible_pairs=possible_pairs, precomputed_negs=precomputed_negs,
)
dataset = TWEntitySetDataset(
    args.data, tokenizer, 'train', max_seq_len=max_seq_len, ent_set_size=ent_set_size, control=control,
    gamefile=args.gamefile, state_key=state_key, tgt_state_key=tgt_state_key, max_data_size=max_data_size[1],
    inform7_game=inform7_game, possible_pairs=possible_pairs, precomputed_negs=precomputed_negs,
)
print(f"Loaded data: {len(dataset)} train examples, {len(dev_dataset)} dev examples")
train_dataloader = TWEntitySetDataLoader(dataset, tokenizer, batchsize, control_input, device=args.device)
dev_dataloader = TWEntitySetDataLoader(dev_dataset, tokenizer, eval_batchsize, control_input, device=args.device)
print("Created batches")

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
per_epoch = []
loss_running_average = 5
for i in range(args.epochs):
    per_epoch.append(0)

    batch_input = []
    batch_labels = []
    input_to_responses = {}
    for j, (inputs, lang_tgts, init_state, tgt_state, game_ids, entities) in enumerate(train_dataloader):
#        print("inputs", inputs)
#        print("lang_tgts", lang_tgts)
#        print("init_state", init_state)
#        print("tgt_state", tgt_state)
#        print("game_ids", game_ids)
#        print("entities", entities)
#        print("##############################")
#        print("##############################")
#        print("##############################")
#        print("##############################")
#        print("============================lang_tgts")
#        print("\n".join(tokenizer.towords_batch(lang_tgts["input_ids"])))
#        print("============================inputs")
#        print("\n".join(tokenizer.towords_batch(inputs["input_ids"])))
#        print("============================tgt_state")
        labels = []
        inputs_and_qs = []
        for q in range(tgt_state["labels"].size()[0]):
            for r in range(tgt_state["labels"].size()[1]):
              if int(tgt_state["labels"][q,r]) > 0:
         #       print(q, r, tokenizer.towords(tgt_state["all_states_input_ids"][q,r]), ["?", "T", "F"][int(tgt_state["labels"][q,r])], tokenizer.towords(inputs["input_ids"][q]))
                #print(inputs["input_ids"][q].size(), tgt_state["all_states_input_ids"][q,r].size())
                input_here = torch.cat([inputs["input_ids"][q], to_device(torch.LongTensor([tokenizer._separator])), tgt_state["all_states_input_ids"][q,r]], dim=0)
                batch_input.append(input_here)

                batch_labels.append(int(tgt_state["labels"][q,r]))
                X = tuple(tgt_state["all_states_input_ids"][q,r].cpu().numpy().tolist())
                Y = tuple(inputs["input_ids"][q].cpu().numpy().tolist())
                Z = int(tgt_state["labels"][q,r])
                if (X,Y) in input_to_responses:
                    assert input_to_responses[(X,Y)] == Z
                input_to_responses[(X,Y)] = Z
                if len(batch_labels) % 3 == 0:
                  inputs_and_qs = batch_input
                  labels = batch_labels

                  batch_input = []
                  batch_labels = []
                  max_length = max([x.size()[0] for x in inputs_and_qs])
                  for q in range(len(labels)):
                      inputs_and_qs[q] = torch.cat([inputs_and_qs[q], to_device((tokenizer._pad_token + torch.zeros(max_length - inputs_and_qs[q].size()[0]))).long()], dim=0)
                  inputs_and_qs = torch.stack(inputs_and_qs, dim=0).long()
                  labels = to_device(torch.LongTensor(labels))
          
                  embedded = embedding(inputs_and_qs)
                  transformed = transformer(embedded)
          
                  prediction = torch.sigmoid(output(transformed.mean(dim=1))).view(-1)
          
          #        z = 0
          #        for q in range(tgt_state["labels"].size()[0]):
          #            for r in range(tgt_state["labels"].size()[1]):
          #              if int(tgt_state["labels"][q,r]) > 0:
          #                print(float(prediction[z]), q, r, tokenizer.towords(tgt_state["all_states_input_ids"][q,r]), ["?", "T", "F"][int(tgt_state["labels"][q,r])], tokenizer.towords(inputs["input_ids"][q]))
          #                z += 1
          #
          #
                  loss = -torch.where(labels == 1, prediction.log(), (1-prediction).log()).sum()/10
           #       print(prediction)
          #        print(labels)
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  num_updates += 1
                  loss_running_average = 0.95 * loss_running_average + (1-0.95) * float(loss)
                  if j%10 == 0:
                      print(f"epoch {i}, batch {j}, loss: {loss.item()}", [round(x,3) for x in per_epoch[-10:-1]], loss_running_average, flush=True)
                  per_epoch[-1] += float(loss)/100
#    avg_val_loss, new_best_loss = eval_checkpoint(
#        args, i, model, dev_dataloader, save_path=save_path, best_val_loss=best_val_loss,
#    )
#    if new_best_loss:
#        best_val_loss = avg_val_loss
#        best_loss_epoch = i
