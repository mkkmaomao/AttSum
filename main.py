
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import ipdb
import math
import os
from transformers import RobertaConfig, RobertaModel, AdamW, get_linear_schedule_with_warmup

from data_tools.hugf_tokenize import get_dataloaders, tokenizer, decode_batch_ids
from models.pretrain_s2s_copy import Seq2Seq
from data_tools.scorer import rouge_from_maps
from mconfig import args, logger, gpu_count, device

torch.cuda.empty_cache()
print(torch.cuda.is_available())
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

'''Fixed random number seed'''
SEED = args.random_seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# Fixed accuracy of convolution algorithm to ensure reproducibility
torch.backends.cudnn.deterministic = True

'''Load training and test data'''
vocab_size = tokenizer.vocab_size
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args)
logger.info(f'dataloaders have been built')

'''Construct and initialize model objects'''
encoder_config = RobertaConfig.from_pretrained(args.model_name) # roBerta-base
encoder = RobertaModel.from_pretrained(args.model_name, config=encoder_config) # encoder -> roBerta RobertaModel
decoder_layer = nn.TransformerDecoderLayer(d_model=encoder_config.hidden_size, nhead=encoder_config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.dec_layers) # decoder -> 8-layer transformer
model = Seq2Seq(encoder, decoder, encoder_config, args.beam_size, args.max_tgt_len,
                tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id).to(device)
#  bos, eos, and pad: A special token representing the beginning, end, and out-of-vocabulary of a sentence.
# logger.info(model)  # print model's structure


if torch.cuda.is_available() and gpu_count > 1:
    model = torch.nn.DataParallel(model)  # multi GPU

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epoch  
scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=int(t_total*0.1),
                                        num_training_steps=t_total)


def get_one_example(source_ids, target_ids, pred_dist):
    '''Used to debug the output generated'''
    source = decode_batch_ids(source_ids.transpose(0, 1))
    pred = decode_batch_ids(pred_dist.transpose(0, 1))
    target = decode_batch_ids(target_ids[1:].transpose(0, 1))
    return source, pred, target

def calc_batch_score(decoder_output, target_ids):
    '''AVG Rouge score is calculated based on DECODER output and Target IDS'''
    # decoder_output [trg_len, batch_size]
    # target_ids [trg_len, batch_size]
    # print("(main) decoder_output: ",decoder_output)
    # print("(main) target_ids: ", target_ids)
    pred_token_ids = decoder_output.transpose(0, 1) # [batch_size, trg_len]
    # pred_tokens [batch_size, tokens]
    # print("(main) pred_token_ids: ", pred_token_ids)
    pred_tokens = decode_batch_ids(pred_token_ids)
    target_tokens = decode_batch_ids(target_ids[1:].transpose(0, 1)) # jump bos token
    avg_f = rouge_from_maps(target_tokens, pred_tokens)
    return avg_f

def epoch_time(start_time, end_time):
    '''Time difference unit conversion'''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, best_rouge):
    logger.info('-------start train-------')
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for i, batch in enumerate(iterator):
        batch = tuple(t.to(device) for t in batch)
        # source_ids [seq_len, batch_size]
        source_ids, source_mask, target_ids, target_mask = batch
        # print('source id:',source_ids) # body
        # print('source mask:', source_mask)
        # print('target id:', target_ids) # title
        # print('target mask:', target_mask)
        source_ids_T = source_ids.transpose(0, 1) # [batch_size, src_len] for multi gpu
        target_input_T = target_ids.transpose(0, 1)
        source_mask_T = source_mask.transpose(0, 1)
        target_mask_T = target_mask.transpose(0, 1)
        loss, logits = model(source_ids_T, source_mask_T, target_input_T, target_mask_T) # [batch_size, seq_len]
        logits = logits.transpose(0, 1) # [seq_len, batch_size]
        # print("logits:",logits)
        # print("target_ids:",target_ids)
        if (i+1) % args.score_interval == 0: # 500
            score = calc_batch_score(logits, target_ids) #(decode_output, target_id)
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge1_f}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge2_f}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rougeL_f}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge1_p}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge2_p}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rougeL_p}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge1_r}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge2_r}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rougeL_r}')
            logger.info(f'Train Batch {i} | avg_Rouge: {score}')

        if args.debug:
            "DEBUG!!!"
            (source, predict, target) = get_one_example(source_ids, target_ids, logits)
            ipdb.set_trace()
            evaluate(model, val_dataloader, best_rouge)

        if gpu_count > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        epoch_loss += loss.item()*args.gradient_accumulation_steps
        loss.backward()

        if (i+1) % args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if (i+1) % args.log_interval == 0: # 200
            logger.info(f'Train Batch {i} | Time: {epoch_mins}m {epoch_secs}s')
            logger.info(f'\tTrain Loss: {epoch_loss/(i+1):.3f} | Train PPL: {math.exp(epoch_loss/(i+1)):7.3f}')

        if (i+1) % args.eval_interval == 0: # 5000
            rouge_score = evaluate(model, val_dataloader, best_rouge)
            best_rouge = rouge_score if rouge_score > best_rouge else best_rouge
            model.train()

    return best_rouge

def evaluate(model, iterator, best_rouge):
    logger.info('------start evaluation------')
    model.eval()
    rouge_score = 0
    # rouge1_f_score = 0
    # rouge2_f_score = 0
    # rougeL_f_score = 0
    # rouge1_p_score = 0
    # rouge2_p_score = 0
    # rougeL_p_score = 0
    # rouge1_r_score = 0
    # rouge2_r_score = 0
    # rougeL_r_score = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            batch = tuple(t.to(device) for t in batch)
            # source_ids [seq_len, batch_size]
            source_ids, source_mask, target_ids, _ = batch
            # logits [batch_size, tgt_len, out_dim]
            source_ids_T = source_ids.transpose(0, 1) # [batch_size, src_len] for multi gpu
            source_mask_T = source_mask.transpose(0, 1)
            _, logits = model(source_ids_T, source_mask_T) # [batch_size, seq_len]
            logits = logits.transpose(0, 1) # [seq_len-1, batch_size]
            score = calc_batch_score(logits, target_ids)  # (decode_output, target_id)
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge1_f}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge2_f}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rougeL_f}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge1_p}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge2_p}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rougeL_p}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge1_r}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rouge2_r}')
            # logger.info(f'Train Batch {i} | avg_Rouge: {rougeL_r}')
            logger.info(f'Train Batch {i} | avg_Rouge: {score}')
            rouge_score += score
            # rouge1_f_score += rouge1_f
            # rouge2_f_score += rouge2_f
            # rougeL_f_score += rougeL_f
            # rouge1_p_score += rouge1_p
            # rouge2_p_score += rouge2_p
            # rougeL_p_score += rougeL_p
            # rouge1_r_score += rouge1_r
            # rouge2_r_score += rouge2_r
            # rougeL_r_score += rougeL_r
            if args.debug:
                (source, predict, target) = get_one_example(source_ids, target_ids, logits)
                ipdb.set_trace()

    rouge_score = rouge_score / len(iterator)
    # rouge1_f_score = rouge1_f_score / len(iterator)
    # rouge2_f_score = rouge2_f_score / len(iterator)
    # rougeL_f_score = rougeL_f_score / len(iterator)
    # rouge1_p_score = rouge1_p_score / len(iterator)
    # rouge2_p_score = rouge2_p_score / len(iterator)
    # rougeL_p_score = rougeL_p_score / len(iterator)
    # rouge1_r_score = rouge1_r_score / len(iterator)
    # rouge2_r_score = rouge2_r_score / len(iterator)
    # rougeL_r_score = rougeL_r_score / len(iterator)
    logger.info(f'\tEval Rouge: {rouge_score}')
    # logger.info(f'\tEval Rouge1_f: {rouge1_f_score}')
    # logger.info(f'\tEval Rouge2_f: {rouge2_f_score}')
    # logger.info(f'\tEval RougeL_f: {rougeL_f_score}')
    # logger.info(f'\tEval Rouge1_p: {rouge1_p_score}')
    # logger.info(f'\tEval Rouge2_p: {rouge2_p_score}')
    # logger.info(f'\tEval RougeL_p: {rougeL_p_score}')
    # logger.info(f'\tEval Rouge1_r: {rouge1_r_score}')
    # logger.info(f'\tEval Rouge2_r: {rouge2_r_score}')
    # logger.info(f'\tEval RougeL_r: {rougeL_r_score}')

    if rouge_score > best_rouge:
        torch.save(model.state_dict(), args.best_rouge_model_save_path) # java_both.brouge.pt
        logger.info(f'\tSave Best Rouge on Val')
    return rouge_score

def generate(model, iterator):
    logger.info('------start generation------')
    logger.info(f'-----total {len(iterator)} batches-----')
    model.eval()
    rouge_score = 0
    # rouge1_f_score = 0
    # rouge2_f_score = 0
    # rougeL_f_score = 0
    # rouge1_p_score = 0
    # rouge2_p_score = 0
    # rougeL_p_score = 0
    # rouge1_r_score = 0
    # rouge2_r_score = 0
    # rougeL_r_score = 0
    golden_titles = []
    pred_titles = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            batch = tuple(t.to(device) for t in batch)
            # source_ids [seq_len, batch_size]
            source_ids, source_mask, target_ids, _ = batch
            # logits [batch_size, tgt_len, out_dim]
            source_ids_T = source_ids.transpose(0, 1) # [batch_size, src_len] for multi gpu
            source_mask_T = source_mask.transpose(0, 1)
            _, logits = model(source_ids_T, source_mask_T) # [batch_size, seq_len]
            logits = logits.transpose(0, 1) # [seq_len-1, batch_size]
            score = calc_batch_score(logits, target_ids)
            rouge_score += score
            # rouge1_f_score += rouge1_f
            # rouge2_f_score += rouge2_f
            # rougeL_f_score += rougeL_f
            # rouge1_p_score += rouge1_p
            # rouge2_p_score += rouge2_p
            # rougeL_p_score += rougeL_p
            # rouge1_r_score += rouge1_r
            # rouge2_r_score += rouge2_r
            # rougeL_r_score += rougeL_r

            logger.info(f'\tBatch {i+1} Eval Rouge: {rouge_score/(i+1)}')
            # logger.info(f'\tBatch {i + 1} Eval Rouge1_f: {rouge1_f_score / (i + 1)}')
            # logger.info(f'\tBatch {i + 1} Eval Rouge2_f: {rouge2_f_score / (i + 1)}')
            # logger.info(f'\tBatch {i + 1} Eval RougeL_f: {rougeL_f_score / (i + 1)}')
            # logger.info(f'\tBatch {i + 1} Eval Rouge1_p: {rouge1_p_score / (i + 1)}')
            # logger.info(f'\tBatch {i + 1} Eval Rouge2_p: {rouge2_p_score / (i + 1)}')
            # logger.info(f'\tBatch {i + 1} Eval RougeL_p: {rougeL_p_score / (i + 1)}')
            # logger.info(f'\tBatch {i + 1} Eval Rouge1_r: {rouge1_r_score / (i + 1)}')
            # logger.info(f'\tBatch {i + 1} Eval Rouge2_r: {rouge2_r_score / (i + 1)}')
            # logger.info(f'\tBatch {i + 1} Eval RougeL_r: {rougeL_r_score / (i + 1)}')

            # decoder_output [trg_len, batch_size]
            # target_ids [trg_len, batch_size]
            # pred_tokens [batch_size, tokens]
            pred_tokens = decode_batch_ids(logits.transpose(0, 1))
            target_tokens = decode_batch_ids(target_ids[1:].transpose(0, 1)) # jump bos token
            golden_titles += target_tokens
            pred_titles += pred_tokens

    def _write_gen(gold, pred):
        with open(args.gen_gold_path, 'w', encoding='utf-8') as g_f:
            for line_tokens in gold:
                g_f.write(f'{" ".join(line_tokens)}\n')
        with open(args.gen_pred_path, 'w', encoding='utf-8') as p_f:
            for line_tokens in pred:
                p_f.write(f'{" ".join(line_tokens)}\n')

    rouge_score = rouge_score / len(iterator) # rouge_score
    # rouge1_f_score = rouge1_f_score / len(iterator)
    # rouge2_f_score = rouge2_f_score / len(iterator)
    # rougeL_f_score = rougeL_f_score / len(iterator)
    # rouge1_p_score = rouge1_p_score / len(iterator)
    # rouge2_p_score = rouge2_p_score / len(iterator)
    # rougeL_p_score = rougeL_p_score / len(iterator)
    # rouge1_r_score = rouge1_r_score / len(iterator)
    # rouge2_r_score = rouge2_r_score / len(iterator)
    # rougeL_r_score = rougeL_r_score / len(iterator)
    logger.info(f'\tEval Rouge: {rouge_score}')
    # logger.info(f'\tEval Rouge1_f: {rouge1_f_score}')
    # logger.info(f'\tEval Rouge2_f: {rouge2_f_score}')
    # logger.info(f'\tEval RougeL_f: {rougeL_f_score}')
    # logger.info(f'\tEval Rouge1_p: {rouge1_p_score}')
    # logger.info(f'\tEval Rouge2_p: {rouge2_p_score}')
    # logger.info(f'\tEval RougeL_p: {rougeL_p_score}')
    # logger.info(f'\tEval Rouge1_r: {rouge1_r_score}')
    # logger.info(f'\tEval Rouge2_r: {rouge2_r_score}')
    # logger.info(f'\tEval RougeL_r: {rougeL_r_score}')
    _write_gen(golden_titles, pred_titles)
    logger.info(f'Write to file finished')
    return rouge_score

best_rouge = 0
# args.load_model_from_epoch = 8
if args.load_model_from_epoch or args.generate:
    logger.info(f'----load model {args.load_model_path}---')
    model.load_state_dict(torch.load(args.load_model_path))
    best_rouge = args.last_best_rouge
    logger.info(f'------best rouge start from {best_rouge}')

if args.generate:
    logger.info(f'----Start generating----')
    generate(model, test_dataloader) # wirte two files under the folder .val/

if args.train:
    logger.info(f'-------------start training-total {args.epoch} epoches------------')
    logger.info(f'-------every epoch has {len(train_dataloader)} batches-------')
    start_time = time.time()
    for epoch in range(args.load_model_from_epoch, args.epoch):
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        logger.info(f'**********************epoch {epoch}********************* | Time {epoch_mins}m {epoch_secs}s')
        rouge_score = train(model, train_dataloader, optimizer, best_rouge)
        best_rouge = rouge_score if rouge_score > best_rouge else best_rouge
        torch.save(model.state_dict(), args.epoch_model_save_path.replace('.pt', f'.ep{epoch}.pt')) # java_both.ep{xx}.pt


