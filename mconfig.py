
import argparse
import logging
import torch
import os


parser = argparse.ArgumentParser()

# file paths
DATA_FILE = 'new_data/refined_sub_{}.json'
parser.add_argument("--train_file_path", default=DATA_FILE.format('train'), type=str)
parser.add_argument("--val_file_path", default=DATA_FILE.format('val'), type=str)
parser.add_argument("--test_file_path", default=f'new_data/50_examples_low_quality/refined_50issues.json', type=str)
# parser.add_argument("--test_file_path", default=DATA_FILE.format('test'), type=str)
parser.add_argument("--gen_gold_path", default=f'val/roBERTa_sub_golden_50_bad_examples.txt', type=str) # these two lines are used when the generate function is called
parser.add_argument("--gen_pred_path", default=f'val/roBERTa_sub_test_50_bad_examples.txt', type=str) #f'val/roBERTa_sub_golden_wo_copy.txt'
# dataset related
parser.add_argument("--max_src_len", default=300, type=int)
parser.add_argument("--max_tgt_len", default=15, type=int)
parser.add_argument("--train_batch_size", default=8, type=int) # default is 8
parser.add_argument("--test_batch_size", default=64, type=int)
# model related
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--dec_layers", default=8, type=int)
parser.add_argument("--beam_size", default=10, type=int)
parser.add_argument("--do_lower_case", default=True, type=bool)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--model_name", default="roberta-base", type=str) # default = "microsoft/codebert-base"
# logs and some other frequently changing args
parser.add_argument("--best_rouge_model_save_path", default=f'roBERTa_sub_best_brouge_wo_copy.pt', type=str)
parser.add_argument("--log_file", default=f'logs/log', type=str)
parser.add_argument("--load_model_from_epoch", default=0, type=int)
parser.add_argument("--last_best_rouge", default=0.2, type=float)
parser.add_argument("--epoch_model_save_path", default=f'roBERTa_sub_epoch_wo_copy.pt', type=str)
parser.add_argument("--load_model_path", default=f'roBERTa_sub_more_best_brouge_1110_6am.pt', type=str) # if args.load_model_from_epoch or args.generate
# training related
parser.add_argument("--gpu", default="0,1", type=str)
parser.add_argument("--random_seed", default=999, type=int)
parser.add_argument("--epoch", default=10, type=int) # default = 15
parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
parser.add_argument("--debug", default=0, type=int)
parser.add_argument("--generate", default=1, type=int)
parser.add_argument("--train", default=0, type=int)
parser.add_argument("--log_interval", default=200, type=int) #100
parser.add_argument("--score_interval", default=500, type=int) #200
parser.add_argument("--eval_interval", default=5000, type=int) #1750
args = parser.parse_args()


'''make directories'''
for directory in ['./logs', './state_dicts']:
    if not os.path.exists(directory):
        os.makedirs(directory)

'''logging'''
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s \t %(message)s')
handler = logging.FileHandler(args.log_file, 'a', 'utf-8')
handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
if not args.debug:
    logger.addHandler(handler)
logger.addHandler(console_handler)

'''print all args'''
logger.info(args)

'''for multi gpu'''
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
gpu_count = len(args.gpu.split(','))
logger.info(f'GPU count {gpu_count}, no. {args.gpu}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')