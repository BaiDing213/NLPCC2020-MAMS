from __future__ import absolute_import, division, print_function

import argparse
import os
import random
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score

import torch
import torch.nn as nn
import torch.functional as F

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoConfig, AutoTokenizer
from transformers import BertForSequenceClassification_MERGE
from transformers import RobertaForSequenceClassification_MERGE
from transformers import AlbertForSequenceClassification_MERGE
from transformers import XLNetForSequenceClassification_MERGE
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import ATSAProcessor, get_Dataset


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_metrics(true_res, pred_res, label_list):
    acc = accuracy_score(y_true=true_res, y_pred=pred_res)
    # rec = recall_score(y_true=true_res, y_pred=pred_res, average="macro")
    f1 = f1_score(y_true=true_res, y_pred=pred_res, labels=label_list, average="macro")
    return acc, f1


def predict(logits, label_list):
    logits = nn.functional.softmax(logits, 1)
    logits = logits.detach().cpu().numpy()
    if np.argmax(logits, axis=1).any() not in label_list:
        print('Attention, predict result not in label list: ', np.argmax(logits, axis=1))
    return np.argmax(logits, axis=1)


def evaluate(args, model, data, global_step, tr_loss_avg):
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

    print("\n******************** Running Eval ********************")
    print("  Num examples = {:d}".format(len(data)))
    print("  Batch size = {:d}".format(args.eval_batch_size))

    true_res, pred_res = [], []
    logits_res = None # output for esemble
    eval_loss, eval_acc, eval_f1 = 0.0, 0.0, 0.0
    nb_eval_steps, nb_eval_examples = 0, 0
    model.eval()
    for _, batch in enumerate(dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels': batch[3]
        }
        if "roberta" in args.model_type:
            inputs["token_type_ids"] = None

        with torch.no_grad():
            outputs = model(**inputs)
            eval_los = outputs[0]
            predicts = predict(outputs[1], args.label_list)

        nb_eval_steps += 1
        eval_loss += eval_los.mean().item()
        pred_res.extend(predicts.tolist())
        true_res.extend(inputs["labels"].cpu().numpy().tolist())

        if logits_res is None:
            logits_res = outputs[1]
        else:
            logits_res = torch.cat((logits_res, outputs[1]), dim=0)

    eval_loss = eval_loss / nb_eval_steps
    eval_acc, eval_f1 = get_metrics(true_res, pred_res, args.label_list)
    print("  global_step: {:d}, train_loss: {:.4f}, eval_loss: {:.4f}, eval_acc: {:.4f}, eval_f1: {:.4f}".format(global_step, tr_loss_avg, eval_loss, eval_acc, eval_f1))

    # output logits result file in the form of dataframe
    check_point = global_step
    logits_res = logits_res.detach().cpu().numpy()
    label_0 = logits_res[:, 0].tolist()
    label_1 = logits_res[:, 1].tolist()
    label_2 = logits_res[:, 2].tolist()
    logits_df = pd.DataFrame({"label_0": label_0, "label_1": label_1, "label_2": label_2})
    logits_df.to_csv(args.model_type + "_" + str(check_point) + "steps_" + str(round(eval_f1, 5)) + ".csv")

    return eval_acc, eval_f1


def main():
    '''
    Parameters
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default="./data/train.pkl", type=str)
    parser.add_argument("--eval_file", default="./data/dev.pkl", type=str)
    parser.add_argument("--test_file", default="./data/test.pkl", type=str)

    parser.add_argument("--model_type", default="bert-base-uncased", type=str)
    parser.add_argument("--model_name_or_path", default="../language_model/bert-base-uncased/", type=str)
    parser.add_argument("--do_lower_case", default=True, type=boolean_string)
    parser.add_argument("--output_dir", default="./state_models/bert-base-uncased/", type=str)

    # parser.add_argument("--model_type", default="bert-large-uncased", type=str)
    # parser.add_argument("--model_name_or_path", default="../language_model/bert-large-uncased/", type=str)
    # parser.add_argument("--do_lower_case", default=True, type=boolean_string)
    # parser.add_argument("--output_dir", default="./state_models/bert-large-uncased/", type=str)

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument(
        "--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument(
        "--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--do_train", default=False, type=boolean_string)
    parser.add_argument("--do_eval", default=False, type=boolean_string)
    parser.add_argument("--do_test", default=True, type=boolean_string)

    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--max_seq_length", default=64, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--num_train_epochs", default=5, type=float)
    parser.add_argument("--no_cuda", default=False, type=boolean_string, help="Whether not to use CUDA when available")

    parser.add_argument("--save_check_point", default=5200, type=int)
    parser.add_argument("--eval_steps", default=200, type=int)
    parser.add_argument("--skip_eval_rate", default=0.30, type=float)
    parser.add_argument("--logging_steps", default=200, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--warmup_proprotion", default=0.1, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--overwrite_output_dir", default=True, type=boolean_string)
    args = parser.parse_args()

    # Setup CUDA, GPU
    if args.no_cuda:
        device = torch.device("cpu")
        args.n_gpu = 0
    else:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    args.device = device
    print("device: {0}, n_gpu: {1}".format(device, args.n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".
                            format(args.gradient_accumulation_steps))

    # Setup path to save model
    if args.overwrite_output_dir and args.do_train:
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    print("Clean the output path: {}".format(c_path))
                    if os.path.isdir(c_path):
                        del_file(c_path)
                        os.rmdir(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup seed
    set_seed(args)

    print("Training/evaluation parameters %s", args)

    '''
    Data
    '''
    processor = ATSAProcessor()
    label_list = processor.get_labels()
    args.num_labels = len(label_list)
    args.label_list = label_list

    '''
    Train
    '''
    if args.do_train:
        # --------------------  loading model --------------------

        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path, 
            num_labels=args.num_labels, 
            output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,                                         
        do_lower_case=args.do_lower_case
        )
        model = BertForSequenceClassification_MERGE.from_pretrained(
            args.model_name_or_path, 
            config=config
        )
        model.to(device)
        
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # -------------------- loading data --------------------

        train_examples, train_features, train_data = get_Dataset(args, processor, tokenizer, mode="train")
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        if args.do_eval:
            eval_examples, eval_features, eval_data = get_Dataset(args, processor, tokenizer, mode="eval")

        # t_total：模型参数更新次数，模型每个batch更新一次
        # len(train_dataloader)：训练集数据总batch数
        # len(train-dataloader) // args.gradient_accumulation_steps：一个epoch模型参数更新次数
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # -------------------- optimizer & schedule (linear warmup and decay) --------------------

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        
        # -------------------- Train --------------------
        
        print("\n******************** Running Train ********************")
        print("  Num examples = {}".format(len(train_data)))
        print("  Num Epochs = {}".format(args.num_train_epochs))
        print("  Total optimization steps = {}".format(t_total))
        
        global_step = 0
        best_f1 = 0.0
        tr_loss, logging_loss = 0.0, 0.0
        model.train()
        model.zero_grad()
        for ep in range(int(args.num_train_epochs)):
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3]
                }
                if "roberta" in args.model_type:
                    inputs["token_type_ids"] = None
                outputs = model(**inputs)
                loss = outputs[0] # model outputs are always tuple in transformers (see doc)
                
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step() # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:  
                    tr_loss_avg = (tr_loss-logging_loss) / args.logging_steps # 计算一个logging_step的平均loss
                    logging_loss = tr_loss
                    print("  epoch: {:d}, global_step: {:d}, train loss: {:.4f}".format(ep, global_step, tr_loss_avg))
                
                # save model trained by train data & eval data
                if args.save_check_point == global_step:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    
                    # Good practice: save your training arguments together with the trained model
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

                '''
                Eval
                '''
                if args.do_eval and global_step > args.skip_eval_rate*t_total and global_step % args.eval_steps == 0:
                    eval_acc, eval_f1 = evaluate(args, model, eval_data, global_step, tr_loss_avg)
                    
                    # save the best performs model
                    if eval_f1 > best_f1:
                        print("**************** the best f1 is {:.4f} ****************\n".format(eval_f1))
                        best_f1 = eval_f1
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        
                        # Good practice: save your training arguments together with the trained model
                        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
            
        '''
        Eval at the end of the train
        '''
        if args.do_eval and global_step > args.skip_eval_rate*t_total and global_step % args.eval_steps == 0:
            eval_acc, eval_f1 = evaluate(args, model, eval_data, global_step)
            
            # save the best performs model
            if eval_f1 > best_f1:
                print("**************** the best f1 is {:.4f} ****************\n".format(eval_f1))
                best_f1 = eval_f1
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                
                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    '''
    Test
    '''
    if args.do_test:
        args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
        config = AutoConfig.from_pretrained(
            args.output_dir,
            num_labels=args.num_labels , 
            output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, 
            do_lower_case=args.do_lower_case
        )
        model = BertForSequenceClassification_MERGE.from_pretrained(
            args.output_dir,
            config=config)
        model.to(device)
        
        test_examples, test_features, test_data = get_Dataset(args, processor, tokenizer, mode="test")
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        
        print("\n******************** Running test ********************")
        print("  Num examples = {:d}".format(len(test_examples)))
        print("  Batch size = {:d}".format(args.eval_batch_size))
        
        logits_res = None # 输出logits用于投票
        pred_res = []
        model.eval()
        for _, batch in enumerate(test_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
            }
            if "roberta" in args.model_type:
                inputs["token_type_ids"] = None
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs[0]
                
                # collect logits output
                if logits_res is None:
                    logits_res = logits
                else:
                    logits_res = torch.cat((logits_res, logits), dim=0)

                # collect label output
                pred_label = predict(logits, args.label_list) # 测试时 logits 为outputs[0]
                pred_res.extend(pred_label.tolist())
        
        # pred_res = np.array(pred_res)
        # ground_truth = np.array(pd.read_pickle("./data/dev.pkl")["label"].tolist())
        # ans = f1_score(y_true=ground_truth, y_pred=pred_res, labels=[0, 1, 2], average="macro")
        # print(ans)

        logits_res = logits_res.detach().cpu().numpy()
        label_0 = logits_res[:, 0].tolist()
        label_1 = logits_res[:, 1].tolist()
        label_2 = logits_res[:, 2].tolist()
        logits_df = pd.DataFrame({"label_0": label_0, "label_1": label_1, "label_2": label_2})
        logits_df.to_csv(args.model_type + "_logits_test.csv")


if __name__ == "__main__":
    main()
