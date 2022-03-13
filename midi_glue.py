# This file is a modification of MIDI-BERT/MidiBERT/CP/finetunetrainer.py
# from https://github.com/wazenmai/MIDI-BERT.git
# which is used to finetune MidiBERT-Piano on symbolic music understanding tasks 
# like composer classification, mood prediction, etc.
# We have modified it to finetune on GLUE tasks

import argparse
import numpy as np
import pickle
import os
import random

from torch.utils.data import DataLoader
import torch 
from transformers import BertConfig

from datasets import load_dataset
import pandas as pd
from transformers import BertTokenizer

from model import MidiBert
from finetune_trainer import FinetuneTrainer
from finetune_dataset import FinetuneDataset
from finetune_model import TokenClassification, SequenceClassification

from matplotlib import pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### mode ###
    parser.add_argument('--task', type=str, required=True)
    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='../../dict/CP.pkl')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--ckpt', default='result/pretrain/test/model_best.ckpt')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true")  # default: false
    parser.add_argument('--resume', action="store_true")  # default: false
    
    ### cuda ###
    parser.add_argument("--cpu", action="store_true") # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1,2,3], help="CUDA device ids")

    args = parser.parse_args()

    return args


def load_data(task):
    #!pip install datasets==1.11.0

    args = get_args()
    max_seq_length = args.max_seq_len

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    print (actual_task)

    task_to_keys = {
        "mnli": ["premise", "hypothesis"],
        "mrpc": ["sentence1", "sentence2"],
        "qnli": ["question", "sentence"],
        "qqp": ["question1", "question2"],
        "rte": ["sentence1", "sentence2"],
        "stsb": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
    }

    # Load the training dataset 
    df = pd.DataFrame(dataset["train"][:])

    labels = df.label.values
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    input_ids = []

    if actual_task=="cola" or actual_task=="sst2":
        sentences = df.sentence.values
        for sent in sentences:
              encoded_dict = tokenizer.encode(
                                  sent,                    
                                  max_length = max_seq_length ,           # Pad 
                                  add_special_tokens = False,
                                  truncation=True,
                                  pad_to_max_length = True,
                                  return_tensors = 'pt',     # Return pytorch tensors
                              )
              input_ids.append(encoded_dict)
    else:
        sentences = df[task_to_keys[actual_task]].values.astype("str")
        for sent in sentences:
              encoded_dict = tokenizer.encode(
                                  sent[0], sent[1],      
                                  max_length = max_seq_length ,           # Pad 
                                  add_special_tokens = False,
                                  truncation=True,
                                  pad_to_max_length = True,
                                  return_tensors = 'pt',     # Return pytorch tensors
                              )
              input_ids.append(encoded_dict)
          
          
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.tensor(labels)

    x = input_ids.numpy()
    y_train = labels.numpy()
    X_train = np.empty((x.shape[0], max_seq_length, 4))

    X_train[:,:,0] = 1                                          # map to 1 in the Bar column
    X_train[:,:,1] = x%16                                       # map to 0-16 in the Position column
    X_train[:,:,2] = ((x/16).astype(int))%86                    # map to 0-86 in the Pitch column
    X_train[:,:,3] = ((x/(16*86)).astype(int))%32               # map to 0-32 in the Duration column

    # the pad token is initially mapped to [1 0 0 0]
    # but we want it mapped to [ 2 16 86 64]
    for i in range (0,x.shape[0]):
      for j in range (0,max_seq_length):
        X_train[i,j,:] = np.array([2, 16, 86, 64]) if(np.sum(X_train[i,j,:]) == 1) else X_train[i,j,:]
                                  
    X_train = X_train.astype(int)      

    
    # Load validation dataset
    df = pd.DataFrame(dataset["validation"][:])

    labels = df.label.values
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    input_ids = []

    if actual_task=="cola" or actual_task=="sst2":
        sentences = df.sentence.values
        for sent in sentences:
              encoded_dict = tokenizer.encode(
                                  sent,                   
                                  max_length = max_seq_length ,        
                                  add_special_tokens = False,
                                  truncation=True,
                                  pad_to_max_length = True,
                                  return_tensors = 'pt',    
                              )
              input_ids.append(encoded_dict)
    else:
        sentences = df[task_to_keys[actual_task]].values.astype("str")
        for sent in sentences:
              encoded_dict = tokenizer.encode(
                                  sent[0], sent[1],                           
                                  max_length = max_seq_length ,        
                                  add_special_tokens = False,
                                  truncation=True,
                                  pad_to_max_length = True,
                                  return_tensors = 'pt',    
                              )
              input_ids.append(encoded_dict)
          
          
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.tensor(labels)

    x = input_ids.numpy()
    y_val = labels.numpy()
    X_val = np.empty((x.shape[0], max_seq_length, 4))

    X_val[:,:,0] = 1                                          # map to 1 in the Bar column
    X_val[:,:,1] = x%16                                       # map to 0-16 in the Position column
    X_val[:,:,2] = ((x/16).astype(int))%86                    # map to 0-86 in the Pitch column
    X_val[:,:,3] = ((x/(16*86)).astype(int))%32               # map to 0-32 in the Duration column

    # the pad token is initially mapped to [1 0 0 0]
    # but we want it mapped to [ 2 16 86 64]
    for i in range (0,x.shape[0]):
      for j in range (0,max_seq_length):
        X_val[i,j,:] = np.array([2, 16, 86, 64]) if(np.sum(X_val[i,j,:]) == 1) else X_val[i,j,:]
                                  
    X_val = X_val.astype(int)      
    
    print('X_train: {}, X_valid: {}'.format(X_train.shape, X_val.shape))
    print('y_train: {}, y_valid: {}'.format(y_train.shape, y_val.shape))

    return X_train, X_val, y_train, y_val


def main():
    # set seed
    seed = 2021
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu
    np.random.seed(seed)
    random.seed(seed)

    # argument
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset") 

    seq_class = True
    args.class_num = 1 if args.task=="stsb" else 2
    X_train, X_val, y_train, y_val = load_data(args.task)
    
    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val) 

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader",len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader",len(valid_loader))

    ###
    X_test, y_test = X_val, y_val
    test_loader = valid_loader

    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                position_embedding_type='relative_key_query',
                                hidden_size=args.hs)

    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    best_mdl = ''
    if args.resume:
        model = SequenceClassification(midibert, args.class_num, args.hs)
        best_mdl = args.ckpt
        print("   Loading model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    elif not args.nopretrain:
        model = None
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        midibert.load_state_dict(checkpoint['state_dict'])
    else:
        model = None
    
    index_layer = int(args.index_layer)-13
    print("\nCreating Finetune Trainer using index layer", index_layer)
    trainer = FinetuneTrainer(midibert, train_loader, valid_loader, test_loader, index_layer, args.lr, args.class_num, args.task,
                                args.hs, y_test.shape, args.cpu, args.cuda_devices, model, seq_class, args.max_seq_len)
    
    
    print("\nTraining Start")
    save_dir = os.path.join('result/finetune/', args.task)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

#    train_accs, valid_accs = [], []
    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
        valid_loss, valid_acc = trainer.valid()
        print("initial valid loss & acc: ", valid_loss, valid_acc)
        for epoch in range(args.epochs):
            train_loss, train_acc = trainer.train()
            valid_loss, valid_acc = trainer.valid()
            test_loss, test_acc, _ = 0,0,0 #trainer.test()

            is_best = valid_acc >= best_acc
            best_acc = max(valid_acc, best_acc)
            
            if is_best:
                bad_cnt, best_epoch = 0, epoch
            else:
                bad_cnt += 1
            
            print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {} | Test loss: {} | Test acc: {}'.format(
                epoch+1, args.epochs, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))

#            train_accs.append(train_acc)
#            valid_accs.append(valid_acc)

            trainer.save_checkpoint(epoch, train_acc, valid_acc, 
                                    valid_loss, train_loss, is_best, filename)


            outfile.write('Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}\n'.format(
                epoch+1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))
        
            if bad_cnt > 3:
                print('valid acc not improving for 3 epochs')
                break

    # draw figure valid_acc & train_acc
    '''plt.figure()
    plt.plot(train_accs)
    plt.plot(valid_accs)
    plt.title(f'{args.task} task accuracy (w/o pre-training)')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','valid'], loc='upper left')
    plt.savefig(f'acc_{args.task}_scratch.jpg')'''

if __name__ == '__main__':
    main()
