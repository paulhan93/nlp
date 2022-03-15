#!/bin/bash

# load in the pre-trained MidiBERT-Piano model from the checkpoint and fine-tune on GLUE tasks

python3 midi_glue.py --task=‘cola’  --epochs=3 --ckpt='pretrain_model.ckpt'  --batch_size=8 --lr=1e-5

python3 midi_glue.py --task=‘rte’  --epochs=3 --ckpt='pretrain_model.ckpt'  --batch_size=8 --lr=3e-5

python3 midi_glue.py --task=‘mrpc’  --epochs=3 --ckpt='pretrain_model.ckpt'  --batch_size=8 --lr=2e-5

python3 midi_glue.py --task=‘sst2’  --epochs=3 --ckpt='pretrain_model.ckpt'  --batch_size=8 --lr=1e-5

python3 midi_glue.py --task=‘qnli’  --epochs=3 --ckpt='pretrain_model.ckpt'  --batch_size=8 --lr=1e-5


# use the --nopretrain argument to fine-tune a MidiBERT-Piano model with no pre-training

python3 midi_glue.py --task=‘cola’  --epochs=3 --nopretrain  --batch_size=8 --lr=1e-5

python3 midi_glue.py --task=‘rte’  --epochs=3 --nopretrain  --batch_size=8 --lr=3e-5

python3 midi_glue.py --task=‘mrpc’  --epochs=3 --nopretrain  --batch_size=8 --lr=2e-5

python3 midi_glue.py --task=‘sst2’  --epochs=3 --nopretrain  --batch_size=8 --lr=1e-5

python3 midi_glue.py --task=‘qnli’  --epochs=3 --nopretrain  --batch_size=8 --lr=1e-5
