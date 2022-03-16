#from google.colab import drive
#drive.mount('/content/drive')
#import os

#%cd /content/drive/MyDrive/Colab

#!pip install git+https://github.com/huggingface/transformers
#!pip install transformers datasets -qq
#!ls

from transformers import BertConfig, BertModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
config = BertConfig.from_pretrained('bert-base-cased')
model = BertModel(config)
model.save_pretrained('/content/drive/MyDrive/Colab')


env TASK_NAME=CoLa
python run_glue.py \
  --model_name_or_path /content/drive/MyDrive/Colab \
  --tokenizer_name bert-base-cased \
	--task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-random-weights-CoLa \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=STSB
python run_glue.py \
  --model_name_or_path /content/drive/MyDrive/Colab \
  --tokenizer_name bert-base-cased \  
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-random-weights-STSB \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=RTE
python run_glue.py \
  --model_name_or_path /content/drive/MyDrive/Colab \
  --tokenizer_name bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-random-weights-RTE \
  --logging_steps 5 \ 
  --overwrite_output_dir \


env TASK_NAME=MRPC
python run_glue.py \
  --model_name_or_path /content/drive/MyDrive/Colab \
  --tokenizer_name bert-base-cased \
	--task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-random-weights-MRPC \
  --logging_steps 5 \ 
  --overwrite_output_dir \


env TASK_NAME=SST2
python run_glue.py \
  --model_name_or_path /content/drive/MyDrive/Colab \
  --tokenizer_name bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-random-weights-SST2 \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=QNLI
python run_glue.py \
  --model_name_or_path /content/drive/MyDrive/Colab \
  --tokenizer_name bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-random-weights-QNLI \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=QQP
python run_glue.py \
  --model_name_or_path /content/drive/MyDrive/Colab \
  --tokenizer_name bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-random-weights-QQP \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=MNLI
python run_glue.py \
  --model_name_or_path /content/drive/MyDrive/Colab \
  --tokenizer_name bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-random-weights-MNLI \
  --logging_steps 50 \ 
  --overwrite_output_dir \




