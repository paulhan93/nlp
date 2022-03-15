#from google.colab import drive
#drive.mount('/content/drive')
#import os

#cd /content/drive/MyDrive/Colab\ Data

#pip install git+https://github.com/huggingface/transformers
#pip install transformers datasets -qq
#ls
#python run_glue.py --help  # this will give you a list of possible arguments; very helpful!

env TASK_NAME=CoLa
python run_glue.py \
  --model_name_or_path 'bert-base-cased' \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-CoLa \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=STSB
python run_glue.py \
  --model_name_or_path 'bert-base-cased' \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-STSB \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=RTE
python run_glue.py \
  --model_name_or_path 'bert-base-cased' \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-RTE \
  --logging_steps 5 \ 
  --overwrite_output_dir \


env TASK_NAME=MRPC
python run_glue.py \
  --model_name_or_path 'bert-base-cased' \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-MRPC \
  --logging_steps 5 \ 
  --overwrite_output_dir \


env TASK_NAME=SST2
python run_glue.py \
  --model_name_or_path 'bert-base-cased' \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-SST2 \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=QNLI
python run_glue.py \
  --model_name_or_path 'bert-base-cased' \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-QNLI \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=QQP
python run_glue.py \
  --model_name_or_path 'bert-base-cased' \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-QQP \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=MNLI
python run_glue.py \
  --model_name_or_path 'bert-base-cased' \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-cased-MNLI \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=CoLa
python run_glue.py \
  --model_name_or_path GroNLP/bert-base-dutch-cased \
  --tokenizer_name /content/drive/MyDrive/Colab\ Data/Dutch \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-dutch-cased-CoLa \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=STSB
python run_glue.py \
  --model_name_or_path GroNLP/bert-base-dutch-cased \
  --tokenizer_name /content/drive/MyDrive/Colab\ Data/Dutch \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-dutch-cased-STSB \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=RTE
python run_glue.py \
  --model_name_or_path GroNLP/bert-base-dutch-cased \
  --tokenizer_name /content/drive/MyDrive/Colab\ Data/Dutch \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-dutch-cased-RTE \
  --logging_steps 5 \ 
  --overwrite_output_dir \


env TASK_NAME=MRPC
python run_glue.py \
  --model_name_or_path GroNLP/bert-base-dutch-cased \
  --tokenizer_name /content/drive/MyDrive/Colab\ Data/Dutch \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-dutch-cased-MRPC \
  --logging_steps 5 \ 
  --overwrite_output_dir \


env TASK_NAME=SST2
python run_glue.py \
  --model_name_or_path GroNLP/bert-base-dutch-cased \
  --tokenizer_name /content/drive/MyDrive/Colab\ Data/Dutch \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-dutch-cased-SST2 \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=QNLI
python run_glue.py \
  --model_name_or_path GroNLP/bert-base-dutch-cased \
  --tokenizer_name /content/drive/MyDrive/Colab\ Data/Dutch \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-bas-dutch-cased-QNLI \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=QQP
python run_glue.py \
  --model_name_or_path GroNLP/bert-base-dutch-cased \
  --tokenizer_name /content/drive/MyDrive/Colab\ Data/Dutch \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-dutch-cased-QQP \
  --logging_steps 50 \ 
  --overwrite_output_dir \


env TASK_NAME=MNLI
python run_glue.py \
  --model_name_or_path GroNLP/bert-base-dutch-cased \
  --tokenizer_name /content/drive/MyDrive/Colab\ Data/Dutch \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --output_dir /content/drive/MyDrive/Colab\ Data/bert-base-dutch-cased-MNLI \
  --logging_steps 50 \ 
  --overwrite_output_dir \