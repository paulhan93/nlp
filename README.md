## Mapping text to MIDI tokens 
* Examples of our process for mapping tokenized text to corresponding MIDI tokens can be found in the notebook file ```tokenization_examples.ipynb```

## To fine-tune MidiBERT-Piano on GLUE tasks

* First clone the MidiBERT-Piano repository and install required packages:
```python
git clone https://github.com/wazenmai/MIDI-BERT.git
cd MIDI-BERT
pip install -r requirements.txt
cd MidiBERT/CP
```
* Then copy the ```midi_glue.py``` and ```finetune_trainer.py``` files from our repository into the directory
* Download the pre-trained model checkpoint from the MidiBERT-Piano repo
* Now you can fine-tune MidiBERT-Piano on GLUE tasks:
```python
python3 midi_glue.py --task='mrpc' --epochs=3 --ckpt='pretrain_model.ckpt' --lr=2e-5
```
* The fine-tuning runs for our experiments are in ```run_finetune_on_midi.sh```

## To fine-tune the Dutch model on GLUE tasks
* We used the run_glue.py script provided by Hugging Face to fine-tune GroNLP/bert-base-dutch-cased on the GLUE tasks.
* The Wordpiece tokenizer that we used for the English vocabulary is saved as a serialized JSON file on our repository as ```tokenizer.json``` and the vocabulary file is ```vocab.txt```
* The fine-tuning runs for our experiments are in ```run_finetune_on_bert-base-cased.sh```, ```run_finetune_on_bert-base-cased_random_weights.sh```, and ```run_finetune_on_bert-base-dutch-cased.sh```
