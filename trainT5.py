import keras
import pandas as pd 
from datasets import load_dataset
from transformers import AutoTokenizer ,TFAutoModelForSeq2SeqLM
import tensorflow as tf

# 1- load dataset
dataset = load_dataset('marmikpandya/mental-health')
data = pd.DataFrame(dataset['train'])
# data = pd.read_csv('banned_questions.tsv',sep='\t').astype('str')
inputs = data['input'].tolist()
targets = data['output'].tolist()


model_name = 't5-small'

# 2- load tokeniser and proccessing data 
tokeniser = AutoTokenizer.from_pretrained(model_name)

encoded_inputs = dict(tokeniser(inputs,max_length=128,padding=True,truncation=True,return_tensors='tf'))
with tokeniser.as_target_tokenizer():
    encoded_targets = tokeniser(targets,max_length=128,padding=True,truncation=True,return_tensors='np')['input_ids']

# 3- create dataset
dataset = tf.data.Dataset.from_tensor_slices((encoded_inputs,encoded_targets))
dataset = dataset.batch(8)
dataset = dataset.shuffle(1024).prefetch(tf.data.AUTOTUNE).cache()

del(data,inputs,targets)

# 4- load model 

t5 = TFAutoModelForSeq2SeqLM.from_pretrained('t5-small')
optimiser = keras.optimizers.legacy.Adam(learning_rate=5e-5)
t5.compile(optimizer = optimiser,metrics = ['accuracy'])
t5.summary()
epochs = 20
t5.fit(x=dataset,epochs=epochs,workers = 8)

t5.save_pretrained('t5_mental_assisstant_v5.1')
