
## use the ai model locally on your device 
### Import libraries
```python
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
```
We import TensorFlow and the Transformer libraries.
Load model and tokenizer
```python
tokenizer = AutoTokenizer.from_pretrained('mental_tokeniser')
model = TFAutoModelForSeq2SeqLM.from_pretrained('t5_mental_assisstant_v5')
```
We load the pretrained tokenizer and T5 model from disk.
Chat loop
```python

while True:

  inputs = input('>>>>')
  
  inputs = tokenizer([inputs], max_length=128, padding=True, truncation=True, return_tensors='tf')
  ```
We create a loop to repeatedly get user input and tokenize it.
The tokenizer prepares the input for the model, padding and truncating as needed.
Generate response
```python

  output = model.generate(inputs.input_ids, max_length=128)
  
  response = tokenizer.decode(output[0], skip_special_tokens=True)
  ```
We pass the tokenized input to the model to generate a sequence.
The tokenizer decodes the output tokens into a readable string.
Print response
```pytho

  print('doctor >', response)
````
