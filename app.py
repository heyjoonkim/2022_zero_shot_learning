import os
import time

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from flask import request, Flask, jsonify
from flask_cors import CORS, cross_origin


model_name = os.environ['MODEL_NAME']

device = torch.device("cuda")
# Main model
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')
print('Start loading model...')
start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
print(f'Done loading model. Total time : {time.time() - start_time}')
model.eval()
model.config.pad_token_id = model.config.eos_token_id
print('Done initialization.')

app = Flask(__name__)
CORS(app)

@app.route('/')
def healthcheck():
    """ 
        healthcheck!!
    """
    return jsonify({'healthcheck':'success', 'model_name': model_name}), 200


@app.route('/inference', methods=['POST'])
def inference():
    """
    Process dataset with prompt
    """
    input_sentence = request.form.get('input_sentence')
    print(f'Input Sentence : {input_sentence}')

    inputs = tokenizer(input_sentence, truncation=True, max_length=1024, return_tensors="pt").to(device)
    
    outputs = model(**inputs)
    # outputs = None

    # shape : (1, length, vocab_size)
    logits = outputs.logits

    probs = torch.softmax(logits, dim=2)
    # shape : (1, length, vocab_size)
    logprobs = torch.log(probs)
    logprobs = logprobs.tolist()

    
    results = {
        "logprobs" : logprobs
    }

    return jsonify(results), 200


if __name__ == '__main__':
    print('Start running server....')
    app.run(host='127.0.0.1', port='6009')
