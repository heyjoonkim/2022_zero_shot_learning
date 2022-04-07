import os

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, LogSoftmax

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM

from flask import render_template, request, Flask, Response, jsonify
from flask_cors import CORS, cross_origin


model_name = os.environ['MODEL_NAME']

device = torch.device("cuda")
# Main model
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.eval()
model.config.pad_token_id = model.config.eos_token_id

app = Flask(__name__)
CORS(app)

@app.route('/')
def healthcheck():
    """ 
        healthcheck!!
    """
    return jsonify({'healthcheck':'success', 'model_name': model_name}), 200


@app.route('/completion', methods=['POST'])
def completion():
    """
    Process dataset with prompt
    """
    params = request.get_json()
    prompt = params['prompt']
    max_tokens = params['max_tokens']
    temperature = params['temperature']
    echo = params['echo']
    presence_penalty = params['presence_penalty']
    frequency_penalty = params['frequency_penalty']
    top_p = params['top_p']
    logprobs = params['logprobs']
    # TODO: implement n!
    n = params['n']

    torch.cuda.empty_cache()

    if temperature == 0:
        temperature = 0.01


    sm = LogSoftmax(dim=1)
    inputs = tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt").to(device)
    input_len = len(inputs['input_ids'][0])
    outputs = model(**inputs)
    logits = outputs.logits[0][:-1].cpu().detach()

    if max_tokens == 0:
        total_logits = logits
        output_ids = inputs['input_ids'][0]
    else:
        sample_outputs = model.generate(
            inputs['input_ids'],
            do_sample=True, 
            temperature=temperature,
            max_new_tokens=max_tokens, 
            num_return_sequences=n,
            return_dict_in_generate=True,
            output_scores=True,
            top_p=0.92, 
            top_k=0,
            repetition_penalty=2.0
        )
        output_ids = sample_outputs.sequences[0].cpu().detach()
        generated_logits = torch.cat(sample_outputs.scores,dim=0).cpu().detach()
        total_logits = torch.cat([logits, generated_logits],dim=0)


    prob = sm(total_logits)

    output_prob = prob[range(prob.shape[0]), output_ids[1:]]
    output_prob = output_prob.cpu().detach().tolist()

    output_ids = output_ids.tolist()

    tokens = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token)) for token in output_ids]
    token_logprobs = [None]
    for p in output_prob:
        token_logprobs.append(p)

    if not echo:
        tokens = tokens[input_len:]
        token_logprobs = token_logprobs[input_len:]
    text = ''.join(tokens)

    result = {'logprobs': {'token_logprobs': token_logprobs, 'tokens':tokens}, 'text': text}
    results = {'choices':[result]}

    return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=True)
