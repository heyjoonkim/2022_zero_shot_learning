# Prompt Learning

## How to Install

1. <code>conda env create -f environment.yml</code> to create conda env.

2. <code>pip install -e .</code> to install customized transformers library.

3. <code>pip install -r requirements.txt</code> to install additional libraries.


## Zero / Few -shot

```bash
# for Huggingface Transformers
sh scripts/transformers_few_shot_trec.sh

# for OpenAI API
sh openai_few_shot_trec.sh
```

## Generation
The scripts directly call <code>xxx_generated_few_shot_trec.sh</code> for Few-shot learning with generated in-context samples.

```bash
# for Huggingface Transformers
# directly calls scripts/transformers_generated_few_shot_trec.sh
sh scripts/transformers_generate_trec.sh

# for OpenAI API
# directly calls scripts/openai_generated_few_shot_trec.sh
sh scripts/openai_generate.sh

```
