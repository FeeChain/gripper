# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")

# input_text = "最常见的水果"
# input_ids = tokenizer.encode(input_text, return_tensors="pt")

# output_ids = model.generate(input_ids,max_new_tokens = 50)
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# print(output_text)


# # [Running] python3 -u "/Users/eric/EricLu_project/project_c/llama.py"

# # Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
# # Loading checkpoint shards:  50%|█████     | 1/2 [01:19<01:19, 79.66s/it]
# # Loading checkpoint shards: 100%|██████████| 2/2 [01:41<00:00, 45.83s/it]
# # Loading checkpoint shards: 100%|██████████| 2/2 [01:41<00:00, 50.90s/it]

# # Downloading (…)neration_config.json:   0%|          | 0.00/137 [00:00<?, ?B/s]
# # Downloading (…)neration_config.json: 100%|██████████| 137/137 [00:00<00:00, 199kB/s]
# # /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/transformers/generation/utils.py:1346: UserWarning: Using `max_length`'s default (20) to control the generation length. This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
# #   warnings.warn(
# # 2023-07-21 00:38:12.687033: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# # To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# # Hello, how are you? I hope you are well.
# # I am a 20

# # [Done] exited with code=0 in 1123.75 seconds

# # [Running] python3 -u "/Users/eric/EricLu_project/project_c/llama.py"

# # Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
# # Loading checkpoint shards:  50%|█████     | 1/2 [01:12<01:12, 72.78s/it]
# # Loading checkpoint shards: 100%|██████████| 2/2 [01:33<00:00, 41.92s/it]
# # Loading checkpoint shards: 100%|██████████| 2/2 [01:33<00:00, 46.55s/it]
# # /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/transformers/generation/utils.py:1346: UserWarning: Using `max_length`'s default (20) to control the generation length. This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
# #   warnings.warn(
# # 2023-07-21 00:52:06.901670: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# # To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# # 告诉我有没有老虎的世界


# # [Done] exited with code=0 in 547.947 seconds

# # [Running] python3 -u "/Users/eric/EricLu_project/project_c/llama.py"

# # Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
# # Loading checkpoint shards:  50%|█████     | 1/2 [01:17<01:17, 77.89s/it]
# # Loading checkpoint shards: 100%|██████████| 2/2 [01:44<00:00, 47.57s/it]
# # Loading checkpoint shards: 100%|██████████| 2/2 [01:44<00:00, 52.12s/it]
# # 2023-07-21 10:33:42.341810: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# # To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# # 最常见的水果
# # The most common fruits in the world are:
# # 1. Banana:
# # Banana is the most common fruit in the world. It is a tropical fruit that grows in the tropics. It is a very popular fruit in the

# # [Done] exited with code=0 in 4493.593 seconds


from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")

# input_text = "最常见的水果"
# input_ids = tokenizer.encode(input_text, return_tensors="pt")


import guidance
# replace your_path with a version of the LLaMA model
guidance.llm = guidance.llms.transformers.LLaMA(model=model,tokenizer=tokenizer, device="cpu") # use cuda for GPU if you have >27GB of VRAM
# guidance.llm = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")
# define the few shot examples
examples = [
    {'input': 'I wrote about shakespeare',
    'entities': [{'entity': 'I', 'time': 'present'}, {'entity': 'Shakespeare', 'time': '16th century'}],
    'reasoning': 'I can write about Shakespeare because he lived in the past with respect to me.',
    'answer': 'No'},
    {'input': 'Shakespeare wrote about me',
    'entities': [{'entity': 'Shakespeare', 'time': '16th century'}, {'entity': 'I', 'time': 'present'}],
    'reasoning': 'Shakespeare cannot have written about me, because he died before I was born',
    'answer': 'Yes'},
    {'input': 'A Roman emperor patted me on the back',
    'entities': [{'entity': 'Roman emperor', 'time': '1st-5th century'}, {'entity': 'I', 'time': 'present'}],
    'reasoning': 'A Roman emperor cannot have patted me on the back, because he died before I was born',
    'answer': 'Yes'}
]

# define the guidance program
structure_prompt = guidance(
'''How to solve anachronism problems:
Below we demonstrate how to test for an anachronism (i.e. whether it could have happened or not based on the time periods associated with the entities).
----

{{~! display the few-shot examples ~}}
{{~#each examples}}
Sentence: {{this.input}}
Entities and dates:{{#each this.entities}}
{{this.entity}}: {{this.time}}{{/each}}
Reasoning: {{this.reasoning}}
Anachronism: {{this.answer}}
---
{{~/each}}

{{~! place the real question at the end }}
Sentence: {{input}}
Entities and dates:{{#geneach 'entities' stop="\\nReasoning:"}}
{{gen 'this.entity' stop=":"}}: {{gen 'this.time' stop="\\n"}}{{/geneach}}
Reasoning:{{gen "reasoning" stop="\\n"}}
Anachronism: {{#select "answer"}}Yes{{or}}No{{/select}}''')

out = structure_prompt(examples=examples, input='The T-rex bit my dog')
print(out)