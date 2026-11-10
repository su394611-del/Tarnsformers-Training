import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DPOTrainer, DPOConfig
from datasets import Dataset
import pandas as pd
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
print(f"GPU: {torch.cuda.get_device_name(0)}")
#Stage 1 — SFT instruction dataset in chat format
SFT_DATA = [
{
"messages": [
{"role": "system", "content": "You are a helpful, concise, and accurate AI assistant."},
{"role": "user", "content": "What is the capital of France?"},
{"role": "assistant", "content": "The capital of France is Paris."}
]
},
{
"messages": [
{"role": "system", "content": "You are a helpful, concise, and accurate AI assistant."},
{"role": "user", "content": "Write a Python function that returns the factorial of a number using recursion."},
{"role": "assistant", "content": "def factorial(n):\n if n <= 1:\n return 1\n return n * factorial(n - 1)"}
]
},
{
"messages": [
{"role": "system", "content": "You are a helpful, concise, and accurate AI assistant."},
{"role": "user", "content": "Explain the difference between supervised and unsupervised learning in one paragraph."},
{"role": "assistant", "content": "Supervised learning trains a model on labeled inputoutput pairs, learning to map inputs to known outputs. Unsupervised learning finds patterns in unlabeled data without any target outputs. Supervised learning is used for classification and regression; unsupervised learning is used for clustering, dimensionality reduction, and anomaly detection."}
]
},
{
"messages": [
{"role": "system", "content": "You are a helpful, concise, and accurate AI assistant."},
{"role": "user", "content": "Translate 'Good morning, how are you?' into Spanish."},
{"role": "assistant", "content": "Buenos días, ¿cómo estás?"}
]
},
{
"messages": [
{"role": "system", "content": "You are a helpful, concise, and accurate AI assistant."},
{"role": "user", "content": "What are three key advantages of using transformers over RNNs for NLP tasks?"},
{"role": "assistant", "content": "Three key advantages of transformers over RNNs are: first, parallel processing of sequences rather than sequential computation, enabling much faster
training; second, the self-attention mechanism captures long-range dependencies without the vanishing gradient problems that affect RNNs; and third, transformers scale more effectively with larger datasets and model sizes, producing better performance as compute increases."}
]
},
{
"messages": [
{"role": "system", "content": "You are a helpful, concise, and accurate AI assistant."},
{"role": "user", "content": "I need to cancel my subscription. What should I do?"},
{"role": "assistant", "content": "To cancel your subscription, go to your Account Settings, select Billing or Subscription, and click Cancel Subscription. You will keep access until the end of your current billing period. If you have trouble finding this option, contact support with your account email and they will assist you."}
]
},
]
sft_dataset = Dataset.from_list(SFT_DATA)
print(f"SFT dataset: {len(sft_dataset)} instruction examples")
print()
print("Sample SFT example (system + user messages):")
sample = SFT_DATA[4]
for msg in sample["messages"][:-1]:
print(f" [{msg['role'].upper()}]: {msg['content'][:100]}...")
print(f" [ASSISTANT]: {sample['messages'][-1]['content'][:100]}...")
# Load model and run SFT stage
from transformers import TrainingArguments
MODEL_ID = "microsoft/phi-2"
quantization_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16,
bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
MODEL_ID,
quantization_config=quantization_config,
device_map="auto",
trust_remote_code=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
r=16, lora_alpha=32,
target_modules=["q_proj", "k_proj", "v_proj", "dense"],
lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
def format_sft_example(examples):
formatted = []
for messages in examples["messages"]:
text = ""
for msg in messages:
if msg["role"] == "system":
text += f"<|system|>\n{msg['content']}\n"
elif msg["role"] == "user":
text += f"<|user|>\n{msg['content']}\n"
elif msg["role"] == "assistant":
text += f"<|assistant|>\n{msg['content']}\n"
formatted.append(text)
return formatted
sft_args = TrainingArguments(
output_dir="./sft_output",
num_train_epochs=3,
per_device_train_batch_size=1,
gradient_accumulation_steps=8,
learning_rate=2e-4,
lr_scheduler_type="cosine",
warmup_ratio=0.1,
optim="paged_adamw_32bit",
fp16=not torch.cuda.is_bf16_supported(),
bf16=torch.cuda.is_bf16_supported(),
logging_steps=1,
save_steps=100,
report_to="none",
)
sft_trainer = SFTTrainer(
model=model,
train_dataset=sft_dataset,
args=sft_args,
formatting_func=format_sft_example,
max_seq_length=512,
tokenizer=tokenizer,
packing=False,
)
print("\nStage 1: Running SFT...")
print("-" * 60)
sft_result = sft_trainer.train()
print(f"\nSFT complete. Loss: {sft_result.training_loss:.4f}")
sft_trainer.save_model("./sft_model")
print("SFT model saved to ./sft_model")
#Stage 2 — DPO preference dataset
# Each example: prompt (instruction), chosen (preferred response), rejected (less preferred)
DPO_DATA = [
{
"prompt": "<|system|>\nYou are a helpful, concise, and accurate AI assistant.\n<|user|>\nExplain what machine learning is.\n<|assistant|>\n",
"chosen": "Machine learning is a subset of artificial intelligence where systems learn from data to improve their performance on tasks without being explicitly programmed. Models identify patterns in training data and use them to make predictions or decisions on new, unseen data.",
"rejected": "Machine learning is a thing where computers learn stuff from data. It is pretty complicated and involves a lot of math. There are many types of machine learning like supervised and unsupervised and they all work differently in various ways that are hard to explain briefly.",
},
{
"prompt": "<|system|>\nYou are a helpful, concise, and accurate AI assistant.\n<|user|>\nWhat is the best way to learn Python?\n<|assistant|>\n",
"chosen": "The most effective approach is: start with the official Python tutorial to learn syntax, then work through a structured course like CS50P or Python.org's beginner guide, practice daily by building small projects, and read other people's code on GitHub. Consistency matters more than intensity.",
"rejected": "You should definitely learn Python because it is a great language. There are many resources online. Some people like books and some people like videos. You should try different things and see what works for you. The main thing is to practice a lot and never give up even when it gets hard.",
},
{
"prompt": "<|system|>\nYou are a helpful, concise, and accurate AI assistant.\n<|user|>\nHow do I reverse a string in Python?\n<|assistant|>\n",
"chosen": "You can reverse a string in Python using slice notation:\n\n```python\nreversed_str = original_str[::-1]\n```\n\nThis creates a new string with all characters in reverse order. For example, `'hello'[::-1]` returns `'olleh'`.",
"rejected": "Reversing a string is something you can do in Python. You would need to use some kind of loop or maybe a built-in function. Python has many ways to work with strings. You could also convert it to a list first. There are different approaches and they all work in their own way.",
},
{
"prompt": "<|system|>\nYou are a helpful, concise, and accurate AI assistant.\n<|user|>\nWhat should I do if my code has a bug?\n<|assistant|>\n",
"chosen": "Debug systematically: first reproduce the bug reliably, then isolate where it occurs by adding print statements or using a debugger (pdb for Python). Read any error messages
carefully, check your assumptions about what the code does, and search for the error message online if it is unclear. Fix one thing at a time and test after each change.",
"rejected": "Bugs happen to everyone so do not feel bad. You should look at your code carefully and try to find where the problem is. Sometimes restarting your computer helps or closing and reopening your editor. You could also try asking a friend or posting on Stack Overflow.",
},
{
"prompt": "<|system|>\nYou are a helpful, concise, and accurate AI assistant.\n<|user|>\nWhat is a REST API?\n<|assistant|>\n",
"chosen": "A REST API (Representational State Transfer Application Programming Interface) is a standard way for systems to communicate over HTTP. It uses standard HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources identified by URLs. REST APIs return data, typically JSON, allowing different applications and languages to exchange information in a consistent, stateless way.",
"rejected": "REST API stands for something with the letters REST. It is a way that computers talk to each other on the internet. APIs are important in modern software development. Many big companies use APIs. You can use them to get data from websites or do things on other services.",
},
]
dpo_dataset = Dataset.from_list(DPO_DATA)
print(f"DPO dataset: {len(dpo_dataset)} preference pairs")
print()
print("Sample preference comparison:")
sample = DPO_DATA[0]
print(f"Prompt: {sample['prompt'][:80]}...")
print(f"Chosen: {sample['chosen'][:120]}...")
print(f"Rejected: {sample['rejected'][:120]}...")
#DPO training stage
from peft import PeftModel
# Load the SFT model as the policy to be aligned
# In production you would load the full SFT model; here we continue from our trained adapter
# The reference model is the frozen SFT model
dpo_config = DPOConfig(
output_dir="./dpo_output",
num_train_epochs=2,
per_device_train_batch_size=1,
gradient_accumulation_steps=4,
learning_rate=5e-5,
beta=0.1,
lr_scheduler_type="cosine",
warmup_ratio=0.1,
optim="paged_adamw_32bit",
fp16=not torch.cuda.is_bf16_supported(),
bf16=torch.cuda.is_bf16_supported(),
logging_steps=1,
report_to="none",
max_length=512,
max_prompt_length=256,
remove_unused_columns=False,
)
print("DPO configuration:")
print(f" Beta (KL strength): {dpo_config.beta}")
print(f" Learning rate: {dpo_config.learning_rate}")
print(f" Epochs: {dpo_config.num_train_epochs}")
print()
print("DPO training requires the SFT model as both the policy and the frozen reference.")
print("The DPOTrainer automatically handles the reference model by keeping a frozen copy.")
print()
dpo_trainer = DPOTrainer(
model=model,
args=dpo_config,
train_dataset=dpo_dataset,
tokenizer=tokenizer,
)
print("Stage 2: Running DPO alignment...")
print("-" * 60)
dpo_result = dpo_trainer.train()
print(f"\nDPO alignment complete. Loss: {dpo_result.training_loss:.4f}")
dpo_trainer.save_model("./dpo_model")
print("DPO-aligned model saved to ./dpo_model")
#Compare SFT vs DPO model outputs
def generate_response(prompt_text, max_new_tokens=200):
full_prompt = f"<|system|>\nYou are a helpful, concise, and accurate AI assistant.\n<|user|>\n{prompt_text}\n<|assistant|>\n"
inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
outputs = model.generate(
**inputs,
max_new_tokens=max_new_tokens,
temperature=0.1,
do_sample=True,
pad_token_id=tokenizer.eos_token_id,
)
return tokenizer.decode(
outputs[0][inputs["input_ids"].shape[1]:],
skip_special_tokens=True
).strip()
evaluation_prompts = [
"What is the difference between a list and a tuple in Python?",
"How does gradient descent work?",
"What are three best practices for writing clean code?",
]
print("Post-alignment model responses:")
print("=" * 65)
for prompt in evaluation_prompts:
response = generate_response(prompt)
print(f"Q: {prompt}")
print(f"A: {response}")
print()
#DPO preference quality analysis
# Analyze the quality signals in our preference dataset before and after alignment
def analyze_preference_quality(dataset):
results = []
for ex in dataset:
chosen_words = len(ex["chosen"].split())
rejected_words = len(ex["rejected"].split())
chosen_has_code = "```" in ex["chosen"] or "def " in ex["chosen"]
rejected_has_code = "```" in ex["rejected"] or "def " in ex["rejected"]
results.append({
"chosen_words": chosen_words,
"rejected_words": rejected_words,
"length_ratio": round(chosen_words / rejected_words, 2),
"chosen_has_code": chosen_has_code,
"rejected_has_code": rejected_has_code,
})
df = pd.DataFrame(results)
return df
analysis = analyze_preference_quality(DPO_DATA)
print("Preference Dataset Quality Analysis")
print("=" * 50)
print(f"\nChosen responses — avg words: {analysis['chosen_words'].mean():.0f}")
print(f"Rejected responses — avg words: {analysis['rejected_words'].mean():.0f}")
print(f"Avg length ratio (chosen/rejected): {analysis['length_ratio'].mean():.2f}")
print()
print("Key observation:")
print(" Chosen responses are more specific, structured, and actionable.")
print(" Rejected responses tend to be vague, hedging, or repetitive.")
print(" DPO will train the model to favor the chosen response patterns.")
print()
print("Detailed per-example analysis:")
print(analysis.to_string(index=False))
#ORPO — unified SFT and alignment in one stage (structural demonstration)
ORPO_TRAINING_PATTERN = '''
# ORPO combines SFT and preference alignment into a single training stage.
# Dataset format: each example includes both the positive (chosen) response
# and the negative (rejected) response alongside the instruction.
# Install the latest TRL which includes ORPO support:
# !pip install -q trl>=0.8.6
from trl import ORPOConfig, ORPOTrainer
# ORPO dataset format: same as DPO (prompt, chosen, rejected)
# but training simultaneously optimizes SFT loss + odds ratio penalty
orpo_config = ORPOConfig(
output_dir="./orpo_output",
num_train_epochs=3,
per_device_train_batch_size=1,
gradient_accumulation_steps=8,
learning_rate=8e-6,
lambda_=0.1, # weight of the odds ratio penalty term
lr_scheduler_type="cosine",
warmup_ratio=0.1,
optim="paged_adamw_32bit",
bf16=True,
logging_steps=1,
report_to="none",
)
# ORPO trainer uses the same interface as DPOTrainer
# but combines SFT and preference optimization in one pass
orpo_trainer = ORPOTrainer(
model=model,
args=orpo_config,
train_dataset=dpo_dataset, # same preference dataset format as DPO
tokenizer=tokenizer,
)
# orpo_trainer.train()
# Key advantage: no separate SFT stage needed before alignment
# Single training run produces a model that both follows instructions
# and is aligned to human preferences simultaneously.
'''
print("ORPO Training Pattern (single-stage SFT + alignment):")
print("=" * 65)
print(ORPO_TRAINING_PATTERN)
print("\nComparison: SFT->DPO vs ORPO")
print("-" * 50)
comparison = pd.DataFrame({
"Property": ["Training stages", "Models in memory", "Dataset requirement",
"Typical LR", "Best for"],
"SFT then DPO": ["2 (SFT + DPO)", "2 (policy + frozen ref)", "SFT data + preference pairs",
"SFT: 2e-4, DPO: 5e-5", "Separate optimization of instruction following and alignment"],
"ORPO": ["1 (combined)", "1 (policy only)", "Preference pairs with chosen as SFT target",
"8e-6", "Simplified pipeline, competitive with two-stage"],
})
print(comparison.to_string(index=False))

