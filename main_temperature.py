from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,BitsAndBytesConfig,Trainer, TrainingArguments
import torch
from tqdm import tqdm
from datasets import load_dataset
from temperature_scaling import ModelWithTemperatureSeq2Seq, Dataset_modif
import temperature_scaling
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

class Args:
    def __init__(self, load_in_8bit=True):
        self.load_in_8bit = load_in_8bit
def get_score(model, tokenizer, input_ids):
    pos_ids = tokenizer('Yes').input_ids
    neg_ids = tokenizer('No').input_ids
    pos_id = pos_ids[0]
    neg_id = neg_ids[0]
    
    print(f"Positive ID: {pos_id}, Negative ID: {neg_id}")
    
    decoder_input_ids = torch.zeros((input_ids.size(0), 1), dtype=torch.long, device=input_ids.device)
    output = model(input_ids, decoder_input_ids=decoder_input_ids)
    
    logits = output.logits
    print(f"Logits shape: {logits.shape}")
    
    pos_logits = logits[:, 0, pos_id]
    neg_logits = logits[:, 0, neg_id]
    print(f"Positive logits: {pos_logits}, Negative logits: {neg_logits}")
    
    posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1)
    scores = torch.nn.functional.softmax(posneg_logits, dim=1)[:, 0]
    print(f"Scores: {scores}")
    return scores
def load_quantization_config(args):
	# Define the quantization configs
	config_8bit = BitsAndBytesConfig(load_in_8bit=True)
	
	if args.load_in_8bit:
		quantization_config = config_8bit
	
	else:
		quantization_config = None

	return quantization_config




tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xxl')
args = Args(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained('soumyasanyal/nli-entailment-verifier-xxl',torch_dtype = torch.bfloat16,quantization_config = load_quantization_config(args),device_map = 'auto')
model = temperature_scaling.ModelWithTemperatureSeq2Seq(model,tokenizer)

dataset = load_dataset("snli")

lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # Whether to use bias in the LoRA layers
    task_type="SEQ_2_SEQ_LM",  # Task type for LoRA
    target_modules = ["q","v"]
)
model = get_peft_model(model, lora_config)

valid_data = [{'text': f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}", 
               'label': example['label']} 
              for example in dataset['validation']]

temp_dataset = Dataset_modif(valid_data, tokenizer)
temp_loader = DataLoader(temp_dataset, batch_size=8, shuffle=False)

# Set the temperature
model.set_temperature(temp_loader)

def preprocess_function(examples):
    inputs = [ex for ex in examples['premise']]
    targets = [ex for ex in examples['hypothesis']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

trainer.train()

# Save the final model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")



