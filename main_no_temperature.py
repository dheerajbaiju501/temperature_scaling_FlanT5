from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,BitsAndBytesConfig
import torch
from tqdm import tqdm
from datasets import load_dataset

class Args:
    def __init__(self, load_in_8bit=True):
        self.load_in_8bit = load_in_8bit
        

def get_score(model, tokenizer, input_ids):
    pos_ids = tokenizer('Yes').input_ids
    neg_ids = tokenizer('No').input_ids
    pos_id = pos_ids[0]
    neg_id = neg_ids[0]
    
    print(f"Positive ID: {pos_id}, Negative ID: {neg_id}")
    
    logits = model(input_ids, decoder_input_ids=torch.zeros((input_ids.size(0), 1), dtype=torch.long)).logits
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


print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xxl')
print("Tokenizer loaded")

args = Args(load_in_8bit=True)
model = AutoModelForSeq2SeqLM.from_pretrained('soumyasanyal/nli-entailment-verifier-xxl',torch_dtype = torch.bfloat16,quantization_config = load_quantization_config(args),device_map = 'auto')
print("Model loaded.")
ds = load_dataset("stanfordnlp/snli")

premise = "Franco Zeffirelli, KBE Grande Ufficiale OMRI (] ; born 12 February 1923) is an Italian director and producer of operas, films and television. He is also a former senator (1994\u20132001) for the Italian centre-right \"Forza Italia\" party. Recently, Italian researchers have found that he is one of the few distant relatives of Leonardo da Vinci."
hypothesis = "Franco Zeffirelli had a political career"
prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nGiven the premise, is the hypothesis correct?\nAnswer:"

print("Tokenizing input...")
input_ids = tokenizer(prompt, return_tensors='pt').input_ids
print(f"Input shape: {input_ids.shape}")

print("Calculating scores...")
scores = get_score(model, tokenizer, input_ids)
print(f'Hypothesis entails the premise: {bool(scores >= 0.5)}')