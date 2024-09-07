import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
class Dataset_modif(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        if 'text' not in item:
            raise KeyError(f"Item at index {idx} does not contain 'text' field")

        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        if 'label' in item:
            label = torch.tensor(item['label'])
        elif 'labels' in item:
            label = torch.tensor(item['labels'])
        else:
            raise KeyError(f"Item at index {idx} does not contain 'label' or 'labels' field")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


class ModelWithTemperatureSeq2Seq(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Initial temperature
    
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if hasattr(output, 'logits'):
            output.logits = self.temperature_scale(output.logits)
        return output


    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, **kwargs):
        if decoder_input_ids is None:
            decoder_input_ids = torch.zeros((input_ids.size(0), 1), dtype=torch.long, device=input_ids.device)
        
        outputs = self.model(input_ids=input_ids, 
                             attention_mask=attention_mask, 
                             decoder_input_ids=decoder_input_ids, 
                             **kwargs)
        
        if hasattr(outputs, 'logits'):
            outputs.logits = self.temperature_scale(outputs.logits)
        return outputs

    def temperature_scale(self, logits):
        
        return logits / self.temperature
    
    def _prepare_encoder_decoder_kwargs_for_generation(self, *args, **kwargs):
        return self.model._prepare_encoder_decoder_kwargs_for_generation(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        decoder_input_ids = torch.zeros((input_ids.size(0), 1), dtype=torch.long, device=input_ids.device)
        return self.model.generate(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=decoder_input_ids,**kwargs)
    
    
    
    def set_temperature(self, valid_loader):
        # Tune the temperature using the validation set (logits and labels)
        nll_criterion = nn.CrossEntropyLoss()
        logits_list = []
        labels_list = []

        pos_ids = self.tokenizer('Yes').input_ids
        neg_ids = self.tokenizer('No').input_ids
        pos_id = pos_ids[0]
        neg_id = neg_ids[0]

        # Collect all logits and labels from the validation set
        with torch.no_grad():
            for batch in valid_loader:
                if 'input_ids' not in batch:
                    if'text' in batch:
                        inputs = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                        input_ids = inputs.input_ids.to(self.model.device)
                    else:
                        raise ValueError("Neither 'input_ids' nor 'text' found in the batch")
                else:
                    input_ids = batch['input_ids'].to(self.model.device)

                if 'label' in batch:
                    labels = batch['label'].to(self.model.device)
                elif 'labels' in batch:
                    labels = batch['labels'].to(self.model.device)
                else:
                    raise ValueError("No 'label' or 'labels' found in the batch")
                

                logits = self.model(input_ids).logits
                pos_logits = logits[:, 0, pos_id]
                neg_logits = logits[:, 0, neg_id]
                posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1)

                logits_list.append(posneg_logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Calculate NLL before scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        print(f"Before temperature - NLL: {before_temperature_nll:.4f}")

        # Optimize temperature parameter
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate NLL after scaling
        after_temperature_nll = nll_criterion(logits / self.temperature, labels).item()
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        print(f"After temperature - NLL: {after_temperature_nll:.4f}")

        return self.temperature.item()