from torch.utils.data import Dataset
import torch

class YourDataset(Dataset):
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