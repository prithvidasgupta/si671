"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques and build your own tokenizer.
"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ComplaintBinner:
    def __init__(self, model_name) -> None:
        self.device = torch.device('cuda')
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def get_sequences(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        top_p = 0.95

        if n_queries == 0:
            return []
        if document == '':
            return []
        input_ids = self.tokenizer.encode(f'{prefix_prompt}{document}', truncation=True, return_tensors='pt')
        outputs = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    do_sample=True,
                    top_p=top_p,
                    num_return_sequences=n_queries)
        queries = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return queries