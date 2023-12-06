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


    def get_sequences(self, document: list, n_queries: int = 5) -> list[str]:
        if n_queries == 0:
            return []
        input_ids = self.tokenizer.encode(document, truncation=True, return_tensors='pt')
        outputs = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    num_beams=10,
                    max_new_tokens=5,
                    no_repeat_ngram_size=1,
                    remove_invalid_values=True,
                    num_return_sequences=n_queries)
        queries = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return queries