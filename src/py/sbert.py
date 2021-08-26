import numpy
import torch
from transformers import AutoTokenizer, AutoModel


# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')


def get_embeddings(utterances: list[str], batch_size=64) -> numpy.ndarray:
    batches = []
    cur = 0
    while cur < len(utterances):
        last = min(len(utterances), cur + batch_size)
        # Tokenize sentences
        encoded_input = tokenizer(utterances[cur: last], padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])

        batches.append(sentence_embeddings.cpu().detach().numpy())
        cur = last
        print('Get embeddings: {}/{}'.format(cur, len(utterances)))
    return numpy.concatenate(batches)


if __name__ == '__main__':
    print(type(get_embeddings(['This is an example sentence', 'Each sentence is converted'])))
