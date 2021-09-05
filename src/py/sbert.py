import random
from math import ceil
from typing import List

import numpy
import torch
from datasets import load_metric
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(model_path='sentence-transformers/paraphrase-mpnet-base-v2', from_tf=False):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_path, from_tf=from_tf)
    model = AutoModel.from_pretrained(model_path, from_tf=from_tf)
    model.to(device)


def save(model_path):
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)


# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeddings(utterances: List[str], batch_size=64) -> numpy.ndarray:
    batches = []
    cur = 0
    while cur < len(utterances):
        last = min(len(utterances), cur + batch_size)
        # Tokenize sentences
        encoded_input = tokenizer(utterances[cur: last], padding=True, truncation=True, return_tensors='pt')
        encoded_input.to(device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])

        batches.append(sentence_embeddings.detach().cpu().numpy())
        cur = last
        print('\rGet embeddings: {}/{} ({:.1f}%)'.format(cur, len(utterances), 100 * cur / len(utterances)),
              end='' if cur < len(utterances) else '\n')
    return numpy.concatenate(batches)


class ClassificationDataset(torch.utils.data.Dataset):
    # labels \in {0,1,2,3,...}
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, base_model_config, num_labels):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(base_model_config.hidden_dropout_prob)
        self.out_proj = nn.Linear(base_model_config.hidden_size, num_labels)

        print('Initializing classifier head')
        self.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.zero_()

    def forward(self, mean_pooling, **kwargs):
        x = torch.tanh(mean_pooling)
        x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PseudoClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.config = base_model.config
        self.num_labels = num_labels
        self.base_model = base_model
        self.classifier = ClassificationHead(base_model.config, num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
        )
        logits = self.classifier(_mean_pooling(outputs, attention_mask))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            # The first should be loss, used by trainer. The second should be logits, used by compute_metrics
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def fine_tune_classification(train_texts, train_labels,
                             val_texts=None, val_labels=None, test_texts=None, test_labels=None):
    label_set = set(train_labels)
    if val_labels is not None:
        label_set.update(val_labels)
    if test_labels is not None:
        label_set.update(test_labels)
    label_map = {l: i for i, l in enumerate(label_set)}

    train_labels = [label_map[l] for l in train_labels]
    if val_labels is not None:
        val_labels = [label_map[l] for l in val_labels]
    if test_labels is not None:
        test_labels = [label_map[l] for l in test_labels]

    metric_accuracy = load_metric("accuracy")

    def compute_accuracy(eval_pred):
        logits, labels = eval_pred
        predictions = numpy.argmax(logits, axis=-1)
        return metric_accuracy.compute(predictions=predictions, references=labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True,
                              return_tensors='pt') if val_texts is not None else None
    test_encodings = tokenizer(test_texts, truncation=True, padding=True,
                               return_tensors='pt') if test_texts is not None else None

    train_dataset = ClassificationDataset(train_encodings, train_labels)
    val_dataset = ClassificationDataset(val_encodings, val_labels) if val_texts is not None else None
    test_dataset = ClassificationDataset(test_encodings, test_labels) if test_texts is not None else None

    classifier = PseudoClassificationModel(model, len(label_set))

    # print(classifier(**train_encodings))

    trainer = Trainer(
        model=classifier,
        args=TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy='epoch' if val_dataset is not None else 'no'
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_accuracy,
    )
    if test_dataset is not None:
        print('Evaluate test_dataset (before):', trainer.evaluate(test_dataset))

    trainer.train()

    if test_dataset is not None:
        print('Evaluate test_dataset (after):', trainer.evaluate(test_dataset))


class UtteranceSimilarityDataset(torch.utils.data.Dataset):
    # labels \in {-1,0,1,2,3,...}, -1 means unseen
    def __init__(self, encodings, labels, negative_to_positive_rate: int = 3):
        self.unseen_indices = []
        self.n_seen_utterances = 0
        self.seen_indices = [[] for _ in range(max(labels) + 1)]

        for i, l in enumerate(labels):
            if l != -1:
                self.seen_indices[l].append(i)
                self.n_seen_utterances += 1
            else:
                self.unseen_indices.append(i)

        self.encodings = encodings
        self.labels = labels
        self.dataset_len = self.n_seen_utterances * (1 + negative_to_positive_rate)

    def _get_positive_pair(self, idx):
        for c in self.seen_indices:
            if idx >= len(c):
                idx -= len(c)
            else:
                return c[idx], random.choice(c)
        raise Exception('Invalid idx')

    def _get_negative_pair(self, idx):
        for cpos_idx, cpos in enumerate(self.seen_indices):
            if idx >= len(cpos):
                idx -= len(cpos)
            else:
                idx2 = random.randrange(self.n_seen_utterances + len(self.unseen_indices) - len(cpos))
                for cneg_idx, cneg in enumerate(self.seen_indices):
                    if cneg_idx == cpos_idx:
                        continue
                    if idx2 >= len(cneg):
                        idx2 -= len(cneg)
                    else:
                        return cpos[idx], cneg[idx2]
                return cpos[idx], self.unseen_indices[idx2]
        raise Exception('Invalid idx')

    def __getitem__(self, idx):
        item = {}
        if idx < self.n_seen_utterances:  # positive sample
            idx, idx2 = self._get_positive_pair(idx)
            item['labels'] = 1.0
        else:  # negative sample
            idx, idx2 = self._get_negative_pair(idx % self.n_seen_utterances)
            item['labels'] = -1.0

        for key, val in self.encodings.items():
            item[key] = torch.stack((val[idx].clone().detach(), val[idx2].clone().detach()))

        return item

    def __len__(self):
        return self.dataset_len


class UtteranceSimilarityModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.base_model = base_model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
    ):
        input_ids_0, attention_mask_0 = input_ids[:, 0, :], attention_mask[:, 0, :]
        input_ids_1, attention_mask_1 = input_ids[:, 1, :], attention_mask[:, 1, :]
        first_outputs = _mean_pooling(self.base_model(input_ids_0, attention_mask=attention_mask_0),
                                      attention_mask_0)
        second_outputs = _mean_pooling(self.base_model(input_ids_1, attention_mask=attention_mask_1),
                                       attention_mask_1)
        loss_fct = MSELoss()
        loss = loss_fct(cosine_similarity(first_outputs, second_outputs), labels)

        return (loss,)


# labels: None means unseen
def fine_tune_utterance_similarity(train_texts, train_labels,
                                   val_texts=None, val_labels=None, test_texts=None, test_labels=None,
                                   n_train_epochs=-1, n_train_steps=-1, negative_to_positive_rate=3):
    label_set = set(train_labels)
    if val_labels is not None:
        label_set.update(val_labels)
    if test_labels is not None:
        label_set.update(test_labels)

    label_map = {None: -1}
    label_count = 0
    for l in label_set:
        if l is not None:
            label_map[l] = label_count
            label_count += 1

    train_labels = [label_map[l] for l in train_labels]
    if val_labels is not None:
        val_labels = [label_map[l] for l in val_labels]
    if test_labels is not None:
        test_labels = [label_map[l] for l in test_labels]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True,
                              return_tensors='pt') if val_texts is not None else None
    test_encodings = tokenizer(test_texts, truncation=True, padding=True,
                               return_tensors='pt') if test_texts is not None else None

    train_dataset = UtteranceSimilarityDataset(train_encodings, train_labels,
                                               negative_to_positive_rate=negative_to_positive_rate)
    val_dataset = UtteranceSimilarityDataset(val_encodings, val_labels,
                                             negative_to_positive_rate=negative_to_positive_rate) if val_texts is not None else None
    test_dataset = UtteranceSimilarityDataset(test_encodings, test_labels,
                                              negative_to_positive_rate=negative_to_positive_rate) if test_texts is not None else None

    estimator = UtteranceSimilarityModel(model)

    # print(estimator(**train_encodings))

    if n_train_epochs == -1:
        if n_train_steps == -1:
            n_train_steps = 10000
        n_train_epochs = max(ceil(n_train_steps / len(train_dataset)), 3)

    trainer = Trainer(
        model=estimator,
        args=TrainingArguments(
            output_dir='./results',
            num_train_epochs=n_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy='epoch' if val_dataset is not None else 'no'
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    if test_dataset is not None:
        print('Evaluate test_dataset (before):', trainer.evaluate(test_dataset))

    trainer.train()

    if test_dataset is not None:
        print('Evaluate test_dataset (after):', trainer.evaluate(test_dataset))


if __name__ == '__main__':
    load('sentence-transformers/paraphrase-mpnet-base-v2')

    # train_texts, train_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 0, 1, 0, 1, 0, 1, 0]
    # val_texts, val_labels = ['i', 'j', 'k', 'l'], [1, 0, 1, 0]
    # fine_tune_classification(train_texts, train_labels, val_texts, val_labels)

    # train_texts, train_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 0, 1, 0, 1, 0, None, None]
    # val_texts, val_labels = ['i', 'j', 'k', 'l'], [1, 0, 1, 0]
    # fine_tune_utterance_similarity(train_texts, train_labels)
