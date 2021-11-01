import json
import os
from typing import Callable, List

import numpy
import torch
from scipy.special import softmax
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MPNetPreTrainedModel, MPNetModel

from sbert import ClassificationHead, TaggerHead, _cls, _remap_clusters, _finetune_model, \
    _split_text_and_slots_into_tokens_and_tags, _encode_tags

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

tokenizer = None
nlu_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NLUTrainingDataset(torch.utils.data.Dataset):
    # slot_labels \in tuple[Literal[0.0,1.0]] (for multi-class)
    # intent_labels \in {0,1,2,3,...} (for single-class)
    def __init__(self, encodings, slot_labels, intent_labels):
        self.encodings = encodings
        self.slot_labels = slot_labels
        self.intent_labels = intent_labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['st_labels'] = torch.tensor(self.slot_labels[idx])
        item['ic_labels'] = torch.tensor(self.intent_labels[idx])
        return item

    def __len__(self):
        return len(self.intent_labels)


class NLUModel(MPNetPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        if 'id2st_label' in kwargs:
            self.config.st_labels = kwargs.pop('id2st_label')
            self.config.st_num_labels = len(self.config.st_labels)
        if 'id2ic_label' in kwargs:
            self.config.ic_labels = kwargs.pop('id2ic_label')
            self.config.ic_num_labels = len(self.config.ic_labels)

        self.base = MPNetModel(config, add_pooling_layer=False)

        self.slot_tagger = TaggerHead(config, self.config.st_num_labels)
        self.st_loss_fct = CrossEntropyLoss()

        self.intent_classifier = ClassificationHead(config, self.config.ic_num_labels)
        self.ic_loss_fct = CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            st_labels=None,
            ic_labels=None,
    ):
        outputs = self.base(input_ids, attention_mask=attention_mask)

        # logits
        ic_logits = self.intent_classifier(cls=_cls(outputs))
        st_logits = self.slot_tagger(outputs[0])

        # losses
        loss = None
        if ic_labels is not None and st_labels is not None:
            ic_loss = self.ic_loss_fct(ic_logits, ic_labels)

            # Only keep active parts of the loss
            active_st_logits = st_logits.view(-1, self.config.st_num_labels)
            active_st_labels = torch.where(
                attention_mask.view(-1) == 1,
                st_labels.view(-1),
                torch.tensor(self.st_loss_fct.ignore_index).type_as(st_labels)
            )
            st_loss = self.st_loss_fct(active_st_logits, active_st_labels)

            loss = torch.add(st_loss, ic_loss)

        output = (st_logits, ic_logits) + outputs[2:]

        return ((loss,) + output) if loss is not None else output


def _split_into_tokens_and_tags(texts, slots):
    train_encodings = tokenizer(texts, padding=True, truncation=True, return_offsets_mapping=True, return_tensors='pt')
    tags = []
    for i in range(len(texts)):
        input_ids, attention_mask, offset_mapping = \
            train_encodings['input_ids'][i], train_encodings['attention_mask'][i], train_encodings['offset_mapping'][i]
        u_tags = [None] * len(attention_mask)

        n = torch.sum(attention_mask).item() - 2

        for j in range(1, n + 1):
            tag = None
            for slot, info in slots[i].items():
                if offset_mapping[j][0] == info['start']:
                    assert offset_mapping[j][0] <= info['end']
                    assert tag is None
                    tag = 'B_' + slot
                    last_b = tag
                elif info['start'] < offset_mapping[j][0] < info['end']:
                    assert offset_mapping[j][1] <= info['end']
                    assert tag is None
                    assert last_b == 'B_' + slot
                    tag = 'I_' + slot

            if tag is None:
                last_b = None
            u_tags[j] = 'O' if tag is None else tag

        tags.append(u_tags)

    train_encodings.pop("offset_mapping")
    return train_encodings, tags


def fine_tune_nlu_model_2(train_texts, train_slots, train_intents,
                          n_train_epochs=None, n_train_steps=None,
                          base_model_path='./models/sentence-transformers.paraphrase-mpnet-base-v2',
                          eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0):
    global nlu_model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    train_encodings, train_tags = _split_into_tokens_and_tags(train_texts, train_slots)
    # Tag set
    id2tag = list(dict.fromkeys(tag for doc in train_tags for tag in doc if tag is not None).keys())
    tag2id = {tag: id for id, tag in enumerate(id2tag)}

    train_slot_labels = [[-100 if tag is None else tag2id[tag] for tag in doc] for doc in train_tags]

    # Intent labels
    train_intent_labels, id2label, _ = _remap_clusters(train_intents)

    nlu_model = NLUModel.from_pretrained(base_model_path, id2st_label=id2tag, id2ic_label=id2label)

    train_dataset = NLUTrainingDataset(train_encodings, train_slot_labels, train_intent_labels)

    _finetune_model(nlu_model, train_dataset, n_train_epochs=n_train_epochs, n_train_steps=n_train_steps,
                    eval_callback=eval_callback,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience,
                    save_fct=save_finetuned, load_fct=load_finetuned)


def get_intents_and_slots_2(utterances: List[str], batch_size=64):
    def extract_intent_and_slots_from_logits(st_logits, ic_logits, text, input_ids, attention_mask, offset_mapping):
        # intent
        ic_prob = softmax(ic_logits, axis=-1)
        ic_idx = numpy.argmax(ic_prob)

        # slots
        st_prob = softmax(st_logits, axis=-1)
        n = numpy.sum(attention_mask) - 2

        # DP decoding
        cost = numpy.zeros((n + 1, nlu_model.config.st_num_labels))
        trace = numpy.zeros((n + 1, nlu_model.config.st_num_labels), dtype=int)
        for t in range(nlu_model.config.st_num_labels):
            cost[0][t] == 0 if nlu_model.config.st_labels[t] == 'O' else -1
        for c in range(1, n + 1):
            for t in range(nlu_model.config.st_num_labels):
                label = nlu_model.config.st_labels[t]
                if label == 'O' or label.startswith('B_'):
                    best_previous_t = numpy.argmax(cost[c - 1])
                    cost[c][t] = -1 if cost[c - 1][best_previous_t] == -1 else cost[c - 1][best_previous_t] + \
                                                                               st_prob[c][t]
                    trace[c][t] = best_previous_t
                else:
                    cost[c][t] = -1
                    for p in range(nlu_model.config.st_num_labels):
                        if nlu_model.config.st_labels[p] not in [label, 'B_' + label[2:]] or cost[c - 1][p] == -1:
                            continue
                        if cost[c - 1][p] + st_prob[c][t] > cost[c][t]:
                            cost[c][t] = cost[c - 1][p] + st_prob[c][t]
                            trace[c][t] = p
        # decode
        tags = [0] * (n + 1)
        prob = [0] * (n + 1)

        t = numpy.argmax(cost[n])
        total_prob = cost[n][t] / n
        assert total_prob >= 0

        c = n
        while c > 0:
            tags[c] = nlu_model.config.st_labels[t]
            prob[c] = st_prob[c][t]
            t = trace[c][t]
            c -= 1

        # resolve
        slots = []
        for c in range(1, n + 1):
            if tags[c].startswith('B_'):
                j = c
                while j < n and tags[j + 1].startswith('I_'):
                    j += 1
                slots.append(
                    {
                        'slot': tags[c][2:],
                        'start': offset_mapping[c][0].item(),
                        'end': offset_mapping[j][1].item(),
                        'value': text[offset_mapping[c][0]:offset_mapping[j][1]],
                        'prob': numpy.average(prob[c:j + 1]).item()
                        # prob of a slot is the average prob of its constituent tags
                    }
                )

        return {
            'intent': (nlu_model.config.ic_labels[ic_idx], ic_prob[ic_idx].item()),
            'slots': {
                'total_prob': total_prob,
                'slots': slots
            },
            'tokens': list(zip(
                tokenizer.convert_ids_to_tokens(input_ids[1:n + 1]),
                tags[1:n + 1],
                [p.item() for p in prob[1:n + 1]]
            ))
        }

    outputs = []
    cur = 0
    while cur < len(utterances):
        last = min(len(utterances), cur + batch_size)
        # Tokenize sentences
        encoded_input = tokenizer(utterances[cur: last], padding=True, truncation=True, return_offsets_mapping=True,
                                  return_tensors='pt')
        encoded_input.to(device)

        offset_mapping = encoded_input.pop('offset_mapping')

        # Compute token embeddings
        with torch.no_grad():
            model_output = nlu_model(**encoded_input)
        st_logits, ic_logits = model_output[0], model_output[1]

        for i in range(len(st_logits)):
            outputs.append(extract_intent_and_slots_from_logits(st_logits[i].detach().cpu().numpy(),
                                                                ic_logits[i].detach().cpu().numpy(),
                                                                utterances[cur + i],
                                                                encoded_input['input_ids'][i].detach().cpu().numpy(),
                                                                encoded_input['attention_mask'][
                                                                    i].detach().cpu().numpy(),
                                                                offset_mapping[i].detach().cpu().numpy()))

        cur = last
        print('\rInferencing: {}/{} ({:.1f}%)'.format(cur, len(utterances), 100 * cur / len(utterances)),
              end='' if cur < len(utterances) else '\n')

    return outputs


def fine_tune_nlu_model(train_texts, train_slots, train_intents,
                        n_train_epochs=None, n_train_steps=None,
                        base_model_path='sentence-transformers/paraphrase-mpnet-base-v2',
                        eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0):
    global nlu_model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    train_texts, train_tags = _split_text_and_slots_into_tokens_and_tags(train_texts, train_slots)
    # Tag set
    id2tag = list(dict.fromkeys(tag for doc in train_tags for tag in doc).keys())
    tag2id = {tag: id for id, tag in enumerate(id2tag)}
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True, return_tensors='pt')
    train_slot_labels = _encode_tags(train_tags, train_encodings, tag2id)
    train_encodings.pop("offset_mapping")

    # Intent labels
    train_intent_labels, id2label, _ = _remap_clusters(train_intents)

    nlu_model = NLUModel.from_pretrained(base_model_path, id2st_label=id2tag, id2ic_label=id2label)

    train_dataset = NLUTrainingDataset(train_encodings, train_slot_labels, train_intent_labels)

    _finetune_model(nlu_model, train_dataset, n_train_epochs=n_train_epochs, n_train_steps=n_train_steps,
                    eval_callback=eval_callback,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience,
                    save_fct=save_finetuned, load_fct=load_finetuned)


def load_finetuned(model_path, from_tf=False):
    global tokenizer, nlu_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, from_tf=from_tf)
    nlu_model = NLUModel.from_pretrained(model_path, from_tf=from_tf)
    nlu_model.to(device)


def save_finetuned(model_path):
    tokenizer.save_pretrained(model_path)
    nlu_model.save_pretrained(model_path)


def get_intents_and_slots(tokenized_utterances: List[List[str]], batch_size=64):
    def extract_intent_and_slots_from_logits(st_logits, ic_logits, tokenized_text,
                                             input_ids, attention_mask, offset_mapping):
        # intent
        ic_prob = softmax(ic_logits, axis=-1)
        ic_idx = numpy.argmax(ic_prob)

        # slots
        active_ids = [i for i, offset in enumerate(offset_mapping) if offset[0] == 0]
        st_logits = st_logits[active_ids]
        input_ids = input_ids[active_ids]
        attention_mask = attention_mask[active_ids]

        st_prob = softmax(st_logits, axis=-1)
        n = numpy.sum(attention_mask) - 2

        # DP decoding
        cost = numpy.zeros((n + 1, nlu_model.config.st_num_labels))
        trace = numpy.zeros((n + 1, nlu_model.config.st_num_labels), dtype=int)
        for t in range(nlu_model.config.st_num_labels):
            cost[0][t] == 0 if nlu_model.config.st_labels[t] == 'O' else -1
        for c in range(1, n + 1):
            for t in range(nlu_model.config.st_num_labels):
                label = nlu_model.config.st_labels[t]
                if label == 'O' or label.startswith('B_'):
                    best_previous_t = numpy.argmax(cost[c - 1])
                    cost[c][t] = -1 if cost[c - 1][best_previous_t] == -1 else cost[c - 1][best_previous_t] + \
                                                                               st_prob[c][t]
                    trace[c][t] = best_previous_t
                else:
                    cost[c][t] = -1
                    for p in range(nlu_model.config.st_num_labels):
                        if nlu_model.config.st_labels[p] not in [label, 'B_' + label[2:]] or cost[c - 1][p] == -1:
                            continue
                        if cost[c - 1][p] + st_prob[c][t] > cost[c][t]:
                            cost[c][t] = cost[c - 1][p] + st_prob[c][t]
                            trace[c][t] = p
        # decode
        tags = [0] * (n + 1)
        prob = [0] * (n + 1)

        t = numpy.argmax(cost[n])
        total_prob = cost[n][t] / n
        assert total_prob >= 0

        c = n
        while c > 0:
            tags[c] = nlu_model.config.st_labels[t]
            prob[c] = st_prob[c][t]
            t = trace[c][t]
            c -= 1

        # resolve
        slots = []
        for c in range(1, n + 1):
            if tags[c].startswith('B_'):
                j = c
                while j < n and tags[j + 1].startswith('I_'):
                    j += 1
                slots.append(
                    {
                        'slot': tags[c][2:],
                        'start_token': c - 1,
                        'end_token': j,
                        'value': ' '.join(tokenized_text[c - 1:j]),
                        'prob': numpy.average(prob[c:j + 1]).item()
                    }
                )

        return {
            'intent': (nlu_model.config.ic_labels[ic_idx], ic_prob[ic_idx].item()),
            'slots': {
                'total_prob': total_prob,
                'slots': slots
            },
            'tokens': list(zip(tokenized_text, tags[1:n + 1], [p.item() for p in prob[1:n + 1]]))
        }

    outputs = []
    cur = 0
    while cur < len(tokenized_utterances):
        last = min(len(tokenized_utterances), cur + batch_size)
        # Tokenize sentences
        encoded_input = tokenizer(tokenized_utterances[cur: last], is_split_into_words=True, padding=True,
                                  truncation=True,
                                  return_offsets_mapping=True, return_tensors='pt')
        encoded_input.to(device)

        offset_mapping = encoded_input.pop('offset_mapping')

        # Compute token embeddings
        with torch.no_grad():
            model_output = nlu_model(**encoded_input)
        st_logits, ic_logits = model_output[0], model_output[1]

        for i in range(len(st_logits)):
            outputs.append(extract_intent_and_slots_from_logits(st_logits[i].detach().cpu().numpy(),
                                                                ic_logits[i].detach().cpu().numpy(),
                                                                tokenized_utterances[cur + i],
                                                                encoded_input['input_ids'][i].detach().cpu().numpy(),
                                                                encoded_input['attention_mask'][
                                                                    i].detach().cpu().numpy(),
                                                                offset_mapping[i].detach().cpu().numpy()))

        cur = last
        print('\rInferencing: {}/{} ({:.1f}%)'.format(cur, len(tokenized_utterances),
                                                      100 * cur / len(tokenized_utterances)),
              end='' if cur < len(tokenized_utterances) else '\n')

    return outputs


if __name__ == '__main__':
    # train_texts, train_slots, train_intents \
    #     = ['this is firsteeeee sentenceeeee', 'this is second with oov word qwertyuiop asdfghjkl'], \
    #       [{'slot_1': {'start': 8, 'end': 18}, 'slot_2': {'start': 0, 'end': 4}},
    #        {'slot_1': {'start': 24, 'end': 49}, 'slot_2': {'start': 0, 'end': 4}}], \
    #       ['intent_1', 'intent_2'],
    # fine_tune_nlu_model(train_texts, train_slots, train_intents, n_train_epochs=3)
    #
    # save_finetuned('./models/temp_model')
    # load_finetuned('./models/temp_model')

    load_finetuned('models/snips_inter-intent_nlu/inter_intent/nlu_model')
    print(json.dumps(get_intents_and_slots(['I want to see movie times at 11:12'.split()]), indent=2))
