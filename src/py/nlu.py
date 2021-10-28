import os
from typing import Callable

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MPNetPreTrainedModel, MPNetModel

from sbert import ClassificationHead, TaggerHead, _cls, _split_text_and_slots_into_tokens_and_tags, _encode_tags, \
    _remap_clusters, _finetune_model

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

        if 'st_num_labels' in kwargs:
            self.config.st_num_labels = kwargs.pop('st_num_labels')
        if 'ic_num_labels' in kwargs:
            self.config.ic_num_labels = kwargs.pop('ic_num_labels')

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


def fine_tune_nlu_model(train_texts, train_slots, train_intents,
                        n_train_epochs=None, n_train_steps=None,
                        base_model_path='sentence-transformers/paraphrase-mpnet-base-v2',
                        eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0):
    global nlu_model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    train_texts, train_tags = _split_text_and_slots_into_tokens_and_tags(train_texts, train_slots)
    # Tag set
    unique_tags = dict.fromkeys(tag for doc in train_tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True, return_tensors='pt')
    train_slot_labels = _encode_tags(train_tags, train_encodings, tag2id)
    train_encodings.pop("offset_mapping")

    # Intent labels
    train_intent_labels = _remap_clusters(train_intents)

    nlu_model = NLUModel.from_pretrained(base_model_path,
                                         st_num_labels=len(unique_tags),
                                         ic_num_labels=len(dict.fromkeys(train_intent_labels)))

    train_dataset = NLUTrainingDataset(train_encodings, train_slot_labels, train_intent_labels)

    _finetune_model(nlu_model, train_dataset, n_train_epochs=n_train_epochs, n_train_steps=n_train_steps,
                    eval_callback=eval_callback,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience)


def load_finetuned(model_path):
    global tokenizer, nlu_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    nlu_model = NLUModel.from_pretrained(model_path)
    nlu_model.to(device)


def save_finetuned(model_path):
    tokenizer.save_pretrained(model_path)
    nlu_model.save_pretrained(model_path)


if __name__ == '__main__':
    train_texts, train_slots, train_intents \
        = ['this is firsteeeee sentenceeeee', 'this is second with oov word qwertyuiop asdfghjkl'], \
          [{'slot_1': {'start': 8, 'end': 18}, 'slot_2': {'start': 0, 'end': 4}},
           {'slot_1': {'start': 24, 'end': 49}, 'slot_2': {'start': 0, 'end': 4}}], \
          ['intent_1', 'intent_2'],
    fine_tune_nlu_model(train_texts, train_slots, train_intents, n_train_epochs=3)

    save_finetuned('./models/temp_model')
    load_finetuned('./models/temp_model')
