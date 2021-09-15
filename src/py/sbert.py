import random
from math import ceil
from typing import List

import numpy
import torch
from datasets import load_metric
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, set_seed, AdamW

set_seed(12993)

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
    # labels \in {0,1,2,3,...} (for single-class) or tuple[Literal[0.0,1.0]] (for multi-class)
    def __init__(self, encodings, labels, sample_weights=None):
        self.encodings = encodings
        self.labels = labels
        self.sample_weights = sample_weights

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        if self.sample_weights is not None:
            item['sample_weights'] = torch.tensor(self.sample_weights[idx])
        return item

    def __len__(self):
        return len(self.labels)


class ClassificationHead(nn.Module):
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
        # mean_pooling = torch.tanh(mean_pooling)
        x = self.dropout(mean_pooling)
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
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            sample_weights=None,
    ):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(_mean_pooling(outputs, attention_mask))

        per_sample_loss = self.loss_fct(logits, labels)
        if sample_weights is not None:
            per_sample_loss = torch.mul(per_sample_loss, sample_weights)
        loss = torch.mean(per_sample_loss)

        output = (logits,) + outputs[2:]
        # The first should be loss, used by trainer. The second should be logits, used by compute_metrics
        return (loss,) + output


def fine_tune_pseudo_classification(train_texts, train_labels, train_sample_weights=None,
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

    train_dataset = ClassificationDataset(train_encodings, train_labels, train_sample_weights)
    val_dataset = ClassificationDataset(val_encodings, val_labels) if val_texts is not None else None
    test_dataset = ClassificationDataset(test_encodings, test_labels) if test_texts is not None else None

    classifier = PseudoClassificationModel(model, len(label_set))

    # print(classifier(**train_encodings))

    trainer = Trainer(
        model=classifier,
        args=TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
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
    def __init__(self, encodings, labels,
                 negative_sampling_rate_from_seen: int = 2,
                 negative_sampling_rate_from_unseen: float = 0.5):
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

        self.positive_bound = self.n_seen_utterances
        self.negative_from_seen_bound = self.positive_bound + self.n_seen_utterances * negative_sampling_rate_from_seen
        self.negative_from_unseen_bound = self.negative_from_seen_bound + int(
            len(self.unseen_indices) * negative_sampling_rate_from_unseen)

    def _get_positive_pair(self, idx):
        for c in self.seen_indices:
            if idx >= len(c):
                idx -= len(c)
            else:
                return c[idx], random.choice(c)
        raise Exception('Invalid idx')

    def _get_negative_pair_from_seen(self, idx):
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

    def _get_negative_pair_from_unseen(self):
        idx = random.randrange(self.n_seen_utterances)

        for c in self.seen_indices:
            if idx >= len(c):
                idx -= len(c)
            else:
                return random.choice(self.unseen_indices), c[idx]

    def __getitem__(self, idx):
        item = {}
        if idx < self.positive_bound:
            idx, idx2 = self._get_positive_pair(idx)
            item['labels'] = torch.tensor(1.0)
        elif idx < self.negative_from_seen_bound:
            idx, idx2 = self._get_negative_pair_from_seen(idx % self.n_seen_utterances)
            item['labels'] = torch.tensor(-1.0)
        else:
            idx, idx2 = self._get_negative_pair_from_unseen()
            item['labels'] = torch.tensor(-1.0)
        for key, val in self.encodings.items():
            item[key] = torch.stack((val[idx].clone().detach(), val[idx2].clone().detach()))

        return item

    def __len__(self):
        return self.negative_from_unseen_bound


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
                                   n_train_epochs=-1, n_train_steps=-1,
                                   negative_sampling_rate_from_seen=2, negative_sampling_rate_from_unseen=0.5):
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
                                               negative_sampling_rate_from_seen=negative_sampling_rate_from_seen,
                                               negative_sampling_rate_from_unseen=negative_sampling_rate_from_unseen)
    val_dataset = UtteranceSimilarityDataset(val_encodings, val_labels,
                                             negative_sampling_rate_from_seen=negative_sampling_rate_from_seen,
                                             negative_sampling_rate_from_unseen=negative_sampling_rate_from_unseen) if val_texts is not None else None
    test_dataset = UtteranceSimilarityDataset(test_encodings, test_labels,
                                              negative_sampling_rate_from_seen=negative_sampling_rate_from_seen,
                                              negative_sampling_rate_from_unseen=negative_sampling_rate_from_unseen) if test_texts is not None else None

    estimator = UtteranceSimilarityModel(model)

    # print(estimator(**train_encodings))

    if n_train_epochs == -1:
        if n_train_steps == -1:
            n_train_steps = 2000
        n_train_epochs = max(ceil(n_train_steps / len(train_dataset)), 2)

    trainer = Trainer(
        model=estimator,
        args=TrainingArguments(
            save_strategy='no',
            output_dir='./results',
            num_train_epochs=n_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_strategy='no',
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


class SlotTaggingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TaggerHead(nn.Module):
    """Head for token-level classification tasks."""

    def __init__(self, base_model_config, num_labels):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(base_model_config.hidden_dropout_prob)
        self.out_proj = nn.Linear(base_model_config.hidden_size, num_labels)

        print('Initializing tagger head')
        self.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.zero_()

    def forward(self, base_output, **kwargs):
        x = self.dropout(base_output)
        x = self.out_proj(x)
        return x


class SlotTaggingModel(nn.Module):

    def __init__(self, base_model, num_labels):
        super().__init__()
        self.config = base_model.config
        self.num_labels = num_labels
        self.base_model = base_model
        self.tagger = TaggerHead(base_model.config, num_labels)
        self.loss_fct = CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.tagger(sequence_output)

        # Only keep active parts of the loss
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels)
        )
        loss = self.loss_fct(active_logits, active_labels)

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


def _split_text_and_slots_into_tokens_and_tags(texts, slots):
    texts = [text.split() for text in texts]

    tags = []

    for i, tokens in enumerate(texts):
        last_b = None
        u_tags = []
        cur = 0
        for j, t in enumerate(tokens):
            tag = None
            if j > 0:
                cur += 1
            for slot, info in slots[i].items():
                if cur == info['start']:
                    assert cur + len(t) <= info['end']
                    assert tag is None
                    tag = 'B_' + slot
                    last_b = tag
                elif info['start'] < cur < info['end']:
                    assert cur + len(t) <= info['end']
                    assert tag is None
                    assert last_b == 'B_' + slot
                    tag = 'I_' + slot

            if tag is None:
                last_b = None
            u_tags.append('O' if tag is None else tag)
            cur += len(t)

        tags.append(u_tags)

    return texts, tags


def _encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = numpy.ones(len(doc_offset), dtype=int) * -100
        arr_offset = numpy.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def fine_tune_slot_tagging(train_texts, train_slots,
                           val_texts=None, val_slots=None, test_texts=None, test_slots=None,
                           n_train_epochs=-1, n_train_steps=-1):
    train_texts, train_tags = _split_text_and_slots_into_tokens_and_tags(train_texts, train_slots)

    if val_texts is not None:
        val_texts, val_tags = _split_text_and_slots_into_tokens_and_tags(val_texts, val_slots)
    if test_texts is not None:
        test_texts, test_tags = _split_text_and_slots_into_tokens_and_tags(test_texts, test_slots)

    # Tag set
    unique_tags = set(tag for doc in train_tags for tag in doc)
    if val_texts is not None:
        unique_tags.update(tag for doc in val_tags for tag in doc)
    if test_texts is not None:
        unique_tags.update(tag for doc in test_tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True, return_tensors='pt')
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                              truncation=True, return_tensors='pt') if val_texts is not None else None
    test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                               truncation=True, return_tensors='pt') if test_texts is not None else None

    train_labels = _encode_tags(train_tags, train_encodings, tag2id)
    val_labels = _encode_tags(val_tags, val_encodings, tag2id) if val_texts is not None else None
    test_labels = _encode_tags(test_tags, test_encodings, tag2id) if test_texts is not None else None

    train_encodings.pop("offset_mapping")
    if val_texts is not None:
        val_encodings.pop("offset_mapping")
    if test_texts is not None:
        test_encodings.pop("offset_mapping")

    train_dataset = SlotTaggingDataset(train_encodings, train_labels)
    val_dataset = SlotTaggingDataset(val_encodings, val_labels) if val_texts is not None else None
    test_dataset = SlotTaggingDataset(test_encodings, test_labels) if test_texts is not None else None

    tagger = SlotTaggingModel(model, len(unique_tags))

    if n_train_epochs == -1:
        if n_train_steps == -1:
            n_train_steps = 2000
        n_train_epochs = max(ceil(n_train_steps / len(train_dataset)), 3)

    trainer = Trainer(
        model=tagger,
        args=TrainingArguments(
            save_strategy='no',
            output_dir='./results',
            num_train_epochs=n_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_strategy='no',
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


# train_cluster_labels is None means unseen cluster
def fine_tune_joint_slot_tagging_and_utterance_similarity(train_texts, train_slots, train_cluster_labels,
                                                          n_train_epochs=-1, n_train_steps=-1,
                                                          us_negative_sampling_rate_from_seen=2,
                                                          us_negative_sampling_rate_from_unseen=0.5,
                                                          ):
    # Prepare for slot tagging
    train_texts_sr = [train_texts[i] for i, l in enumerate(train_cluster_labels) if l is not None]
    train_slots = [s for s in train_slots if s is not None]
    train_texts_sr, train_tags = _split_text_and_slots_into_tokens_and_tags(train_texts_sr, train_slots)
    unique_tags = set(tag for doc in train_tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    train_encodings_sr = tokenizer(train_texts_sr, is_split_into_words=True, return_offsets_mapping=True,
                                   padding=True, truncation=True, return_tensors='pt')
    train_slot_labels = _encode_tags(train_tags, train_encodings_sr, tag2id)
    train_encodings_sr.pop("offset_mapping")
    sr_train_dataset = SlotTaggingDataset(train_encodings_sr, train_slot_labels)
    sr_train_loader = DataLoader(sr_train_dataset, batch_size=16, shuffle=True)
    tagger = SlotTaggingModel(model, len(unique_tags))
    sr_optim = AdamW(tagger.parameters(), lr=5e-5)

    # Prepare for utterance similarity
    label_set = set(train_cluster_labels)
    label_map = {None: -1}
    label_count = 0
    for l in label_set:
        if l is not None:
            label_map[l] = label_count
            label_count += 1

    train_cluster_labels = [label_map[l] for l in train_cluster_labels]
    train_encodings_us = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    us_train_dataset = UtteranceSimilarityDataset(train_encodings_us, train_cluster_labels,
                                                  negative_sampling_rate_from_seen=us_negative_sampling_rate_from_seen,
                                                  negative_sampling_rate_from_unseen=us_negative_sampling_rate_from_unseen)
    us_train_loader = DataLoader(us_train_dataset, batch_size=16, shuffle=True)
    estimator = UtteranceSimilarityModel(model)
    us_optim = AdamW(estimator.parameters(), lr=5e-5)

    # Compute optimal n_epochs
    if n_train_epochs == -1:
        if n_train_steps == -1:
            n_train_steps = 2000
        n_train_epochs = max(ceil(n_train_steps / min(len(sr_train_dataset), len(us_train_dataset))), 3)

    tagger.to(device)
    estimator.to(device)

    tagger.train()  # Switch mode
    estimator.train()

    train_ids_len = len(us_train_loader) + len(sr_train_loader)

    cur = 0
    for epoch in range(n_train_epochs):
        train_ids = list(range(train_ids_len))
        random.shuffle(train_ids)
        us_train_loader_iter = iter(us_train_loader)
        sr_train_loader_iter = iter(sr_train_loader)
        for idx in train_ids:
            cur += 1
            print('\rJoint training: {}/{} ({:.1f}%)'.format(cur, train_ids_len * n_train_epochs,
                                                             100 * cur / (train_ids_len * n_train_epochs)),
                  end='' if cur < train_ids_len * n_train_epochs else '\n')
            if idx < len(us_train_loader):
                us_optim.zero_grad()
                batch = next(us_train_loader_iter)
                batch = {key: val.to(device) for key, val in batch.items()}
                outputs = estimator(**batch)
                loss = outputs[0]
                loss.backward()
                us_optim.step()
            else:
                sr_optim.zero_grad()
                batch = next(sr_train_loader_iter)
                batch = {key: val.to(device) for key, val in batch.items()}
                outputs = tagger(**batch)
                loss = outputs[0]
                loss.backward()
                sr_optim.step()

    tagger.eval()  # Switch mode
    estimator.eval()


class SlotMulticlassClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.config = base_model.config
        self.num_labels = num_labels
        self.base_model = base_model
        self.classifier = ClassificationHead(base_model.config, num_labels)
        self.loss_fct = BCEWithLogitsLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
    ):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
        )
        logits = self.classifier(_mean_pooling(outputs, attention_mask))

        loss = self.loss_fct(logits, labels)

        output = (logits,) + outputs[2:]
        # The first should be loss, used by trainer. The second should be logits, used by compute_metrics
        return (loss,) + output


def fine_tune_slot_multiclass_classification(train_texts, train_slots,
                                             val_texts=None, val_slots=None, test_texts=None, test_slots=None,
                                             n_train_epochs=-1, n_train_steps=-1):
    # Tag set
    unique_tags = set(tag for doc in train_slots for tag in doc.keys())
    if val_texts is not None:
        unique_tags.update(tag for doc in val_slots for tag in doc.keys())
    if test_texts is not None:
        unique_tags.update(tag for doc in test_slots for tag in doc.keys())

    train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
    val_encodings = tokenizer(val_texts, padding=True, truncation=True, return_tensors='pt') \
        if val_texts is not None else None
    test_encodings = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt') \
        if test_texts is not None else None

    train_labels = [[1.0 if s in u_slots else 0.0 for s in unique_tags] for u_slots in train_slots]
    val_labels = [[1.0 if s in u_slots else 0.0 for s in unique_tags] for u_slots in val_slots] \
        if val_texts is not None else None
    test_labels = [[1.0 if s in u_slots else 0.0 for s in unique_tags] for u_slots in test_slots] \
        if test_texts is not None else None

    train_dataset = ClassificationDataset(train_encodings, train_labels)
    val_dataset = ClassificationDataset(val_encodings, val_labels) if val_texts is not None else None
    test_dataset = ClassificationDataset(test_encodings, test_labels) if test_texts is not None else None

    classifier = SlotMulticlassClassificationModel(model, len(unique_tags))

    if n_train_epochs == -1:
        if n_train_steps == -1:
            n_train_steps = 2000
        n_train_epochs = max(ceil(n_train_steps / len(train_dataset)), 3)

    trainer = Trainer(
        model=classifier,
        args=TrainingArguments(
            save_strategy='no',
            output_dir='./results',
            num_train_epochs=n_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_strategy='no',
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

    # train_texts, train_labels, train_labels_weights \
    #     = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 0, 1, 0, 1, 0, 1, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # val_texts, val_labels = ['i', 'j', 'k', 'l'], [1, 0, 1, 0]
    # fine_tune_pseudo_classification(train_texts, train_labels, train_sample_weights=train_labels_weights)

    # train_texts, train_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 0, 1, 0, 1, 0, None, None]
    # val_texts, val_labels = ['i', 'j', 'k', 'l'], [1, 0, 1, 0]
    # fine_tune_utterance_similarity(train_texts, train_labels)

    # train_texts, train_slots \
    #     = ['this is first sentence', 'this is second with oov word qwertyuiop asdfghjkl'], \
    #       [{'slot_1': {'start': 8, 'end': 22}, 'slot_2': {'start': 0, 'end': 4}},
    #        {'slot_1': {'start': 24, 'end': 49}, 'slot_2': {'start': 0, 'end': 4}}, ]
    # fine_tune_slot_multiclass_classification(train_texts, train_slots)
    # fine_tune_slot_tagging(train_texts, train_slots)

    # train_texts, train_slots, train_labels \
    #     = ['this is first sentence', 'this is second with oov word qwertyuiop asdfghjkl', 'test'], \
    #       [{'slot_1': {'start': 8, 'end': 22}, 'slot_2': {'start': 0, 'end': 4}},
    #        {'slot_1': {'start': 24, 'end': 49}, 'slot_2': {'start': 0, 'end': 4}}], \
    #       [0, 1, None]
    # fine_tune_joint_slot_tagging_and_utterance_similarity(train_texts, train_slots, train_labels)
