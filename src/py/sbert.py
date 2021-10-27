import os
import random
import tempfile
from math import ceil
from typing import List, Callable

import numpy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, set_seed, AdamW

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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


def _cls(model_output):
    return model_output[0][:, 0, :]

# map to 0,1,2...
def _remap_clusters(clusters):
    label_set = dict.fromkeys(clusters)
    label_map = {l: i for i, l in enumerate(label_set)}
    return [label_map[l] for l in clusters]


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


def _finetune_model(finetune_model, train_dataset, n_train_epochs=None, n_train_steps=None,
                    eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    optim = AdamW(finetune_model.parameters(), lr=5e-5, weight_decay=0.01)

    finetune_model.to(device)

    with tempfile.TemporaryDirectory() as temp_dir:
        if not early_stopping:
            if n_train_epochs is None:
                n_train_epochs = ceil((n_train_steps if n_train_steps is not None else 2000) / len(train_dataset))
            """ Uncomment the below block for using Huggingface's Trainer interface.
            Trainer(
                model=finetune_model,
                args=TrainingArguments(
                    save_strategy='no',
                    output_dir=os.path.join(temp_dir, 'results'),
                    num_train_epochs=n_train_epochs,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=64,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_strategy='no',
                    evaluation_strategy='no'
                ),
                train_dataset=train_dataset,
            ).train()
            """

            for epoch in range(n_train_epochs):
                finetune_model.train()  # Switch mode
                cur = 0
                for batch in train_loader:
                    cur += 1
                    print('\r==== Epoch: {} Num examples: {} Batch: {}/{} ({:.1f}%)'
                          .format(epoch + 1, len(train_dataset), cur, len(train_loader), 100 * cur / len(train_loader)),
                          end='' if cur < len(train_loader) else '\n')
                    optim.zero_grad()
                    batch = {key: val.to(device) for key, val in batch.items()}
                    outputs = finetune_model(**batch)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                finetune_model.eval()

                if eval_callback is not None:
                    print('Validation score: {:.3f}'.format(eval_callback()))
        else:
            best_epoch = None
            best_eval = None
            epoch = 0
            while True:
                epoch += 1

                finetune_model.train()  # Switch mode
                cur = 0
                for batch in train_loader:
                    cur += 1
                    print('\r==== Epoch: {} Num examples: {} Batch: {}/{} ({:.1f}%)'
                          .format(epoch, len(train_dataset), cur, len(train_loader), 100 * cur / len(train_loader)),
                          end='' if cur < len(train_loader) else '\n')
                    optim.zero_grad()
                    batch = {key: val.to(device) for key, val in batch.items()}
                    outputs = finetune_model(**batch)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                finetune_model.eval()

                eval = eval_callback()
                print('Validation score: {:.3f}'.format(eval), end='')
                if best_eval is None or eval > best_eval:
                    best_eval = eval
                    best_epoch = epoch
                    print(' -> Save model')
                    save(temp_dir)
                else:
                    if epoch > best_epoch + early_stopping_patience:
                        print(' -> Stop')
                        load(temp_dir)
                        break
                    else:
                        print(' -> Be patient')


def _joint_finetune_model(finetune_model_1, finetune_model_2, train_dataset_1, train_dataset_2,
                          n_train_epochs=None, n_train_steps=None,
                          eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0):
    train_loader_1 = DataLoader(train_dataset_1, batch_size=16, shuffle=True)
    train_loader_2 = DataLoader(train_dataset_2, batch_size=16, shuffle=True)
    optim_1 = AdamW(finetune_model_1.parameters(), lr=5e-5, weight_decay=0.01)
    optim_2 = AdamW(finetune_model_2.parameters(), lr=5e-5, weight_decay=0.01)

    finetune_model_1.to(device)
    finetune_model_2.to(device)

    with tempfile.TemporaryDirectory() as temp_dir:
        if not early_stopping:
            if n_train_epochs is None:
                n_train_epochs = ceil(
                    (n_train_steps if n_train_steps is not None else 2000) / min(len(train_dataset_1),
                                                                                 len(train_dataset_2)))

            train_ids_len = len(train_loader_1) + len(train_loader_2)
            for epoch in range(n_train_epochs):
                finetune_model_1.train()  # Switch mode
                finetune_model_2.train()

                train_ids = list(range(train_ids_len))
                random.shuffle(train_ids)
                train_loader_1_iter = iter(train_loader_1)
                train_loader_2_iter = iter(train_loader_2)

                cur_1 = 0
                cur_2 = 0
                for idx in train_ids:
                    if idx < len(train_loader_1):
                        cur_1 += 1
                    else:
                        cur_2 += 1
                    print('\r==== Joint training: Epoch: {} Batch: {}+{}/{} ({:.1f}%)'
                          .format(epoch + 1, cur_1, cur_2, len(train_ids), 100 * (cur_1 + cur_2) / len(train_ids)),
                          end='' if cur_1 + cur_2 < len(train_ids) else '\n')
                    if idx < len(train_loader_1):
                        optim_1.zero_grad()
                        batch = next(train_loader_1_iter)
                        batch = {key: val.to(device) for key, val in batch.items()}
                        outputs = finetune_model_1(**batch)
                        loss = outputs[0]
                        loss.backward()
                        optim_1.step()
                    else:
                        optim_2.zero_grad()
                        batch = next(train_loader_2_iter)
                        batch = {key: val.to(device) for key, val in batch.items()}
                        outputs = finetune_model_2(**batch)
                        loss = outputs[0]
                        loss.backward()
                        optim_2.step()

                finetune_model_1.eval()  # Switch mode
                finetune_model_2.eval()

                if eval_callback is not None:
                    print('Validation score: {:.3f}'.format(eval_callback()))
        else:
            best_epoch = None
            best_eval = None
            epoch = 0
            while True:
                epoch += 1

                train_ids = list(range(len(train_loader_1) + len(train_loader_2)))
                random.shuffle(train_ids)
                train_loader_1_iter = iter(train_loader_1)
                train_loader_2_iter = iter(train_loader_2)

                finetune_model_1.train()  # Switch mode
                finetune_model_2.train()

                cur_1 = 0
                cur_2 = 0
                for idx in train_ids:
                    if idx < len(train_loader_1):
                        cur_1 += 1
                    else:
                        cur_2 += 1
                    print('\r==== Joint training: Epoch: {} Batch: {}+{}/{} ({:.1f}%)'
                          .format(epoch, cur_1, cur_2, len(train_ids), 100 * (cur_1 + cur_2) / len(train_ids)),
                          end='' if cur_1 + cur_2 < len(train_ids) else '\n')
                    if idx < len(train_loader_1):
                        optim_1.zero_grad()
                        batch = next(train_loader_1_iter)
                        batch = {key: val.to(device) for key, val in batch.items()}
                        outputs = finetune_model_1(**batch)
                        loss = outputs[0]
                        loss.backward()
                        optim_1.step()
                    else:
                        optim_2.zero_grad()
                        batch = next(train_loader_2_iter)
                        batch = {key: val.to(device) for key, val in batch.items()}
                        outputs = finetune_model_2(**batch)
                        loss = outputs[0]
                        loss.backward()
                        optim_2.step()

                finetune_model_1.eval()  # Switch mode
                finetune_model_2.eval()

                eval = eval_callback()
                print('Validation score: {:.3f}'.format(eval), end='')
                if best_eval is None or eval > best_eval:
                    best_eval = eval
                    best_epoch = epoch
                    print(' -> Save model')
                    save(temp_dir)
                else:
                    if epoch > best_epoch + early_stopping_patience:
                        print(' -> Stop')
                        load(temp_dir)
                        break
                    else:
                        print(' -> Be patient')


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
    def __init__(self, base_model_config, num_labels, n_dense_layers=0, mean_pooling_activation=False):
        super().__init__()
        self.dense = nn.ModuleList([nn.Linear(base_model_config.hidden_size, base_model_config.hidden_size)
                                    for _ in range(n_dense_layers)])
        self.dropout = nn.Dropout(base_model_config.hidden_dropout_prob)
        self.out_proj = nn.Linear(base_model_config.hidden_size, num_labels)
        self.mean_pooling_activation = mean_pooling_activation

        print('Initializing classifier head')
        for d in self.dense:
            d.weight.data.normal_(mean=0.0, std=0.02)
            if d.bias is not None:
                d.bias.data.zero_()
        self.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.zero_()

    def forward(self, cls=None, mean=None, **kwargs):
        x = cls if cls is not None else mean

        if cls is None and self.mean_pooling_activation:
            x = torch.tanh(x)

        for d in self.dense:
            x = torch.tanh(d(self.dropout(x)))

        return self.out_proj(self.dropout(x))


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
        logits = self.classifier(mean=_mean_pooling(outputs, attention_mask))

        per_sample_loss = self.loss_fct(logits, labels)
        if sample_weights is not None:
            per_sample_loss = torch.mul(per_sample_loss, sample_weights)
        loss = torch.mean(per_sample_loss)

        output = (logits,) + outputs[2:]
        # The first should be loss, used by trainer. The second should be logits, used by compute_metrics
        return (loss,) + output


def fine_tune_pseudo_classification(train_texts, train_cluster_ids, train_sample_weights=None,
                                    previous_classifier=None, previous_optim=None):
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    train_dataset = ClassificationDataset(train_encodings, train_cluster_ids, train_sample_weights)

    if previous_classifier is None:
        classifier = PseudoClassificationModel(model, len(dict.fromkeys(train_cluster_ids)))
        classifier.to(device)
    else:
        classifier = previous_classifier

    # print(classifier(**train_encodings))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    optim = AdamW(classifier.parameters(), lr=5e-5, weight_decay=0.01) if previous_optim is None else previous_optim

    classifier.train()  # Switch mode
    cur = 0
    for batch in train_loader:
        cur += 1
        print('\rNum examples: {} Batch: {}/{} ({:.1f}%)'
              .format(len(train_dataset), cur, len(train_loader), 100 * cur / len(train_loader)),
              end='' if cur < len(train_loader) else '\n')
        optim.zero_grad()
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = classifier(**batch)
        loss = outputs[0]
        loss.backward()
        optim.step()
    classifier.eval()

    return classifier, optim


class UtteranceSimilarityDataset(torch.utils.data.Dataset):
    # labels \in {-1,0,1,2,3,...}, -1 means unseen
    def __init__(self, encodings, labels,
                 negative_sampling_rate_from_seen: int = 3,
                 negative_sampling_rate_from_unseen: float = 0.0):
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
                return c[idx], random.choice(self.unseen_indices)

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
def fine_tune_utterance_similarity(
        train_texts, train_labels,
        n_train_epochs=None, n_train_steps=None,
        negative_sampling_rate_from_seen=3, negative_sampling_rate_from_unseen=0.0,
        eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0
):
    label_set = dict.fromkeys(train_labels)
    label_map = {None: -1}
    label_count = 0
    for l in label_set:
        if l is not None:
            label_map[l] = label_count
            label_count += 1
    train_labels = [label_map[l] for l in train_labels]

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    train_dataset = UtteranceSimilarityDataset(train_encodings, train_labels,
                                               negative_sampling_rate_from_seen=negative_sampling_rate_from_seen,
                                               negative_sampling_rate_from_unseen=negative_sampling_rate_from_unseen)

    estimator = UtteranceSimilarityModel(model)

    _finetune_model(estimator, train_dataset, n_train_epochs=n_train_epochs, n_train_steps=n_train_steps,
                    eval_callback=eval_callback,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience)


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

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)

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


def fine_tune_slot_tagging(train_texts, train_slots, n_train_epochs=None, n_train_steps=None,
                           eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0):
    train_texts, train_tags = _split_text_and_slots_into_tokens_and_tags(train_texts, train_slots)

    # Tag set
    unique_tags = dict.fromkeys(tag for doc in train_tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True, return_tensors='pt')
    train_labels = _encode_tags(train_tags, train_encodings, tag2id)

    train_encodings.pop("offset_mapping")

    train_dataset = SlotTaggingDataset(train_encodings, train_labels)

    tagger = SlotTaggingModel(model, len(unique_tags))

    _finetune_model(tagger, train_dataset, n_train_epochs=n_train_epochs, n_train_steps=n_train_steps,
                    eval_callback=eval_callback,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience)


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
        logits = self.classifier(cls=_cls(outputs))

        loss = self.loss_fct(logits, labels)

        output = (logits,) + outputs[2:]
        # The first should be loss, used by trainer. The second should be logits, used by compute_metrics
        return (loss,) + output


def fine_tune_slot_multiclass_classification(
        train_texts, train_slots, n_train_epochs=None, n_train_steps=None,
        eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0
):
    # Tag set
    unique_tags = dict.fromkeys(tag for doc in train_slots for tag in doc.keys())

    train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
    train_labels = [[1.0 if s in u_slots else 0.0 for s in unique_tags] for u_slots in train_slots]

    train_dataset = ClassificationDataset(train_encodings, train_labels)

    classifier = SlotMulticlassClassificationModel(model, len(unique_tags))

    _finetune_model(classifier, train_dataset, n_train_epochs=n_train_epochs, n_train_steps=n_train_steps,
                    eval_callback=eval_callback,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience)


class JointUtteranceSimilarityAndSlotClassificationDataset(torch.utils.data.Dataset):
    # cluster_labels \in {-1,0,1,2,3,...}, -1 means unseen
    # slot_labels: tuple[Literal[0.0,1.0]] or None for unseen TODO: remove this line
    # slot_labels: tuple[Literal[0.0,1.0]]
    def __init__(self, encodings, cluster_labels, slot_labels,
                 us_negative_sampling_rate_from_seen: int = 3,
                 us_negative_sampling_rate_from_unseen: float = 0.0):
        self.unseen_indices = []
        self.n_seen_utterances = 0
        self.seen_indices = [[] for _ in range(max(cluster_labels) + 1)]

        for i, l in enumerate(cluster_labels):
            if l != -1:
                self.seen_indices[l].append(i)
                self.n_seen_utterances += 1
            else:
                self.unseen_indices.append(i)

        self.encodings = encodings
        self.slot_labels = slot_labels

        self.positive_bound = self.n_seen_utterances
        self.negative_from_seen_bound = self.positive_bound + self.n_seen_utterances * us_negative_sampling_rate_from_seen
        self.negative_from_unseen_bound = self.negative_from_seen_bound + int(
            len(self.unseen_indices) * us_negative_sampling_rate_from_unseen)

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
                return c[idx], random.choice(self.unseen_indices)

    def __getitem__(self, idx):
        item = {}
        if idx < self.positive_bound:
            idx, idx2 = self._get_positive_pair(idx)
            item['us_labels'] = torch.tensor(1.0)
        elif idx < self.negative_from_seen_bound:
            idx, idx2 = self._get_negative_pair_from_seen(idx % self.n_seen_utterances)
            item['us_labels'] = torch.tensor(-1.0)
        else:
            idx, idx2 = self._get_negative_pair_from_unseen()
            item['us_labels'] = torch.tensor(-1.0)

        for key, val in self.encodings.items():
            item[key] = torch.stack((val[idx].clone().detach(), val[idx2].clone().detach()))

        item['smc_labels'] = torch.tensor(self.slot_labels[idx])
        item['smc_labels_2'] = torch.tensor(self.slot_labels[idx2])

        return item

    def __len__(self):
        return self.negative_from_unseen_bound


class JointUtteranceSimilarityAndSlotClassificationModel(nn.Module):
    def __init__(self, base_model, smc_num_labels, us_loss_weight=0.5, smc_loss_weight=0.5,
                 us_negative_to_positive_rate=3):
        super().__init__()
        self.base_model = base_model
        self.us_loss_weight = us_loss_weight
        self.smc_loss_weight = smc_loss_weight

        self.us_loss_fct = MSELoss()

        self.smc_classifier = ClassificationHead(base_model.config, smc_num_labels)
        self.smc_loss_fct = BCEWithLogitsLoss()

        self.us_negative_to_positive_rate = us_negative_to_positive_rate

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            us_labels=None,
            smc_labels=None,
            smc_labels_2=None,
    ):
        input_ids_0, attention_mask_0 = input_ids[:, 0, :], attention_mask[:, 0, :]
        input_ids_1, attention_mask_1 = input_ids[:, 1, :], attention_mask[:, 1, :]

        output_0 = self.base_model(input_ids_0, attention_mask=attention_mask_0)
        mean_0 = _mean_pooling(output_0, attention_mask_0)
        mean_1 = _mean_pooling(self.base_model(input_ids_1, attention_mask=attention_mask_1), attention_mask_1)

        us_loss = self.us_loss_fct(cosine_similarity(mean_0, mean_1), us_labels)

        # smc_logits = self.smc_classifier(cls=_cls(output_0))
        smc_logits = self.smc_classifier(mean=mean_0)
        smc_logits_2 = self.smc_classifier(mean=mean_1)

        smc_loss = torch.add(
            torch.mul(self.smc_loss_fct(smc_logits, smc_labels),
                      1 / (2 + self.us_negative_to_positive_rate)),
            torch.mul(self.smc_loss_fct(smc_logits_2, smc_labels_2),
                      (1 + self.us_negative_to_positive_rate) / (2 + self.us_negative_to_positive_rate)))

        total_loss = torch.add(torch.mul(us_loss, self.us_loss_weight), torch.mul(smc_loss, self.smc_loss_weight))
        return (total_loss,)


# train_cluster_labels is None means unseen cluster
def fine_tune_joint_slot_multiclass_classification_and_utterance_similarity(
        train_texts, train_slots, train_cluster_labels,
        us_loss_weight=0.5, smc_loss_weight=0.5,
        n_train_epochs=None, n_train_steps=None,
        us_negative_sampling_rate_from_seen=3, us_negative_sampling_rate_from_unseen=0.0,
        eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0
):
    # Prepare for slot multiclass classification
    unique_tags = dict.fromkeys(tag for doc in [s for s in train_slots if s is not None] for tag in doc.keys())

    # train_slot_labels = [None if u_slots is None else [1.0 if s in u_slots else 0.0 for s in unique_tags] for u_slots in
    #                      train_slots]

    train_slot_labels = [[1.0 if s in u_slots else 0.0 for s in unique_tags] for u_slots in train_slots]

    # Prepare for utterance similarity
    label_set = dict.fromkeys(train_cluster_labels)
    label_map = {None: -1}
    label_count = 0
    for l in label_set:
        if l is not None:
            label_map[l] = label_count
            label_count += 1

    train_cluster_labels = [label_map[l] for l in train_cluster_labels]
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')

    train_dataset = JointUtteranceSimilarityAndSlotClassificationDataset(
        train_encodings, train_cluster_labels, train_slot_labels,
        us_negative_sampling_rate_from_seen=us_negative_sampling_rate_from_seen,
        us_negative_sampling_rate_from_unseen=us_negative_sampling_rate_from_unseen
    )

    estimator = JointUtteranceSimilarityAndSlotClassificationModel(
        model, len(unique_tags), us_loss_weight=us_loss_weight, smc_loss_weight=smc_loss_weight)

    _finetune_model(estimator, train_dataset, n_train_epochs=n_train_epochs, n_train_steps=n_train_steps,
                    eval_callback=eval_callback,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience)


class JointUtteranceSimilarityAndSlotClassificationAndIntentClassificationDataset(torch.utils.data.Dataset):
    # cluster_labels \in {-1,0,1,2,3,...}, -1 means unseen
    # slot_labels: tuple[Literal[0.0,1.0]] or None for unseen
    # intent_labels: \in {0,1,2,3,...}, None for unseen
    def __init__(self, encodings, cluster_labels, slot_labels, intent_labels,
                 us_negative_sampling_rate_from_seen: int = 3,
                 us_negative_sampling_rate_from_unseen: float = 0.0):
        self.unseen_indices = []
        self.n_seen_utterances = 0
        self.seen_indices = [[] for _ in range(max(cluster_labels) + 1)]

        for i, l in enumerate(cluster_labels):
            if l != -1:
                self.seen_indices[l].append(i)
                self.n_seen_utterances += 1
            else:
                self.unseen_indices.append(i)

        self.encodings = encodings
        self.slot_labels = slot_labels
        self.intent_labels = intent_labels

        self.positive_bound = self.n_seen_utterances
        self.negative_from_seen_bound = self.positive_bound + self.n_seen_utterances * us_negative_sampling_rate_from_seen
        self.negative_from_unseen_bound = self.negative_from_seen_bound + int(
            len(self.unseen_indices) * us_negative_sampling_rate_from_unseen)

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
                return c[idx], random.choice(self.unseen_indices)

    def __getitem__(self, idx):
        item = {}
        if idx < self.positive_bound:
            idx, idx2 = self._get_positive_pair(idx)
            item['us_labels'] = torch.tensor(1.0)
        elif idx < self.negative_from_seen_bound:
            idx, idx2 = self._get_negative_pair_from_seen(idx % self.n_seen_utterances)
            item['us_labels'] = torch.tensor(-1.0)
        else:
            idx, idx2 = self._get_negative_pair_from_unseen()
            item['us_labels'] = torch.tensor(-1.0)

        for key, val in self.encodings.items():
            item[key] = torch.stack((val[idx].clone().detach(), val[idx2].clone().detach()))

        item['smc_labels'] = torch.tensor(self.slot_labels[idx])
        item['smc_labels_2'] = torch.tensor(self.slot_labels[idx2])
        item['ic_labels'] = torch.tensor(self.intent_labels[idx])
        item['ic_labels_2'] = torch.tensor(self.intent_labels[idx2])

        return item

    def __len__(self):
        return self.negative_from_unseen_bound


class JointUtteranceSimilarityAndSlotClassificationAndIntentClassificationModel(nn.Module):
    def __init__(self, base_model, smc_num_labels, ic_num_labels,
                 us_loss_weight=0.4, smc_loss_weight=0.4, ic_loss_weight=0.2,
                 us_negative_to_positive_rate=3):
        super().__init__()
        self.base_model = base_model
        self.us_loss_weight = us_loss_weight
        self.smc_loss_weight = smc_loss_weight
        self.ic_loss_weight = ic_loss_weight

        self.us_loss_fct = MSELoss()

        self.smc_classifier = ClassificationHead(base_model.config, smc_num_labels)
        self.smc_loss_fct = BCEWithLogitsLoss()

        self.ic_classifier = ClassificationHead(base_model.config, ic_num_labels)
        self.ic_loss_fct = CrossEntropyLoss()

        self.us_negative_to_positive_rate = us_negative_to_positive_rate

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            us_labels=None,
            smc_labels=None,
            smc_labels_2=None,
            ic_labels=None,
            ic_labels_2=None,
    ):
        input_ids_0, attention_mask_0 = input_ids[:, 0, :], attention_mask[:, 0, :]
        input_ids_1, attention_mask_1 = input_ids[:, 1, :], attention_mask[:, 1, :]

        output_0 = self.base_model(input_ids_0, attention_mask=attention_mask_0)
        mean_0 = _mean_pooling(output_0, attention_mask_0)
        mean_1 = _mean_pooling(self.base_model(input_ids_1, attention_mask=attention_mask_1), attention_mask_1)

        us_loss = self.us_loss_fct(cosine_similarity(mean_0, mean_1), us_labels)

        # smc_logits = self.smc_classifier(cls=_cls(output_0))
        smc_logits = self.smc_classifier(mean=mean_0)
        smc_logits_2 = self.smc_classifier(mean=mean_1)
        smc_loss = torch.add(
            torch.mul(self.smc_loss_fct(smc_logits, smc_labels),
                      1 / (2 + self.us_negative_to_positive_rate)),
            torch.mul(self.smc_loss_fct(smc_logits_2, smc_labels_2),
                      (1 + self.us_negative_to_positive_rate) / (2 + self.us_negative_to_positive_rate)))

        ic_logits = self.ic_classifier(mean=mean_0)
        ic_logits_2 = self.ic_classifier(mean=mean_1)
        ic_loss = torch.add(
            torch.mul(self.ic_loss_fct(ic_logits, ic_labels),
                      1 / (2 + self.us_negative_to_positive_rate)),
            torch.mul(self.ic_loss_fct(ic_logits_2, ic_labels_2),
                      (1 + self.us_negative_to_positive_rate) / (2 + self.us_negative_to_positive_rate)))

        total_loss = torch.sum(torch.stack([torch.mul(us_loss, self.us_loss_weight),
                                            torch.mul(smc_loss, self.smc_loss_weight),
                                            torch.mul(ic_loss, self.ic_loss_weight)]))
        return (total_loss,)


def fine_tune_joint_slot_multiclass_classification_and_utterance_similarity_and_intent_classification(
        train_texts, train_slots, train_cluster_labels, train_intents,
        us_loss_weight=0.4, smc_loss_weight=0.4, ic_loss_weight=0.2,
        n_train_epochs=None, n_train_steps=None,
        us_negative_sampling_rate_from_seen=3, us_negative_sampling_rate_from_unseen=0.0,
        eval_callback: Callable[..., float] = None, early_stopping=False, early_stopping_patience=0
):
    # Prepare for slot multiclass classification
    unique_tags = dict.fromkeys(tag for doc in [s for s in train_slots if s is not None] for tag in doc.keys())

    train_slot_labels = [None if u_slots is None else [1.0 if s in u_slots else 0.0 for s in unique_tags] for u_slots in
                         train_slots]

    # Prepare for utterance similarity
    label_set = dict.fromkeys(train_cluster_labels)
    label_map = {None: -1}
    label_count = 0
    for l in label_set:
        if l is not None:
            label_map[l] = label_count
            label_count += 1

    train_cluster_labels = [label_map[l] for l in train_cluster_labels]
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')

    # Prepare for intent classification
    unique_intents = dict.fromkeys(i for i in train_intents if i is not None)
    unique_intents_map = {l: i for i, l in enumerate(unique_intents)}
    train_intent_labels = [None if i is None else unique_intents_map[i] for i in train_intents]

    train_dataset = JointUtteranceSimilarityAndSlotClassificationAndIntentClassificationDataset(
        train_encodings, train_cluster_labels, train_slot_labels, train_intent_labels,
        us_negative_sampling_rate_from_seen=us_negative_sampling_rate_from_seen,
        us_negative_sampling_rate_from_unseen=us_negative_sampling_rate_from_unseen
    )

    estimator = JointUtteranceSimilarityAndSlotClassificationAndIntentClassificationModel(
        model, len(unique_tags), len(unique_intents),
        us_loss_weight=us_loss_weight, smc_loss_weight=smc_loss_weight, ic_loss_weight=ic_loss_weight)

    _finetune_model(estimator, train_dataset, n_train_epochs=n_train_epochs, n_train_steps=n_train_steps,
                    eval_callback=eval_callback,
                    early_stopping=early_stopping,
                    early_stopping_patience=early_stopping_patience)


class JointClassificationPseudoAndIntentClassificationDataset(torch.utils.data.Dataset):
    # labels \in {0,1,2,3,...} (for single-class) or tuple[Literal[0.0,1.0]] (for multi-class)
    def __init__(self, encodings, pseudo_labels, intent_labels, pseudo_sample_weights=None):
        self.encodings = encodings
        self.pseudo_labels = pseudo_labels
        self.intent_labels = intent_labels
        self.pseudo_sample_weights = pseudo_sample_weights

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['pseudo_labels'] = torch.tensor(self.pseudo_labels[idx])
        item['intent_labels'] = torch.tensor(self.intent_labels[idx])
        if self.pseudo_sample_weights is not None:
            item['pseudo_sample_weights'] = torch.tensor(self.pseudo_sample_weights[idx])
        return item

    def __len__(self):
        return len(self.pseudo_labels)


class JointPseudoClassificationAndIntentClassificationModel(nn.Module):
    def __init__(self, base_model, num_pseudo_labels, num_intent_labels, intent_classifier_weight=0.1):
        super().__init__()
        self.config = base_model.config
        self.base_model = base_model
        self.pseudo_classifier = ClassificationHead(base_model.config, num_pseudo_labels)
        self.pseudo_loss_fct = CrossEntropyLoss(reduction='none')

        self.intent_classifier = ClassificationHead(base_model.config, num_intent_labels)
        self.intent_loss_fct = CrossEntropyLoss()
        self.intent_classifier_weight = intent_classifier_weight

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            pseudo_labels=None,
            pseudo_sample_weights=None,
            intent_labels=None
    ):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
        )
        mean = _mean_pooling(outputs, attention_mask)

        pc_logits = self.pseudo_classifier(mean=mean)
        per_sample_loss = self.pseudo_loss_fct(pc_logits, pseudo_labels)
        if pseudo_sample_weights is not None:
            per_sample_loss = torch.mul(per_sample_loss, pseudo_sample_weights)
        pc_loss = torch.mean(per_sample_loss)

        ic_logits = self.intent_classifier(mean=mean)
        ic_loss = self.intent_loss_fct(ic_logits, intent_labels)

        total_loss = torch.add(torch.mul(pc_loss, 1 - self.intent_classifier_weight),
                               torch.mul(ic_loss, self.intent_classifier_weight))

        return (total_loss,)


def fine_tune_joint_pseudo_classification_and_intent_classification(
        train_texts, train_cluster_ids, train_intent_ids, train_sample_weights=None,
        intent_classifier_weight=0.1, previous_classifier=None, previous_optim=None):
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    train_dataset = JointClassificationPseudoAndIntentClassificationDataset(train_encodings, train_cluster_ids,
                                                                            train_intent_ids,
                                                                            pseudo_sample_weights=train_sample_weights)

    if previous_classifier is None:
        classifier = JointPseudoClassificationAndIntentClassificationModel(model, len(dict.fromkeys(train_cluster_ids)),
                                                                           len(dict.fromkeys(train_intent_ids)),
                                                                           intent_classifier_weight=intent_classifier_weight)
        classifier.to(device)
    else:
        classifier = previous_classifier

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    optim = AdamW(classifier.parameters(), lr=5e-5, weight_decay=0.01) if previous_optim is None else previous_optim

    classifier.train()  # Switch mode
    cur = 0
    for batch in train_loader:
        cur += 1
        print('\rNum examples: {} Batch: {}/{} ({:.1f}%)'
              .format(len(train_dataset), cur, len(train_loader), 100 * cur / len(train_loader)),
              end='' if cur < len(train_loader) else '\n')
        optim.zero_grad()
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = classifier(**batch)
        loss = outputs[0]
        loss.backward()
        optim.step()
    classifier.eval()

    return classifier, optim


if __name__ == '__main__':
    set_seed(12993)
    load()

    # train_texts, train_labels, train_labels_weights \
    #     = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 0, 1, 0, 1, 0, 1, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # fine_tune_pseudo_classification(train_texts, train_labels, train_sample_weights=train_labels_weights)

    # train_texts, train_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 0, 1, 0, 1, 0, None, None]
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
    #        {'slot_1': {'start': 24, 'end': 49}, 'slot_2': {'start': 0, 'end': 4}}, {}], \
    #       [0, 1, None]
    # fine_tune_joint_slot_multiclass_classification_and_utterance_similarity(train_texts, train_slots, train_labels)
