import numpy
import torch
from datasets import load_metric
from sentence_transformers.losses import MSELoss
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer

tokenizer = None
model = None


def load(model_path):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)


def save(model_path):
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)


# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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
            return ((loss,) + output) if loss is not None else output

        return ClassificationHead(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def fine_tune_classification(train_texts, train_labels, val_texts, val_labels, test_texts=None, test_labels=None):
    label_set = set(train_labels)
    label_set.update(val_labels)
    if test_labels is not None:
        label_set.update(test_labels)
    label_map = {l: i for i, l in enumerate(label_set)}

    train_labels = [label_map[l] for l in train_labels]
    val_labels = [label_map[l] for l in val_labels]
    if test_labels is not None:
        test_labels = [label_map[l] for l in test_labels]

    metric_accuracy = load_metric("accuracy")

    def compute_accuracy(eval_pred):
        logits, labels = eval_pred
        predictions = numpy.argmax(logits, axis=-1)
        return metric_accuracy.compute(predictions=predictions, references=labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True,
                               return_tensors='pt') if test_texts is not None else None

    train_dataset = ClassificationDataset(train_encodings, train_labels)
    val_dataset = ClassificationDataset(val_encodings, val_labels)
    test_dataset = ClassificationDataset(test_encodings, test_labels) if test_texts is not None else None

    classifier = PseudoClassificationModel(model, len(label_set))

    # print(classifier(**train_encodings))

    trainer = Trainer(
        model=classifier,
        args=TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy='epoch'
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


if __name__ == '__main__':
    load('sentence-transformers/paraphrase-mpnet-base-v2')

    print(type(get_embeddings(['This is an example sentence', 'Each sentence is converted'])))
    train_texts, train_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] * 10, [1, 0, 1, 0, 1, 0, 1, 0] * 10
    val_texts, val_labels = ['i', 'j', 'k', 'l'], [1, 0, 1, 0]

    fine_tune_classification(train_texts, train_labels, val_texts, val_labels)
