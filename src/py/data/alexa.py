import json
import os
import random
from math import ceil

from transformers import set_seed

from data.entity import Utterance


def _parse_text_and_slots(utr):
    tokens = []
    for t in utr.strip().split(' '):
        pos = t.find('|')
        assert pos >= 0
        tokens.append((t[:pos], t[pos + 1:]))

    text = ' '.join([t[0] for t in tokens])
    slot = {}
    for i in range(len(tokens)):
        if tokens[i][1] != 'Other' and (i == 0 or tokens[i - 1][1] != tokens[i][1]):
            j = i
            while j + 1 < len(tokens) and tokens[j + 1][1] == tokens[j][1]:
                j += 1
            slot[tokens[i][1]] = {
                'value': ' '.join([t[0] for t in tokens[i:j + 1]]),
                'start': sum([len(t[0]) for t in tokens[:i]]) + i,
                'end': sum([len(t[0]) for t in tokens[:j + 1]]) + j,
            }
    return text, slot


def read_feature_data(folder='./data/alexa/fr/frXX_feature_data/'):
    utterances = []
    uniq_utrs = set()
    for f in [f for f in os.listdir(folder)]:
        if not os.path.isdir(os.path.join(folder, f)):
            continue
        for line in open(os.path.join(folder, f, 'feature_data_uniq.tsv')):
            parts = line.strip().split('\t')

            ucheck = '\t'.join(parts[0:3])
            if ucheck in uniq_utrs:
                continue
            else:
                uniq_utrs.add(ucheck)

            u = Utterance()

            u.domain, u.intent_name = parts[0], parts[1]
            u.feature_name = f
            u.text, u.slots = _parse_text_and_slots(parts[2])

            utterances.append(u)

    return utterances


def get_train_test_data(generate_data=False):
    train_test_data_file = './data/alexa/fr/slot_based_clusters/train_test_data_global.json'

    if generate_data:
        inter_intent_data = read_feature_data()

        clusters = dict.fromkeys([u.feature_name for u in inter_intent_data])
        predefined_clusters = {  # TODO: pre-define TRAIN/TEST features?
            'TRAIN': [],
            'TEST': []
        }
        predefined_clusters = None

        if predefined_clusters is not None:
            train_clusters = predefined_clusters['TRAIN']
            test_clusters = predefined_clusters['TEST']
            for c in train_clusters + test_clusters:
                if c not in clusters:
                    raise Exception('Invalid cluster:', c)
        else:
            # Filter clusters with small size
            clusters = [c for c in clusters if sum([1 for u in inter_intent_data if u.feature_name == c]) >= 100]

            # Random train/dev/test clusters
            random.shuffle(clusters)
            train_clusters = clusters[:ceil(len(clusters) * 0.7)]
            test_clusters = clusters[ceil(len(clusters) * 0.7):]

        # TODO: add feature_type

        # Split into train/test utterances
        splitted_data = []
        for c in clusters:
            cluster_data = [u for u in inter_intent_data if u.feature_name == c]
            random.shuffle(cluster_data)
            if len(cluster_data) > 2000:  # Down sample
                cluster_data = cluster_data[0:2000]

            if c in train_clusters:
                for i, u in enumerate(cluster_data):
                    u.feature_name += '_TRAIN'
                    u.part_type = 'TRAIN' if i < ceil(len(cluster_data) * 0.7) else 'TEST'
            elif c in test_clusters:
                for i, u in enumerate(cluster_data):
                    u.feature_name += '_TEST'
                    u.part_type = 'TEST'
            splitted_data.extend(cluster_data)

        inter_intent_data = splitted_data
        json_data = [
            [u.text, u.feature_name, u.part_type, u.slots, u.intent_name, u.domain]
            for u in inter_intent_data
        ]
        os.makedirs(os.path.dirname(train_test_data_file), exist_ok=True)
        with open(train_test_data_file, 'w') as f:
            f.write(json.dumps(json_data))
    else:
        json_data = json.loads(open(train_test_data_file).read())
        inter_intent_data = [
            Utterance(text=u[0], feature_name=u[1], part_type=u[2], slots=u[3], intent_name=u[4], domain=u[5])
            for u in json_data
        ]

    def get_feature_type_for_Alexa(feature_name):
        feature_name_2_type = {
            # TODO
        }

        if feature_name.endswith('_TRAIN'):
            feature_name = feature_name[:-6]
        if feature_name.endswith('_DEV'):
            feature_name = feature_name[:-4]
        if feature_name.endswith('_TEST'):
            feature_name = feature_name[:-5]

        return feature_name_2_type[feature_name]

    print('======== Cluster information ========')
    clusters = dict.fromkeys(u.feature_name for u in inter_intent_data)
    all_clusters = []
    train_clusters = []
    test_clusters = []
    for c in clusters:
        total = len([u for u in inter_intent_data if u.feature_name == c])
        if c.endswith('_TRAIN'):
            all_clusters.append((c[:-6], total))
            train_clusters.append((c, total))
        elif c.endswith('_TEST'):
            all_clusters.append((c[:-5], total))
            test_clusters.append((c, total))

    print('All ({}):'.format(len(all_clusters)), all_clusters)
    print('Train ({}):'.format(len(train_clusters)), train_clusters)
    print('Test ({}):'.format(len(test_clusters)), test_clusters)

    return inter_intent_data


def print_train_dev_test_stats(intent_data):
    train_size = len([u for u in intent_data if u.part_type == 'TRAIN'])
    dev_size = len([u for u in intent_data if u.part_type == 'DEV'])
    test_size = len([u for u in intent_data if u.part_type == 'TEST'])
    print('Train/Dev/Test utterances: {}/{}/{} ({:.1f}%,{:.1f}%,{:.1f}%)'.format(train_size, dev_size, test_size,
                                                                                 100 * train_size / len(intent_data),
                                                                                 100 * dev_size / len(intent_data),
                                                                                 100 * test_size / len(intent_data)))


if __name__ == '__main__':
    set_seed(12993)

    inter_intent_data = get_train_test_data(generate_data=True)
    print_train_dev_test_stats(inter_intent_data)

    # Write utterances to file
    if False:
        f = open('./data/alexa/fr/slot_based_clusters/utterances.txt', 'w')
        intents = dict.fromkeys(u.feature_name[:u.feature_name.find('_')] for u in inter_intent_data)
        for intent in intents:
            data = [u for u in inter_intent_data if u.feature_name.startswith(intent)]
            sets = dict.fromkeys(u.feature_name for u in data)
            print('======== Intent:', intent, '========', file=f)
            sets = [s for s in sets if s.endswith('TRAIN')] + \
                   [s for s in sets if s.endswith('DEV')] + \
                   [s for s in sets if s.endswith('TEST')]
            for s in sets:
                utrs = [u.text for u in data if u.feature_name == s]
                print('Feature:', s[s.find('_') + 1:], file=f)
                print(utrs, file=f)
                print(file=f)
        f.close()
