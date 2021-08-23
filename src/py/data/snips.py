import json
import random


def get_utterances(input_file):
    data = json.loads(open(input_file, 'r').read())

    data = list(data.values())[0]
    random.shuffle(data)

    utterances = []
    for v in data:
        utterances.append(''.join([t['text'] for t in v['data']]).strip().lower())

    return utterances


def get_all_utterances_and_labels():
    """
    :return: list[tuple[utterance, cluster_index, is_train]]
    """
    input_files = ['data/snips/AddToPlaylist/train_AddToPlaylist_full.json',
                   'data/snips/BookRestaurant/train_BookRestaurant_full.json',
                   'data/snips/GetWeather/train_GetWeather_full.json',
                   'data/snips/PlayMusic/train_PlayMusic_full.json',
                   'data/snips/RateBook/train_RateBook_full.json',
                   'data/snips/SearchCreativeWork/train_SearchCreativeWork_full.json',
                   'data/snips/SearchCreativeWork/validate_SearchCreativeWork.json']

    utterances_and_labels = []
    for i, inp in enumerate(input_files):
        utterances_and_labels.extend([(u, i, False) for u in get_utterances(inp)])
    return utterances_and_labels


def print_file(input_file):
    data = json.loads(open(input_file, 'r').read())

    slot_count = {}

    data = list(data.values())[0]
    random.shuffle(data)
    for v in data:
        text = ''.join([t['text'] for t in v['data']]).strip().lower()
        slots = ' | '.join(
            ['{}:{}'.format(t['entity'], t['text']) for t in filter(lambda x: 'entity' in x, v['data'])]).strip()

        for t in filter(lambda x: 'entity' in x, v['data']):
            slot_count[t['entity']] = slot_count.get(t['entity'], 0) + 1

        print(text, '|', slots)

    print(sorted(slot_count.items(), key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    print_file('data/snips/AddToPlaylist/train_AddToPlaylist_full.json')
    # print_file('data/snips/BookRestaurant/train_BookRestaurant_full.json')
    # print_file('data/snips/GetWeather/train_GetWeather_full.json')
    # print_file('data/snips/PlayMusic/train_PlayMusic_full.json')
    # print_file('data/snips/RateBook/train_RateBook_full.json')
    # print_file('data/snips/SearchCreativeWork/train_SearchCreativeWork_full.json')
    # print_file('data/snips/SearchCreativeWork/validate_SearchCreativeWork.json')
