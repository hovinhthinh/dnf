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


def get_utterances_and_slots(input_file):
    """
    :param input_file:
    :return: list[dict]
    """
    data = json.loads(open(input_file, 'r').read())

    utrs = []

    data = list(data.values())[0]
    for i, v in enumerate(data):
        utrs.append({
            'id': i,
            'text': ''.join([t['text'] for t in v['data']]).strip().lower(),
            'slots': {t['entity']: t['text'] for t in filter(lambda x: 'entity' in x, v['data'])}
        })

    return utrs


def print_file(input_file):
    slot_count = {}
    for u in get_utterances_and_slots(input_file):
        print(u)
        for k in u['slots']:
            slot_count[k] = slot_count.get(k, 0) + 1

    print(sorted(slot_count.items(), key=lambda x: x[1], reverse=True))


def extract_utterances_splitted_by_features(filters: dict, utrs, intent_name, output_file=None):
    clusters = {}
    for n, f in filters.items():
        clusters[n] = list(filter(f, utrs))

    # Print stats:
    print('==== Cluster Size ====')
    for k, v in clusters.items():
        print('{}: {}'.format(k, len(v)))
    print('==== Overlap ====')

    overlap_ids = set()

    for u, cu in clusters.items():
        for v, cv in clusters.items():
            if u >= v:
                continue
            overlap = [x for x in cu if x in cv]
            print('{} & {}: {}'.format(u, v, len(overlap)))

            for x in overlap:
                overlap_ids.add(x['id'])
    for u, cu in clusters.items():
        for i in reversed(range(len(cu))):
            if cu[i]['id'] in overlap_ids:
                cu.pop(i)

    print('==== Cluster Size After Removing Overlaps ====')

    f = None
    if output_file is not None:
        f = open(output_file, 'w')

    for k, v in clusters.items():
        print('{}: {}'.format(k, len(v)))
        if f is not None:
            for u in v:
                u = u.copy()
                u.pop('id')
                u['intent'] = intent_name
                u['cluster'] = k
                f.write('{}\n'.format(json.dumps(u)))

    if f is not None:
        f.close()

    return clusters


if __name__ == '__main__':
    print_file('data/snips/AddToPlaylist/train_AddToPlaylist_full.json')
    # print_file('data/snips/BookRestaurant/train_BookRestaurant_full.json')
    # print_file('data/snips/GetWeather/train_GetWeather_full.json')
    # print_file('data/snips/PlayMusic/train_PlayMusic_full.json')
    # print_file('data/snips/RateBook/train_RateBook_full.json')
    # print_file('data/snips/SearchCreativeWork/train_SearchCreativeWork_full.json')
    # print_file('data/snips/SearchCreativeWork/validate_SearchCreativeWork.json')

    # AddToPlaylist
    utrs = get_utterances_and_slots('data/snips/AddToPlaylist/train_AddToPlaylist_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[0], reverse=True))

    extract_utterances_splitted_by_features(
        {
            'AddASpecificSongToASpecificPlaylist':
                lambda u: ('entity_name' in u['slots']) and ('playlist' in u['slots']),

            'AddASpecificArtistToASpecificPlaylist':
                lambda u: ('artist' in u['slots']) and ('playlist' in u['slots']) and ('music_item' not in u['slots']),

            'AddCurrentSongToASpecificPlaylist':
                lambda u: ('music_item' in u['slots'] and u['slots']['music_item'] in ['song', 'track', 'tune'])
                          and ('playlist' in u['slots']) and ('artist' not in u['slots']),

            'AddCurrentAlbumToASpecificPlaylist':
                lambda u: (('music_item', 'album') in u['slots'].items()) and ('playlist' in u['slots'])
                          and (' a ' not in u['text'] and ' an ' not in u['text'])
                          and (('artist' not in u['slots']) or ' this ' in u['text']),

            'AddAnArtistAlbumToASpecificPlaylist':
                lambda u: (('music_item', 'album') in u['slots'].items()) and ('playlist' in u['slots'])
                          and ('artist' in u['slots']) and (' this ' not in u['text'])
        },
        utrs,
        'AddToPlaylist',
        'data/snips/slot_based_clusters/AddToPlaylist.json'
    )
