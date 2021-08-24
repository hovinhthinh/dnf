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
    clustered_ids = set()
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
            else:
                clustered_ids.add(cu[i]['id'])

    # Add an unknown cluster
    unk_cluster = list(filter(lambda u: u['id'] not in clustered_ids, utrs))
    if len(unk_cluster) > 0:
        clusters[intent_name + '_UNK'] = unk_cluster

    print('==== Cluster Size After Removing Overlaps ====')

    f = None
    if output_file is not None:
        f = open(output_file, 'w')

    for k, v in clusters.items():
        print('{}: {}'.format(k, len(v)))
        for u in v:
            if '_UNK' in k:
                # print(u)
                pass
            if f is not None:
                u = u.copy()
                u.pop('id')
                u['intent'] = intent_name
                u['cluster'] = k
                f.write('{}\n'.format(json.dumps(u)))

    if f is not None:
        f.close()

    return clusters


def split_by_features_AddToPlaylist():
    # print_file('data/snips/AddToPlaylist/train_AddToPlaylist_full.json')
    utrs = get_utterances_and_slots('data/snips/AddToPlaylist/train_AddToPlaylist_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    extract_utterances_splitted_by_features(
        {
            'AddASongToAPlaylist':
                lambda u: ('entity_name' in u['slots']) and ('playlist' in u['slots']),

            'AddAnArtistToAPlaylist':
                lambda u: ('artist' in u['slots']) and ('playlist' in u['slots']) and (
                        'music_item' not in u['slots'] or u['slots']['music_item'] == 'artist'),

            'AddCurrentSongToAPlaylist':
                lambda u: ('music_item' in u['slots'] and u['slots']['music_item'] in ['song', 'track', 'tune'])
                          and ('playlist' in u['slots'])
                          and ('artist' not in u['slots'] or ' this ' in u['text'])
                          and (' a ' not in u['text'] and ' an ' not in u['text'] and ' another ' not in u['text']),

            'AddCurrentAlbumToAPlaylist':
                lambda u: (('music_item', 'album') in u['slots'].items()) and ('playlist' in u['slots'])
                          and (' a ' not in u['text'] and ' an ' not in u['text'])
                          and (('artist' not in u['slots']) or ' this ' in u['text']),

            'AddCurrentArtistToAPlaylist':
                lambda u: (('music_item', 'artist') in u['slots'].items()) and ('playlist' in u['slots']) and (
                        'artist' not in u['slots']),

            'AddAnArtistAlbumToAPlaylist':
                lambda u: (('music_item', 'album') in u['slots'].items()) and ('playlist' in u['slots'])
                          and ('artist' in u['slots']) and (' this ' not in u['text'])
        },
        utrs,
        'AddToPlaylist',
        'data/snips/slot_based_clusters/AddToPlaylist.json'
    )


def split_by_features_BookRestaurant():
    # print_file('data/snips/BookRestaurant/train_BookRestaurant_full.json')
    utrs = get_utterances_and_slots('data/snips/BookRestaurant/train_BookRestaurant_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    extract_utterances_splitted_by_features(
        {
            # TODO
        },
        utrs,
        'BookRestaurant',
        'data/snips/slot_based_clusters/BookRestaurant.json'
    )


def split_by_features_GetWeather():
    # print_file('data/snips/GetWeather/train_GetWeather_full.json')
    utrs = get_utterances_and_slots('data/snips/GetWeather/train_GetWeather_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    def GetCurrentWeatherInALocation(u):
        found = False
        good_time_range = True
        for k in u['slots']:
            if k in ['city', 'country', 'state', 'geographic_poi']:
                found = True
            elif k == 'timeRange':
                if u['slots'][k] != 'now':
                    good_time_range = False
            elif k not in ['condition_description', 'condition_temperature', 'spatial_relation']:
                return False
        return found and good_time_range

    def GetCurrentWeatherInCurrentPosition(u):
        slots = u['slots']
        # if 'condition_description' in slots or 'condition_temperature' in slots:
        #     return False
        if 'current_location' not in slots:
            for k in u['slots']:
                if k in ['city', 'country', 'state', 'geographic_poi']:
                    return False
        if 'timeRange' in slots and slots['timeRange'] != 'now':
            return False
        return True

    def GetWeatherInALocationAtATimeRange(u):
        found = False
        good_time_range = False
        for k in u['slots']:
            if k in ['city', 'country', 'state', 'geographic_poi']:
                found = True
            elif k == 'timeRange' and u['slots'][k] != 'now':
                good_time_range = True
            elif k not in ['condition_description', 'condition_temperature', 'spatial_relation']:
                return False
        return found and good_time_range

    def GetWeatherInCurrentPositionAtATimeRange(u):
        slots = u['slots']
        # if 'condition_description' in slots or 'condition_temperature' in slots:
        #     return False
        if 'current_location' not in slots:
            for k in u['slots']:
                if k in ['city', 'country', 'state', 'geographic_poi']:
                    return False
        if 'timeRange' not in slots or slots['timeRange'] == 'now':
            return False
        return True

    extract_utterances_splitted_by_features(
        {
            'GetCurrentWeatherInALocation': lambda u: GetCurrentWeatherInALocation(u),
            'GetCurrentWeatherInCurrentPosition': lambda u: GetCurrentWeatherInCurrentPosition(u),
            'GetWeatherInALocationAtATimeRange': lambda u: GetWeatherInALocationAtATimeRange(u),
            'GetWeatherInCurrentPositionAtATimeRange': lambda u: GetWeatherInCurrentPositionAtATimeRange(u),
        },
        utrs,
        'GetWeather',
        'data/snips/slot_based_clusters/GetWeather.json'
    )


def split_by_features_RateBook():
    # print_file('data/snips/RateBook/train_RateBook_full.json')
    utrs = get_utterances_and_slots('data/snips/RateBook/train_RateBook_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1

    def RateCurrentBook(u):
        if ('rating_value' not in u['slots']) or ('object_select' not in u['slots']) or (
                u['slots']['object_select'] not in ['this', 'current', 'this current']):
            return False
        otype = u['slots'].get('object_type', None) or u['slots'].get('object_part_of_series_type', None)
        return otype in ['textbook', 'essay', 'novel', 'book', 'album', 'series', 'saga', 'chronicle', 'essay book',
                         'book novel', 'book album', 'series chronicle']

    def RatePreviousBook(u):
        if ('rating_value' not in u['slots']) or ('object_select' not in u['slots']) or (
                u['slots']['object_select'] not in ['previous', 'last']):
            return False
        otype = u['slots'].get('object_type', None) or u['slots'].get('object_part_of_series_type', None)
        return otype in ['textbook', 'essay', 'novel', 'book', 'album', 'series', 'saga', 'chronicle', 'essay book',
                         'book novel', 'book album', 'series chronicle']

    def RateNextBook(u):
        if ('rating_value' not in u['slots']) or ('object_select' not in u['slots']) or (
                u['slots']['object_select'] not in ['next']):
            return False
        otype = u['slots'].get('object_type', None) or u['slots'].get('object_part_of_series_type', None)
        return otype in ['textbook', 'essay', 'novel', 'book', 'album', 'series', 'saga', 'chronicle', 'essay book',
                         'book novel', 'book album', 'series chronicle']

    extract_utterances_splitted_by_features(
        {
            'RateABook': lambda u: 'object_name' in u['slots'] and 'rating_value' in u['slots'],
            'RateCurrentBook': lambda u: RateCurrentBook(u),
            'RatePreviousBook': lambda u: RatePreviousBook(u),
            'RateNextBook': lambda u: RateNextBook(u)
        },
        utrs,
        'RateBook',
        'data/snips/slot_based_clusters/RateBook.json'
    )


if __name__ == '__main__':
    # print_file('data/snips/PlayMusic/train_PlayMusic_full.json')
    # print_file('data/snips/SearchCreativeWork/train_SearchCreativeWork_full.json')
    # print_file('data/snips/SearchScreeningEvent/train_SearchScreeningEvent_full.json')

    # split_by_features_AddToPlaylist()
    # split_by_features_BookRestaurant()
    # split_by_features_GetWeather()
    # split_by_features_RateBook()

    pass
