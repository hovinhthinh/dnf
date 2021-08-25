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
                # print(x)
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
        clusters['UNK'] = unk_cluster

    print('==== Cluster Size After Removing Overlaps ====')

    f = None
    if output_file is not None:
        f = open(output_file, 'w')
        f.write('[\n')

    n_printed = 0
    for k, v in clusters.items():
        print('{}: {}'.format(k, len(v)))
        for u in v:
            if k == 'UNK':
                # print(u)
                pass
            if f is not None:
                if n_printed > 0:
                    f.write(',\n')
                u = u.copy()
                u.pop('id')
                u['intent'] = intent_name
                u['cluster'] = k
                f.write('{}'.format(json.dumps(u)))
            n_printed += 1

    if f is not None:
        f.write('\n]')
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
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

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


def split_by_features_PlayMusic():
    # print_file('data/snips/PlayMusic/train_PlayMusic_full.json')
    utrs = get_utterances_and_slots('data/snips/PlayMusic/train_PlayMusic_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    def PlayOnService(u):
        found = False
        for k in u['slots']:
            if k == 'service':
                found = True
            elif k not in ['music_item', 'sort']:
                return False
        return found

    def PlayTrack(u):
        found = False
        for k in u['slots']:
            if k == 'track':
                found = True
            elif k == 'music_item':
                if u['slots'][k] not in ['song', 'track', 'tune', 'soundtrack', 'sound track', 'record']:
                    return False
            elif k not in ['artist']:
                return False
        return found

    def PlayTrackOnService(u):
        found = False
        service_found = False
        for k in u['slots']:
            if k == 'track':
                found = True
            elif k == 'music_item':
                if u['slots'][k] not in ['song', 'track']:
                    return False
            elif k == 'service':
                service_found = True
            elif k not in ['artist']:
                return False
        return found and service_found

    def PlayAlbum(u):
        found = False
        for k in u['slots']:
            if k == 'album':
                found = True
            elif k == 'music_item':
                if u['slots'][k] not in ['album', 'song', 'track']:
                    return False
            elif k not in ['artist', 'sort']:
                return False
        return found

    def PlayAlbumOnService(u):
        found = False
        service_found = False
        for k in u['slots']:
            if k == 'album':
                found = True
            elif k == 'music_item':
                if u['slots'][k] not in ['album', 'song', 'track']:
                    return False
            elif k == 'service':
                service_found = True
            elif k not in ['artist', 'sort']:
                return False
        return found and service_found

    def PlayMusicByArtist(u):
        found = False
        for k in u['slots']:
            if k == 'artist':
                found = True
            elif k not in ['music_item', 'sort']:
                return False
        return found

    def PlayMusicByArtistOnService(u):
        found = False
        service_found = False
        for k in u['slots']:
            if k == 'artist':
                found = True
            elif k == 'service':
                service_found = True
            elif k not in ['music_item', 'sort']:
                return False
        return found and service_found

    def PlayMusicByYear(u):
        found = False
        for k in u['slots']:
            if k == 'year':
                found = True
            elif k not in ['music_item', 'sort']:
                return False
        return found

    def PlayMusicByYearOnService(u):
        found = False
        service_found = False
        for k in u['slots']:
            if k == 'year':
                found = True
            elif k == 'service':
                service_found = True
            elif k not in ['music_item', 'sort']:
                return False
        return found and service_found

    def PlayMusicByArtistAndYear(u):
        artist_found = False
        year_found = False
        for k in u['slots']:
            if k == 'artist':
                artist_found = True
            elif k == 'year':
                year_found = True
            elif k not in ['music_item', 'sort']:
                return False
        return year_found and artist_found

    def PlayMusicByArtistAndYearOnService(u):
        artist_found = False
        year_found = False
        service_found = False
        for k in u['slots']:
            if k == 'artist':
                artist_found = True
            elif k == 'year':
                year_found = True
            elif k == 'service':
                service_found = True
            elif k not in ['music_item', 'sort']:
                return False
        return year_found and artist_found and service_found

    def PlayMusicByGenre(u):
        found = False
        for k in u['slots']:
            if k == 'genre':
                found = True
            elif k == 'music_item':
                if u['slots'][k] not in ['track', 'song']:
                    return False
            elif k not in ['sort']:
                return False
        return found

    def PlayMusicByGenreOnService(u):
        found = False
        service_found = False
        for k in u['slots']:
            if k == 'genre':
                found = True
            elif k == 'music_item':
                if u['slots'][k] not in ['track', 'song']:
                    return False
            elif k == 'service':
                service_found = True
            elif k not in ['sort']:
                return False
        return found and service_found

    extract_utterances_splitted_by_features(
        {
            'PlayOnService': lambda u: PlayOnService(u),
            'PlayTrack': lambda u: PlayTrack(u),
            'PlayTrackOnService': lambda u: PlayTrackOnService(u),
            'PlayAlbum': lambda u: PlayAlbum(u),
            'PlayAlbumOnService': lambda u: PlayAlbumOnService(u),
            'PlayMusicByArtist': lambda u: PlayMusicByArtist(u),
            'PlayMusicByArtistOnService': lambda u: PlayMusicByArtistOnService(u),
            'PlayPlaylist': lambda u: ('playlist' in u['slots']) and ('service' not in u['slots']),
            'PlayPlaylistOnService': lambda u: ('playlist' in u['slots']) and ('service' in u['slots']),
            'PlayMusicByYear': lambda u: PlayMusicByYear(u),
            'PlayMusicByYearOnService': lambda u: PlayMusicByYearOnService(u),
            'PlayMusicByGenre': lambda u: PlayMusicByGenre(u),
            'PlayMusicByGenreOnService': lambda u: PlayMusicByGenreOnService(u),
            'PlayMusicByArtistAndYear': lambda u: PlayMusicByArtistAndYear(u),
            'PlayMusicByArtistAndYearOnService': lambda u: PlayMusicByArtistAndYearOnService(u),
        },
        utrs,
        'PlayMusic',
        'data/snips/slot_based_clusters/PlayMusic.json'
    )


# Split by slot values
def split_by_features_SearchCreativeWork():
    slot_count = {}
    for u in get_utterances_and_slots('data/snips/SearchCreativeWork/train_SearchCreativeWork_full.json'):
        for k, v in u['slots'].items():
            p = k + '|' + v if k == 'object_type' else k
            slot_count[p] = slot_count.get(p, 0) + 1
    print(sorted(slot_count.items(), key=lambda x: x[1], reverse=True))

    utrs = get_utterances_and_slots('data/snips/SearchCreativeWork/train_SearchCreativeWork_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    def is_wh_question(u):
        text = u['text'].lower()
        return text.startswith('what ') or text.startswith('where ') or text.startswith('when ') or text.startswith(
            'who ') or text.startswith('whom ') or text.startswith('which ') or text.startswith(
            'whose ') or text.startswith('why ') or text.startswith('how ')

    extract_utterances_splitted_by_features(
        {
            'SearchCreativeWork': lambda u: ('object_type' not in u['slots']) and not is_wh_question(u),
            'PlayTVProgram': lambda u: ('object_type' in u['slots'])
                                       and (u['slots']['object_type'] in ['TV show', 'show', 'TV series',
                                                                          'television show', 'movie', 'program',
                                                                          'saga'])
                                       and not is_wh_question(u),
            'PlayTVProgramTrailer': lambda u: ('object_type' in u['slots'])
                                              and (u['slots']['object_type'] in ['trailer'])
                                              and not is_wh_question(u),
            'PlaySong': lambda u: ('object_type' in u['slots'])
                                  and (u['slots']['object_type'] in ['song', 'soundtrack'])
                                  and not is_wh_question(u),
            'SearchAlbum': lambda u: ('object_type' in u['slots'])
                                     and (u['slots']['object_type'] in ['album'])
                                     and not is_wh_question(u),
            'SearchBook': lambda u: ('object_type' in u['slots'])
                                    and (u['slots']['object_type'] in ['book', 'novel'])
                                    and not is_wh_question(u),
            'SearchPicture': lambda u: ('object_type' in u['slots'])
                                       and (u['slots']['object_type'] in ['picture', 'photograph', 'painting'])
                                       and not is_wh_question(u),
            'SearchGame': lambda u: ('object_type' in u['slots'])
                                    and (u['slots']['object_type'] in ['game', 'video game'])
                                    and not is_wh_question(u),
            'WHQuestion': lambda u: is_wh_question(u),
        },
        utrs,
        'SearchCreativeWork',
        'data/snips/slot_based_clusters/SearchCreativeWork.json'
    )


# Split by slot values
def split_by_features_SearchScreeningEvent():
    slot_count = {}
    for u in get_utterances_and_slots('data/snips/SearchScreeningEvent/train_SearchScreeningEvent_full.json'):
        for k, v in u['slots'].items():
            p = k + '|' + v if k == 'object_type' else k
            slot_count[p] = slot_count.get(p, 0) + 1
    print(sorted(slot_count.items(), key=lambda x: x[1], reverse=True))

    utrs = get_utterances_and_slots('data/snips/SearchScreeningEvent/train_SearchScreeningEvent_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    def is_yn_question(u):
        return u['text'].startswith('is ')

    def GetScheduleForAMovie(u):
        found = False
        for k in u['slots']:
            if k == 'movie_name':
                found = True
            elif k not in ['object_type', 'object_location_type']:
                return False
        return not is_yn_question(u) and found

    def GetScheduleForAMovieAtALocation(u):
        found = False
        location_found = False

        for k in u['slots']:
            if k == 'movie_name':
                found = True
            elif k == 'location_name':
                location_found = True
            elif k not in ['object_type', 'spatial_relation']:
                return False
        return not is_yn_question(u) and found and location_found

    def GetScheduleForAMovieAtNearbyCinemas(u):
        if 'movie_name' not in u['slots'] \
                or 'object_location_type' not in u['slots'] \
                or 'spatial_relation' not in u['slots']:
            return False
        for k in u['slots']:
            if k not in ['object_type', 'movie_name', 'object_location_type', 'spatial_relation']:
                return False
        return not is_yn_question(u) and True

    def FindCinemasPlayingAMovieAtATimeRange(u):
        if 'timeRange' not in u['slots'] or 'movie_name' not in u['slots']:
            return False
        for k in u['slots']:
            if k not in ['movie_name', 'timeRange', 'object_location_type', 'object_type', 'spatial_relation']:
                return False
        return not is_yn_question(u) and True

    def GetSchedule(u):
        for k in u['slots']:
            if k == 'movie_type':
                if u['slots'][k] not in ['films', 'movies', 'movie', 'film']:
                    return False
            elif k not in ['object_type']:
                return False
        return not is_yn_question(u) and True

    def GetScheduleAtNearbyCinemas(u):
        if 'spatial_relation' not in u['slots']:
            return False
        for k in u['slots']:
            if k == 'movie_type':
                if u['slots'][k] not in ['films', 'movies', 'film', 'movie']:
                    return False
            elif k not in ['object_type', 'object_location_type', 'spatial_relation']:
                return False
        return not is_yn_question(u) and True

    def GetScheduleForAnimatedMoviesAtNearbyCinemas(u):
        if 'object_location_type' not in u['slots'] or 'spatial_relation' not in u['slots']:
            return False
        if ('movie_type', 'animated movies') not in u['slots'].items():
            return False
        for k in u['slots']:
            if k not in ['movie_type', 'object_type', 'object_location_type', 'spatial_relation']:
                return False
        return not is_yn_question(u) and True

    def GetScheduleAtALocation(u):
        found = False
        for k in u['slots']:
            if k == 'location_name':
                found = True
            elif k == 'movie_type':
                if u['slots'][k] not in ['films', 'movies', 'film', 'movie']:
                    return False
            elif k not in ['object_type']:  # timeRange: checking if a movie is playing?
                return False
        return not is_yn_question(u) and found

    def GetScheduleForAnimatedMovies(u):
        found = False
        for k in u['slots']:
            if k == 'movie_type':
                if u['slots'][k] in ['animated movies', 'animated movie']:
                    found = True
            elif k not in ['object_type', 'spatial_relation']:
                return False
        return not is_yn_question(u) and found

    def GetScheduleForAnimatedMoviesAtALocation(u):
        found = False
        animated_movie_found = False
        for k in u['slots']:
            if k == 'location_name':
                found = True
            elif k == 'movie_type':
                if u['slots'][k] in ['animated movies', 'animated movie']:
                    animated_movie_found = True
            elif k not in ['object_type']:  # timeRange: checking if a movie is playing?
                return False
        return not is_yn_question(u) and found and animated_movie_found

    def GetScheduleAtATimeRange(u):
        if 'timeRange' not in u['slots']:
            return False
        for k in u['slots']:
            if k not in ['object_type', 'timeRange', 'object_location_type', 'spatial_relation', 'movie_type']:
                return False
        return not is_yn_question(u) and True

    def GetScheduleAtATimeRangeAtALocation(u):
        time_found = False
        location_found = False
        for k in u['slots']:
            if k == 'location_name':
                time_found = True
            elif k == 'timeRange':
                location_found = True
            elif k == 'movie_type':
                if u['slots'][k] not in ['films', 'movies', 'film', 'movie']:
                    return False
            elif k not in ['object_type']:  # timeRange: checking if a movie is playing?
                return False
        return not is_yn_question(u) and time_found and location_found

    extract_utterances_splitted_by_features(
        {
            'GetSchedule': lambda u: GetSchedule(u),
            'GetScheduleAtATimeRange': lambda u: GetScheduleAtATimeRange(u),
            'GetScheduleAtALocation': lambda u: GetScheduleAtALocation(u),
            'GetScheduleAtATimeRangeAtALocation': lambda u: GetScheduleAtATimeRangeAtALocation(u),
            'GetScheduleAtNearbyCinemas': lambda u: GetScheduleAtNearbyCinemas(u),
            'GetScheduleForAnimatedMoviesAtNearbyCinemas': lambda u: GetScheduleForAnimatedMoviesAtNearbyCinemas(u),
            'GetScheduleForAnimatedMovies': lambda u: GetScheduleForAnimatedMovies(u),
            'GetScheduleForAnimatedMoviesAtALocation': lambda u: GetScheduleForAnimatedMoviesAtALocation(u),
            # Below are movie-specific clusters
            'GetScheduleForAMovie': lambda u: GetScheduleForAMovie(u),
            'GetScheduleForAMovieAtALocation': lambda u: GetScheduleForAMovieAtALocation(u),
            'GetScheduleForAMovieAtNearbyCinemas': lambda u: GetScheduleForAMovieAtNearbyCinemas(u),
            'FindCinemasPlayingAMovieAtATimeRange': lambda u: FindCinemasPlayingAMovieAtATimeRange(u),
        },
        utrs,
        'SearchScreeningEvent',
        'data/snips/slot_based_clusters/SearchScreeningEvent.json'
    )


if __name__ == '__main__':
    # split_by_features_AddToPlaylist()
    # split_by_features_BookRestaurant()
    # split_by_features_GetWeather()
    # split_by_features_RateBook()
    # split_by_features_PlayMusic()
    # split_by_features_SearchCreativeWork()
    # split_by_features_SearchScreeningEvent()

    pass
