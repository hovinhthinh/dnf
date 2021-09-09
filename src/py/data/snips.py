import json
import random
from math import ceil


def get_utterances(input_file):
    data = json.loads(open(input_file, 'r').read())

    data = list(data.values())[0]
    random.shuffle(data)

    utterances = []
    for v in data:
        utterances.append(''.join([t['text'] for t in v['data']]).strip().lower())

    return utterances


def get_utterances_and_slots(input_file):
    """
    :param input_file:
    :return: List[dict]
    """
    data = json.loads(open(input_file, 'r').read())

    utrs = []

    data = list(data.values())[0]
    for i, v in enumerate(data):
        u = {
            'id': i,
            'slots': {}
        }
        text = ''
        for t in v['data']:
            t_text = ' '.join(t['text'].split())
            if t_text == '':
                continue
            if len(text) > 0:
                text += ' '
            if 'entity' in t:
                slot = {
                    'value': t_text,
                    'start': len(text),
                    'end': len(text) + len(t_text)
                }
                u['slots'][t['entity']] = slot

            text += t_text

        for s in u['slots'].values():
            # Verify
            assert text[s['start']:s['end']] == s['value']

        u['text'] = text
        utrs.append(u)
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

    results = []
    n_printed = 0
    for k, v in clusters.items():
        print('{}: {}'.format(k, len(v)))
        for u in v:
            if k == 'UNK':
                # print(u)
                pass
            u = u.copy()
            u.pop('id')
            u['intent'] = intent_name
            u['cluster'] = k
            results.append(u)
            if f is not None:
                if n_printed > 0:
                    f.write(',\n')
                f.write('{}'.format(json.dumps(u)))
            n_printed += 1

    if f is not None:
        f.write('\n]')
        f.close()

    return results


def split_by_features_AddToPlaylist(output_file=None):
    # print_file('data/snips/AddToPlaylist/train_AddToPlaylist_full.json')
    utrs = get_utterances_and_slots('data/snips/AddToPlaylist/train_AddToPlaylist_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    return extract_utterances_splitted_by_features(
        {
            'AddASongToAPlaylist':
                lambda u: ('entity_name' in u['slots']) and ('playlist' in u['slots']),

            'AddAnArtistToAPlaylist':
                lambda u: ('artist' in u['slots']) and ('playlist' in u['slots']) and (
                        'music_item' not in u['slots'] or u['slots']['music_item']['value'] == 'artist'),

            'AddCurrentSongToAPlaylist':
                lambda u: ('music_item' in u['slots'] and u['slots']['music_item']['value'] in ['song', 'track',
                                                                                                'tune'])
                          and ('playlist' in u['slots'])
                          and ('artist' not in u['slots'] or ' this ' in u['text'])
                          and (' a ' not in u['text'] and ' an ' not in u['text'] and ' another ' not in u['text']),

            'AddCurrentAlbumToAPlaylist':
                lambda u: ('music_item' in u['slots'] and u['slots']['music_item']['value'] == 'album') and (
                        'playlist' in u['slots'])
                          and (' a ' not in u['text'] and ' an ' not in u['text'])
                          and (('artist' not in u['slots']) or ' this ' in u['text']),

            'AddCurrentArtistToAPlaylist':
                lambda u: ('music_item' in u['slots'] and u['slots']['music_item']['value'] == 'artist') and (
                        'playlist' in u['slots']) and ('artist' not in u['slots']),

            'AddAnArtistAlbumToAPlaylist':
                lambda u: ('music_item' in u['slots'] and u['slots']['music_item']['value'] == 'album') and (
                        'playlist' in u['slots'])
                          and ('artist' in u['slots']) and (' this ' not in u['text'])
        },
        utrs,
        'AddToPlaylist',
        output_file
    )


def split_by_features_BookRestaurant(output_file=None):
    # print_file('data/snips/BookRestaurant/train_BookRestaurant_full.json')
    utrs = get_utterances_and_slots('data/snips/BookRestaurant/train_BookRestaurant_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    return extract_utterances_splitted_by_features(
        {
            # TODO
        },
        utrs,
        'BookRestaurant',
        output_file
    )


def split_by_features_GetWeather(output_file=None):
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
                if u['slots'][k]['value'] != 'now':
                    good_time_range = False
            elif k not in ['condition_description', 'condition_temperature', 'spatial_relation']:
                return False
        return found and good_time_range

    def GetCurrentWeatherInCurrentPosition(u):
        slots = u['slots']
        # if 'condition_description' in slots or 'condition_temperature' in slots:
        #     return False
        if 'current_location' not in slots:
            for k in slots:
                if k in ['city', 'country', 'state', 'geographic_poi']:
                    return False
        if 'timeRange' in slots and slots['timeRange']['value'] != 'now':
            return False
        return True

    def GetWeatherInALocationAtATimeRange(u):
        found = False
        good_time_range = False
        for k in u['slots']:
            if k in ['city', 'country', 'state', 'geographic_poi']:
                found = True
            elif k == 'timeRange' and u['slots'][k]['value'] != 'now':
                good_time_range = True
            elif k not in ['condition_description', 'condition_temperature', 'spatial_relation']:
                return False
        return found and good_time_range

    def GetWeatherInCurrentPositionAtATimeRange(u):
        slots = u['slots']
        # if 'condition_description' in slots or 'condition_temperature' in slots:
        #     return False
        if 'current_location' not in slots:
            for k in slots:
                if k in ['city', 'country', 'state', 'geographic_poi']:
                    return False
        if 'timeRange' not in slots or slots['timeRange']['value'] == 'now':
            return False
        return True

    return extract_utterances_splitted_by_features(
        {
            'GetCurrentWeatherInALocation': lambda u: GetCurrentWeatherInALocation(u),
            'GetCurrentWeatherInCurrentPosition': lambda u: GetCurrentWeatherInCurrentPosition(u),
            'GetWeatherInALocationAtATimeRange': lambda u: GetWeatherInALocationAtATimeRange(u),
            'GetWeatherInCurrentPositionAtATimeRange': lambda u: GetWeatherInCurrentPositionAtATimeRange(u),
        },
        utrs,
        'GetWeather',
        output_file
    )


def split_by_features_RateBook(output_file=None):
    # print_file('data/snips/RateBook/train_RateBook_full.json')
    utrs = get_utterances_and_slots('data/snips/RateBook/train_RateBook_full.json')
    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    def RateCurrentBook(u):
        if ('rating_value' not in u['slots']) or ('object_select' not in u['slots']) or (
                u['slots']['object_select']['value'] not in ['this', 'current', 'this current']):
            return False
        otype = u['slots'].get('object_type', None) or u['slots'].get('object_part_of_series_type', None)
        return otype is not None and otype['value'] in ['textbook', 'essay', 'novel', 'book', 'album', 'series', 'saga',
                                                        'chronicle', 'essay book', 'book novel', 'book album',
                                                        'series chronicle']

    def RatePreviousBook(u):
        if ('rating_value' not in u['slots']) or ('object_select' not in u['slots']) or (
                u['slots']['object_select']['value'] not in ['previous', 'last']):
            return False
        otype = u['slots'].get('object_type', None) or u['slots'].get('object_part_of_series_type', None)
        return otype is not None and otype['value'] in ['textbook', 'essay', 'novel', 'book', 'album', 'series', 'saga',
                                                        'chronicle', 'essay book', 'book novel', 'book album',
                                                        'series chronicle']

    def RateNextBook(u):
        if ('rating_value' not in u['slots']) or ('object_select' not in u['slots']) or (
                u['slots']['object_select']['value'] not in ['next']):
            return False
        otype = u['slots'].get('object_type', None) or u['slots'].get('object_part_of_series_type', None)
        return otype is not None and otype['value'] in ['textbook', 'essay', 'novel', 'book', 'album', 'series', 'saga',
                                                        'chronicle', 'essay book', 'book novel', 'book album',
                                                        'series chronicle']

    return extract_utterances_splitted_by_features(
        {
            'RateABook': lambda u: 'object_name' in u['slots'] and 'rating_value' in u['slots'],
            'RateCurrentBook': lambda u: RateCurrentBook(u),
            'RatePreviousBook': lambda u: RatePreviousBook(u),
            'RateNextBook': lambda u: RateNextBook(u)
        },
        utrs,
        'RateBook',
        output_file
    )


def split_by_features_PlayMusic(output_file=None):
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
                if u['slots'][k]['value'] not in ['song', 'track', 'tune', 'soundtrack', 'sound track', 'record']:
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
                if u['slots'][k]['value'] not in ['song', 'track']:
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
                if u['slots'][k]['value'] not in ['album', 'song', 'track']:
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
                if u['slots'][k]['value'] not in ['album', 'song', 'track']:
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
                if u['slots'][k]['value'] not in ['track', 'song']:
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
                if u['slots'][k]['value'] not in ['track', 'song']:
                    return False
            elif k == 'service':
                service_found = True
            elif k not in ['sort']:
                return False
        return found and service_found

    return extract_utterances_splitted_by_features(
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
        output_file
    )


# Split by slot values
def split_by_features_SearchCreativeWork(output_file=None):
    utrs = get_utterances_and_slots('data/snips/SearchCreativeWork/train_SearchCreativeWork_full.json')

    slot_count = {}
    for u in utrs:
        for k, v in u['slots'].items():
            p = k + '|' + v['value'] if k == 'object_type' else k
            slot_count[p] = slot_count.get(p, 0) + 1
    print(sorted(slot_count.items(), key=lambda x: x[1], reverse=True))

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

    return extract_utterances_splitted_by_features(
        {
            'SearchCreativeWork': lambda u: ('object_type' not in u['slots']) and not is_wh_question(u),
            'PlayTVProgram': lambda u: ('object_type' in u['slots'])
                                       and (u['slots']['object_type']['value'] in ['TV show', 'show', 'TV series',
                                                                                   'television show', 'movie',
                                                                                   'program',
                                                                                   'saga'])
                                       and not is_wh_question(u),
            'PlayTVProgramTrailer': lambda u: ('object_type' in u['slots'])
                                              and (u['slots']['object_type']['value'] in ['trailer'])
                                              and not is_wh_question(u),
            'PlaySong': lambda u: ('object_type' in u['slots'])
                                  and (u['slots']['object_type']['value'] in ['song', 'soundtrack'])
                                  and not is_wh_question(u),
            'SearchAlbum': lambda u: ('object_type' in u['slots'])
                                     and (u['slots']['object_type']['value'] in ['album'])
                                     and not is_wh_question(u),
            'SearchBook': lambda u: ('object_type' in u['slots'])
                                    and (u['slots']['object_type']['value'] in ['book', 'novel'])
                                    and not is_wh_question(u),
            'SearchPicture': lambda u: ('object_type' in u['slots'])
                                       and (u['slots']['object_type']['value'] in ['picture', 'photograph', 'painting'])
                                       and not is_wh_question(u),
            'SearchGame': lambda u: ('object_type' in u['slots'])
                                    and (u['slots']['object_type']['value'] in ['game', 'video game'])
                                    and not is_wh_question(u),
            'WHQuestion': lambda u: is_wh_question(u),
        },
        utrs,
        'SearchCreativeWork',
        output_file
    )


# Split by slot values
def split_by_features_SearchScreeningEvent(output_file=None):
    utrs = get_utterances_and_slots('data/snips/SearchScreeningEvent/train_SearchScreeningEvent_full.json')

    slot_count = {}
    for u in utrs:
        for k, v in u['slots'].items():
            p = k + '|' + v['value'] if k == 'object_type' else k
            slot_count[p] = slot_count.get(p, 0) + 1
    print(sorted(slot_count.items(), key=lambda x: x[1], reverse=True))

    intent_count = {}
    for u in utrs:
        i = ' '.join(sorted(list(u['slots'].keys())))
        intent_count[i] = intent_count.get(i, 0) + 1
    print(sorted(intent_count.items(), key=lambda k: k[1], reverse=True))

    def is_yn_question(u):
        return u['text'].lower().startswith('is ')

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
                if u['slots'][k]['value'] not in ['films', 'movies', 'movie', 'film']:
                    return False
            elif k not in ['object_type']:
                return False
        return not is_yn_question(u) and True

    def GetScheduleAtNearbyCinemas(u):
        if 'spatial_relation' not in u['slots']:
            return False
        for k in u['slots']:
            if k == 'movie_type':
                if u['slots'][k]['value'] not in ['films', 'movies', 'film', 'movie']:
                    return False
            elif k not in ['object_type', 'object_location_type', 'spatial_relation']:
                return False
        return not is_yn_question(u) and True

    def GetScheduleForAnimatedMoviesAtNearbyCinemas(u):
        if 'object_location_type' not in u['slots'] or 'spatial_relation' not in u['slots']:
            return False
        if 'movie_type' not in u['slots'] or u['slots']['movie_type']['value'] != 'animated movies':
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
                if u['slots'][k]['value'] not in ['films', 'movies', 'film', 'movie']:
                    return False
            elif k not in ['object_type']:  # timeRange: checking if a movie is playing?
                return False
        return not is_yn_question(u) and found

    def GetScheduleForAnimatedMovies(u):
        found = False
        for k in u['slots']:
            if k == 'movie_type':
                if u['slots'][k]['value'] in ['animated movies', 'animated movie']:
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
                if u['slots'][k]['value'] in ['animated movies', 'animated movie']:
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
                if u['slots'][k]['value'] not in ['films', 'movies', 'film', 'movie']:
                    return False
            elif k not in ['object_type']:  # timeRange: checking if a movie is playing?
                return False
        return not is_yn_question(u) and time_found and location_found

    return extract_utterances_splitted_by_features(
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
        output_file
    )


def get_train_test_data(generate_data=False, use_dev=True):
    """
    :return: intra_intent_data: List[Tuple[utterance: str, cluster_label: any, sample_type: 'TRAIN'|'DEV'|'TEST', slots: dict]],
    inter_intent_data: List[Tuple[intent_name, List]]
    """
    train_test_data_file = './data/snips/slot_based_clusters/train_test_data_global.json'

    if generate_data:
        intent_data = [
            (split_by_features_GetWeather(), {  # Hard-code train/dev/test clusters
                'TRAIN': ['GetCurrentWeatherInALocation', 'GetWeatherInCurrentPositionAtATimeRange'],
                'DEV': ['GetCurrentWeatherInCurrentPosition'],
                'TEST': ['GetWeatherInALocationAtATimeRange'],
            }),
            (split_by_features_AddToPlaylist(), {
                'TRAIN': ['AddCurrentArtistToAPlaylist', 'AddASongToAPlaylist'],
                'DEV': ['AddAnArtistToAPlaylist', 'AddCurrentSongToAPlaylist'],
                'TEST': ['AddCurrentAlbumToAPlaylist'],
            }),
            (split_by_features_RateBook(), {
                'TRAIN': ['RateABook'],
                'DEV': ['RateCurrentBook'],
                'TEST': [],
            }),
            (split_by_features_BookRestaurant(), None),  # Set to None to automatically randomly sample
            (split_by_features_PlayMusic(), {
                'TRAIN': ['PlayAlbum', 'PlayMusicByYear', 'PlayMusicByGenre', 'PlayMusicByArtistOnService',
                          'PlayMusicByArtistAndYearOnService', 'PlayMusicByArtist'],
                'DEV': ['PlayTrack', 'PlayMusicByYearOnService', 'PlayTrackOnService', 'PlayPlaylist'],
                'TEST': ['PlayAlbumOnService', 'PlayMusicByGenreOnService', 'PlayOnService',
                         'PlayMusicByArtistAndYear'],
            }),
            (split_by_features_SearchCreativeWork(), {
                'TRAIN': ['WHQuestion', 'SearchPicture', 'PlayTVProgram', 'SearchGame'],
                'DEV': ['PlaySong', 'SearchCreativeWork', 'SearchAlbum'],
                'TEST': ['SearchBook', 'PlayTVProgramTrailer'],
            }),
            (split_by_features_SearchScreeningEvent(), {  # Hard-code train/dev/test clusters
                'TRAIN': ['GetScheduleForAMovieAtNearbyCinemas', 'GetSchedule', 'GetScheduleForAMovie',
                          'GetScheduleAtALocation'],
                'DEV': ['GetScheduleForAnimatedMovies', 'GetScheduleAtNearbyCinemas', 'GetScheduleAtATimeRange'],
                'TEST': ['FindCinemasPlayingAMovieAtATimeRange', 'GetScheduleForAMovieAtALocation'],
            }),
        ]

        intra_intent_data = []
        # Prepare data
        for data, predefined_clusters in intent_data:
            clusters = list(set([u['cluster'] for u in data]))

            if predefined_clusters is not None:
                train_clusters = predefined_clusters['TRAIN']
                dev_clusters = predefined_clusters['DEV']
                test_clusters = predefined_clusters['TEST']
                for c in train_clusters + dev_clusters + test_clusters:
                    if c not in clusters:
                        raise Exception('Invalid cluster:', c)
            else:
                # Filter clusters with small size
                clusters = [c for c in clusters if sum([1 for u in data if u['cluster'] == c]) >= 50 and c != 'UNK']

                # Random train/dev/test clusters
                random.shuffle(clusters)
                train_clusters = clusters[:ceil(len(clusters) * 0.4)]
                dev_clusters = clusters[ceil(len(clusters) * 0.4):ceil(len(clusters) * 0.7)]
                test_clusters = clusters[ceil(len(clusters) * 0.7):]

            # Split into train/test utterances
            splitted_data = []
            for c in clusters:
                cluster_data = [u for u in data if u['cluster'] == c]
                random.shuffle(cluster_data)
                if c in train_clusters:
                    splitted_data.extend([(u['text'], c + '_TRAIN', 'TRAIN', u['slots']) for u in
                                          cluster_data[:ceil(len(cluster_data) * 0.4)]])
                    splitted_data.extend([(u['text'], c + '_TRAIN', 'DEV', u['slots']) for u in
                                          cluster_data[ceil(len(cluster_data) * 0.4):ceil(len(cluster_data) * 0.7)]])
                    splitted_data.extend([(u['text'], c + '_TRAIN', 'TEST', u['slots']) for u in
                                          cluster_data[ceil(len(cluster_data) * 0.7):]])
                elif c in dev_clusters:
                    splitted_data.extend([(u['text'], c + '_DEV', 'DEV', u['slots']) for u in
                                          cluster_data[:ceil(len(cluster_data) * 0.7)]])
                    splitted_data.extend([(u['text'], c + '_DEV', 'TEST', u['slots']) for u in
                                          cluster_data[ceil(len(cluster_data) * 0.7):]])
                elif c in test_clusters:
                    splitted_data.extend([(u['text'], c + '_TEST', 'TEST', u['slots']) for u in cluster_data])

            intra_intent_data.append((data[0]['intent'], splitted_data))

        with open(train_test_data_file, 'w') as f:
            f.write(json.dumps(intra_intent_data))
    else:
        intra_intent_data = json.loads(open(train_test_data_file).read())

    # Merge DEV into TRAIN if dev is not used.
    if not use_dev:
        for intent_name, cluster_data in intra_intent_data:
            for u in cluster_data:
                if u[1].endswith('_DEV'):
                    u[1] = u[1][:-4] + '_TRAIN'
                if u[2] == 'DEV':
                    u[2] = 'TRAIN'

    inter_intent_data = []
    for intent_name, cluster_data in intra_intent_data:
        inter_intent_data.extend(
            [(text, intent_name + '_' + cluster, sample_type, slots)
             for text, cluster, sample_type, slots in cluster_data])

    print('======== Cluster information ========')
    intra_intent_data_filtered = []
    for intent_name, cluster_data in intra_intent_data:
        print("====", intent_name)
        clusters = set(u[1] for u in cluster_data)
        train_clusters = []
        dev_clusters = []
        test_clusters = []
        for c in clusters:
            total = len([u for u in cluster_data if u[1] == c])
            if c.endswith('_TRAIN'):
                train_clusters.append((c, total))
            elif c.endswith('_DEV'):
                dev_clusters.append((c, total))
            elif c.endswith('_TEST'):
                test_clusters.append((c, total))

        print('Train:', train_clusters)
        if use_dev:
            print('Dev:', dev_clusters)
        print('Test:', test_clusters)

        if len(train_clusters) < 1 or len(test_clusters) < 1 or (use_dev and len(dev_clusters) < 1):
            print('Ignore this intent for intra-intent setting')
        else:
            intra_intent_data_filtered.append((intent_name, cluster_data))

    return intra_intent_data_filtered, inter_intent_data


def print_train_dev_test_stats(intent_data):
    train_size = len([u for u in intent_data if u[2] == 'TRAIN'])
    dev_size = len([u for u in intent_data if u[2] == 'DEV'])
    test_size = len([u for u in intent_data if u[2] == 'TEST'])
    print('Train/Dev/Test utterances: {}/{}/{} ({:.1f}%,{:.1f}%,{:.1f}%)'.format(train_size, dev_size, test_size,
                                                                                 100 * train_size / len(intent_data),
                                                                                 100 * dev_size / len(intent_data),
                                                                                 100 * test_size / len(intent_data)))


if __name__ == '__main__':
    intra_intent_data, _ = get_train_test_data()
    for name, data in intra_intent_data:
        print('Intent: ', name)
        print_train_dev_test_stats(data)
    pass
