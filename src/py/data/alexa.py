import json
import os
import random
from math import ceil

from transformers import set_seed

from data.entity import Utterance


def _parse_text_and_slots(utr):
    tokens = []

    sts = []
    for t in utr.strip().split(' '):
        pos = t.find('|')
        assert pos >= 0
        sf = t[:pos]
        sl = t[pos + 1:]

        for s in ['\u200b', '\ufeff', '\u200e']:
            sf = sf.replace(s, '')
            assert s not in sl
        if sf == '':
            if sl != 'Other':
                sts.append(sl)
            continue
        tokens.append((sf, sl))

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

    for s in sts:
        assert (s in slot)

    return text, slot


def read_utterances_from_file(file):
    utterances = []
    uniq_domain = set()
    uniq_utrs = set()
    for line in open(file):
        parts = line.strip().split('\t')

        uniq_domain.add(parts[0])
        ucheck = '\t'.join(parts[0:3])
        if ucheck in uniq_utrs:
            continue
        else:
            uniq_utrs.add(ucheck)

        u = Utterance(domain=parts[0], intent_name=parts[1])
        u.text, u.slots = _parse_text_and_slots(parts[2])
        utterances.append(u)

    for u in utterances:
        u.feature_type = 'DOMAIN' if len(uniq_domain) <= 1 else 'CROSS_DOMAIN'

    return utterances


def read_feature_data(folder='./data/alexa/fr/frXX_feature_data/'):
    utterances = []
    for f in [f for f in os.listdir(folder)]:
        feature_utrs = read_utterances_from_file(os.path.join(folder, f, 'feature_data_uniq.tsv'))
        for u in feature_utrs:
            u.feature_name = f
        utterances.extend(feature_utrs)
    return utterances


def read_live_data(folder='./data/alexa/fr/frXX_live_traffic/'):
    utterances = []
    for f in [f for f in os.listdir(folder)]:
        utterances.extend(read_utterances_from_file(os.path.join(folder, f)))
    for u in utterances:
        u.feature_type = None  # No feature label for live data
    return utterances


def get_train_test_data(generate_data=False):
    set_seed(12993)

    train_test_data_file = './data/alexa/fr/slot_based_clusters/train_test_data_global.json'

    if generate_data:
        # Process feature data
        feature_data = read_feature_data()

        clusters = dict.fromkeys([u.feature_name for u in feature_data])
        predefined_clusters = {
            'TRAIN': [
                'alexa,_print_a_note',
                'alexa_print_games',
                'ambience_control_device_control_utterances_for_powell',
                'artist_experiences',
                'ask_for_subscribe_--_fast-follow_effort',
                'calendar_recurrence_patterns',
                'catapult_-_adding_appliance_slot_to_starttutorialintent',
                'covid-19_vaccination_local_search',
                'customer_reviews_on_alexa_shopping',
                'dab_radio_band_and_stations_recognition_by_nlu',
                'enable_speaker_mode_for_calling',
                'evi_deprecation_for_recipe_domain',
                'follow_me:_move_with_me_fast-follow',
                'gallery_view',
                'hide_photos',
                'kindle_-_navigation_v2',
                'notes_for_you',
                'podcast_ask_for_subscribe',
                'recipes_-_recipe_search',
                'recipes_-_save_recipes',
                'reminder_assignment_bug_fixes_v3',
                'stickynotes-surprisenotes-notesforyou-reminderassignment_-_additions',
                'sticky_notes_-_widgets',
                'tell_me_when_central_cancel_and_browse',
                'tell_me_when_–_info_graphiq_scheduled_events',
                'translateutteranceintent',
                'vui_privacy_settings',
            ],
            'DEV': [
                'browse_alarm-timer_volume',
                'call_via_commprovider_ci',
                'echo_mm_widgets_vui',
                'stickynotes_-_enhancements',
            ],
            'TEST': [
                'artist_experiences_-_fst_improvement',
                'calendar_and_email_account_linking',
                'countdown_feature',
                'follow_me:_voice_move_-_expansion_to_radio_and_podcasts',
                'pedestrian_navigation',
                'photo_booth',
                'sticky_notes_-_emoji',
                'topicspecifiedfeedback',
                'turn_on-off_calendar_notifications',
                'voice_settings_-_accessibility',
            ]
        }
        train_clusters = predefined_clusters['TRAIN']
        dev_clusters = predefined_clusters['DEV']
        test_clusters = predefined_clusters['TEST']
        for c in train_clusters + dev_clusters + test_clusters:
            if c not in clusters:
                raise Exception('Invalid cluster:', c)

        # Split into train/dev/test utterances
        max_utrs_per_train_feature = -1
        max_utrs_per_dev_feature = -1
        max_utrs_per_test_feature = -1
        splitted_data = []
        for c in train_clusters + dev_clusters + test_clusters:
            cluster_data = [u for u in feature_data if u.feature_name == c]
            random.shuffle(cluster_data)

            if c in train_clusters:
                for i, u in enumerate(cluster_data):
                    u.feature_name += '_TRAIN'
                train_data = cluster_data[:ceil(len(cluster_data) * 0.4)]
                dev_data = cluster_data[ceil(len(cluster_data) * 0.4): ceil(len(cluster_data) * 0.5)]
                test_data = cluster_data[ceil(len(cluster_data) * 0.5):]
                for u in train_data:
                    u.part_type = 'TRAIN'
                for u in dev_data:
                    u.part_type = 'DEV'
                for u in test_data:
                    u.part_type = 'TEST'

                if max_utrs_per_train_feature != -1 and len(train_data) > max_utrs_per_train_feature:
                    train_data = train_data[:max_utrs_per_train_feature]
                if max_utrs_per_dev_feature != -1 and len(dev_data) > max_utrs_per_dev_feature:
                    dev_data = dev_data[:max_utrs_per_dev_feature]
                if max_utrs_per_test_feature != -1 and len(test_data) > max_utrs_per_test_feature:
                    test_data = test_data[:max_utrs_per_test_feature]

                splitted_data.extend(train_data)
                splitted_data.extend(dev_data)
                splitted_data.extend(test_data)
            elif c in dev_clusters:
                for u in cluster_data:
                    u.feature_name += '_DEV'
                dev_data = cluster_data[:ceil(len(cluster_data) * 0.5)]
                test_data = cluster_data[ceil(len(cluster_data) * 0.5):]
                for u in dev_data:
                    u.part_type = 'DEV'
                for u in test_data:
                    u.part_type = 'TEST'

                if max_utrs_per_dev_feature != -1 and len(dev_data) > max_utrs_per_dev_feature:
                    dev_data = dev_data[:max_utrs_per_dev_feature]
                if max_utrs_per_test_feature != -1 and len(test_data) > max_utrs_per_test_feature:
                    test_data = test_data[:max_utrs_per_test_feature]

                splitted_data.extend(dev_data)
                splitted_data.extend(test_data)
            elif c in test_clusters:
                for u in cluster_data:
                    u.feature_name += '_TEST'
                    u.part_type = 'TEST'

                if max_utrs_per_test_feature != -1 and len(cluster_data) > max_utrs_per_test_feature:
                    cluster_data = cluster_data[:max_utrs_per_test_feature]

                splitted_data.extend(cluster_data)

        feature_data = splitted_data

        # Process live data
        live_data = read_live_data()
        random.shuffle(live_data)
        sample_size = int(len([u for u in feature_data if u.part_type == 'TRAIN']) * 0.5)
        live_data = live_data[:sample_size]

        for u in live_data:
            u.part_type = 'DEV'

        final_data = feature_data + live_data

        json_data = [
            [u.text, u.feature_name, u.part_type, u.slots, u.intent_name, u.feature_type, u.domain]
            for u in final_data
        ]
        os.makedirs(os.path.dirname(train_test_data_file), exist_ok=True)
        with open(train_test_data_file, 'w') as f:
            f.write(json.dumps(json_data))
    else:
        json_data = json.loads(open(train_test_data_file).read())
        final_data = [
            Utterance
            (text=u[0], feature_name=u[1], part_type=u[2], slots=u[3], intent_name=u[4], feature_type=u[5], domain=u[6])
            for u in json_data
        ]

    return final_data


def print_train_dev_test_stats(intent_data):
    print('======== Cluster information ========')
    clusters = dict.fromkeys(u.feature_name for u in intent_data if u.feature_name is not None)
    all_clusters = []
    train_clusters = []
    dev_clusters = []
    test_clusters = []
    for c in clusters:
        total = len([u for u in intent_data if u.feature_name == c])
        if c is None:
            all_clusters.append((c, total))
            dev_clusters.append((c, total))
        elif c.endswith('_TRAIN'):
            all_clusters.append((c[:-6], total))
            train_clusters.append((c, total))
        elif c.endswith('_DEV'):
            all_clusters.append((c[:-4], total))
            dev_clusters.append((c, total))
        elif c.endswith('_TEST'):
            all_clusters.append((c[:-5], total))
            test_clusters.append((c, total))

    print('Total features:', len(all_clusters))
    print('Train ({}):'.format(len(train_clusters)), train_clusters)
    print('Dev ({}):'.format(len(dev_clusters)), dev_clusters)
    print('Test ({}):'.format(len(test_clusters)), test_clusters)

    # Stats by features:
    features = dict.fromkeys([u.feature_name for u in intent_data if u.feature_name is not None])
    print('######## Stats by feature ########')
    for f in features:
        cnt = {'TRAIN': 0, 'DEV': 0, 'TEST': 0}
        utrs = [u for u in intent_data if u.feature_name == f]
        for u in utrs:
            cnt[u.part_type] += 1
        print('  {}: Train/Dev/Test: {}/{}/{} ({:.1f}%,{:.1f}%,{:.1f}%)'.format(
            f, cnt['TRAIN'], cnt['DEV'], cnt['TEST'],
            100 * cnt['TRAIN'] / len(utrs),
            100 * cnt['DEV'] / len(utrs),
            100 * cnt['TEST'] / len(utrs)))

    print('######## Stats by part type ########')
    for t in ['TRAIN', 'DEV', 'TEST']:
        print(f'{t}:')
        utrs = [u for u in intent_data if u.part_type == t]
        for f in dict.fromkeys([u.feature_name for u in utrs]):
            print('  {}: {}'.format(f, len([u for u in utrs if u.feature_name == f])))

    print('######## Summary ########')
    train_size = len([u for u in intent_data if u.part_type == 'TRAIN'])
    dev_size = len([u for u in intent_data if u.part_type == 'DEV'])
    dev_live_size = len([u for u in intent_data if u.part_type == 'DEV' and u.feature_name is None])
    test_size = len([u for u in intent_data if u.part_type == 'TEST'])
    print('Train/Dev/Test: {}/{}({} live)/{} ({:.1f}%,{:.1f}%,{:.1f}%)'.format(
        train_size, dev_size, dev_live_size, test_size,
        100 * train_size / len(intent_data),
        100 * dev_size / len(intent_data),
        100 * test_size / len(intent_data)))

    supported_test = len([u for u in intent_data if u.part_type == 'TEST' and not u.feature_name.endswith('_TEST')])
    unsupport_test = len([u for u in intent_data if u.part_type == 'TEST' and u.feature_name.endswith('_TEST')])

    print('Supported/Unsupported TEST utterances: {}/{} ({:.1f}%,{:.1f}%)'.format(
        supported_test, unsupport_test, 100 * supported_test / test_size, 100 * unsupport_test / test_size,
    ))


if __name__ == '__main__':
    alexa_data = get_train_test_data(generate_data=False)
    print_train_dev_test_stats(alexa_data)
