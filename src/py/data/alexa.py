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


def read_utterances_from_file(file):
    utterances = []
    uniq_utrs = set()
    for line in open(file):
        parts = line.strip().split('\t')

        ucheck = '\t'.join(parts[0:3])
        if ucheck in uniq_utrs:
            continue
        else:
            uniq_utrs.add(ucheck)

        u = Utterance(domain=parts[0], intent_name=parts[1])
        u.text, u.slots = _parse_text_and_slots(parts[2])
        utterances.append(u)
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

        # TODO: add feature_type

        # Split into train/dev/test utterances
        max_utrs_per_train_feature = -1
        max_utrs_per_dev_feature = 200
        max_utrs_per_test_feature = -1
        splitted_data = []
        for c in clusters:
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
            [u.text, u.feature_name, u.part_type, u.slots, u.intent_name, u.domain]
            for u in final_data
        ]
        os.makedirs(os.path.dirname(train_test_data_file), exist_ok=True)
        with open(train_test_data_file, 'w') as f:
            f.write(json.dumps(json_data))
    else:
        json_data = json.loads(open(train_test_data_file).read())
        final_data = [
            Utterance(text=u[0], feature_name=u[1], part_type=u[2], slots=u[3], intent_name=u[4], domain=u[5])
            for u in json_data
        ]

    print('======== Cluster information ========')
    clusters = dict.fromkeys(u.feature_name for u in final_data)
    all_clusters = []
    train_clusters = []
    dev_clusters = []
    test_clusters = []
    for c in clusters:
        total = len([u for u in final_data if u.feature_name == c])
        if c is None:
            all_clusters.append((c, total))
            dev_clusters.append((c, total))
        elif c.endswith('_TRAIN'):
            all_clusters.append((c[:-6], total))
            train_clusters.append((c, total))
        elif c.endswith('_TEST'):
            all_clusters.append((c[:-5], total))
            test_clusters.append((c, total))

    print('All ({}):'.format(len(all_clusters)), all_clusters)
    print('Train ({}):'.format(len(train_clusters)), train_clusters)
    print('Dev ({}):'.format(len(dev_clusters)), dev_clusters)
    print('Test ({}):'.format(len(test_clusters)), test_clusters)

    return final_data


def print_train_dev_test_stats(intent_data):
    train_size = len([u for u in intent_data if u.part_type == 'TRAIN'])
    dev_size = len([u for u in intent_data if u.part_type == 'DEV'])
    test_size = len([u for u in intent_data if u.part_type == 'TEST'])
    print('Train/Dev/Test utterances: {}/{}/{} ({:.1f}%,{:.1f}%,{:.1f}%)'.format(train_size, dev_size, test_size,
                                                                                 100 * train_size / len(intent_data),
                                                                                 100 * dev_size / len(intent_data),
                                                                                 100 * test_size / len(intent_data)))


if __name__ == '__main__':
    alexa_data = get_train_test_data(generate_data=True)
    print_train_dev_test_stats(alexa_data)
