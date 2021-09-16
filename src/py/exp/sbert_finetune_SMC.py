import os

from data import snips
from data.snips import print_train_dev_test_stats
from ufd import Pipeline

report_folder = './reports/global/snips_SMC/'

intra_intent_data, inter_intent_data = snips.get_train_test_data(use_dev=True)

pipeline_steps = [
    'SMC',
]

# Processing intra-intents
for intent_name, intent_data in intra_intent_data:
    print('======================================== Intra-intent:', intent_name,
          '========================================')

    if len([u for u in intent_data if u[1].endswith('_TRAIN')]) == 0 or len(
            [u for u in intent_data if u[1].endswith('_TEST')]) == 0:
        print('Ignore this intent for intra-intent setting')
        continue

    print_train_dev_test_stats(intent_data)
    p = Pipeline(intent_data, dataset_name=intent_name)
    intent_report_folder = os.path.join(report_folder, intent_name) if report_folder is not None else None
    p.run(report_folder=intent_report_folder, steps=pipeline_steps)

# Processing inter-intents
print('======================================== Inter-intent ======================================')
print_train_dev_test_stats(inter_intent_data)
p = Pipeline(inter_intent_data, dataset_name='inter-intent')
intent_report_folder = os.path.join(report_folder, 'inter_intent') if report_folder is not None else None
p.run(report_folder=intent_report_folder, steps=pipeline_steps)

# Apply back to intra-intent
print('======== Apply inter-intent model back to intra-intent ========')

stats_file = None
if report_folder is not None:
    os.makedirs(report_folder, exist_ok=True)
    stats_file = open(os.path.join(report_folder, 'stats.txt'), 'w')
    stats_file.write('======== Apply inter-intent model back to intra-intent ========\n')

for intent_name, intent_data in intra_intent_data:
    if len([u for u in intent_data if u[1].endswith('_TEST')]) == 0:
        continue

    p = Pipeline(intent_data)
    p.update_test_embeddings()
    test_quality = p.get_test_clustering_quality()
    print('Clustering test quality [{}]: {}'.format(intent_name, test_quality))
    if stats_file is not None:
        stats_file.write('Clustering test quality [{}]: {}\n'.format(intent_name, test_quality))

if stats_file is not None:
    stats_file.close()
