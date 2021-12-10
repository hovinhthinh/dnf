from data import snips
from ufd import run_all_intents

report_folder = './reports/global/snips_SMC+US_PC_hac-single/'

intra_intent_data, inter_intent_data = snips.get_train_test_data(use_dev=True)

pipeline_steps = [
    'SMC+US',
    'PC',
]

run_all_intents(pipeline_steps, intra_intent_data, inter_intent_data, report_folder=report_folder,
                config={
                    'dev_test_clustering_method' : 'hac-single',
                })
