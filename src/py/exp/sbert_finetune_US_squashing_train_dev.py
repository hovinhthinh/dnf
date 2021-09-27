from data import snips
from ufd import run_all_intents

report_folder = './reports/global/snips_US_squashing_train_dev/'

intra_intent_data, inter_intent_data = snips.get_train_test_data(use_dev=True)

pipeline_steps = [
    'US',
]

run_all_intents(pipeline_steps, intra_intent_data, inter_intent_data, report_folder=report_folder,
                config={
                    'squashing_train_dev': True,
                    'US_n_train_epochs': 3,
                    'PC_iterations': 3
                })
