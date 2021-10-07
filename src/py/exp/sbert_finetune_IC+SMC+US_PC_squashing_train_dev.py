from data import snips
from ufd import run_all_intents

report_folder = './reports/global/snips_IC+SMC+US_PC_squashing_train_dev/'

_, inter_intent_data = snips.get_train_test_data(use_dev=True)

pipeline_steps = [
    'IC+SMC+US',
    'PC',
]

run_all_intents(pipeline_steps, None, inter_intent_data, report_folder=report_folder,
                config={
                    'squashing_train_dev': True,
                    'PC_iterations': 3
                })
