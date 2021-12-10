from data import alexa
from ufd import run_all_intents

report_folder = './reports/global/alexa_fr/US_PC/'

alexa_data = alexa.get_train_test_data()

pipeline_steps = [
    'US',
    'PC',
]

run_all_intents(pipeline_steps, None, alexa_data, report_folder=report_folder,
                config={
                    # 'US_n_train_epochs': 3,
                    # 'PC_iterations': 3
                })
