from data import alexa
from ufd import run_all_intents

report_folder = './reports/global/alexa_fr/SMC+US_PC_silhouette/'

alexa_data = alexa.get_train_test_data()

pipeline_steps = [
    'SMC+US',
    'PC',
]

run_all_intents(pipeline_steps, None, alexa_data, report_folder=report_folder, config={
    'n_test_clusters_selection_method': 'silhouette'
})
