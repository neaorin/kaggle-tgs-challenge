import os

from azureml.train.dnn import TensorFlow
from azureml.train.estimator import Estimator
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outputs-folder', help='Local folder where outputs should be saved')
parser.add_argument('--experiment-name', help='The experiment name to collect')
args = parser.parse_args()

output_folder_path = args.outputs_folder if args.outputs_folder else '../outputs/azureml'
expname = args.experiment_name if args.experiment_name else 'tests'

os.makedirs(f'{output_folder_path}/{expname}', exist_ok=True)
submissions_download_folder = f'{output_folder_path}/{expname}/submissions'
os.makedirs(submissions_download_folder, exist_ok=True)


# check workspace 
ws = Workspace.from_config('azureml/config.json')
print(f'Using Azure ML Workspace {ws.name} in location {ws.location}')

experiment = Experiment(ws, "tests")

for run in experiment.get_runs():
    for file in run.get_file_names():
        if file.endswith('submission.csv'):
            run.download_file(file, submissions_download_folder)
            print(f'Downloading {file}')