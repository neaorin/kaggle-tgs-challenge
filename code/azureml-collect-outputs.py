import os
from azureml.core import Workspace, Experiment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outputs-folder', default='../outputs/azureml', help='Local folder where outputs should be saved')
parser.add_argument('--experiment-name', default='tests', help='The experiment name to collect')
parser.add_argument('--collect-tensorboard-logs', default=False, help='Whether to also collect TensorBoard logs')
args = parser.parse_args()

output_folder_path = args.outputs_folder
expname = args.experiment_name

os.makedirs(f'{output_folder_path}/{expname}', exist_ok=True)
submissions_download_folder = f'{output_folder_path}/{expname}/submissions'
tblogs_download_folder = f'{output_folder_path}/{expname}/tb_logs'

os.makedirs(submissions_download_folder, exist_ok=True)
if args.collect_tensorboard_logs:
    os.makedirs(tblogs_download_folder, exist_ok=True)

# check workspace 
ws = Workspace.from_config('aml_config/config.json')
print(f'Using Azure ML Workspace {ws.name} in location {ws.location}')

experiment = Experiment(ws, expname)

for run in experiment.get_runs():
    for file in run.get_file_names():
        if file.endswith('submission.csv'):
            print(f'Downloading {file}')
            run.download_file(file, submissions_download_folder)
        if 'tfevents' in file and args.collect_tensorboard_logs:
            _,_,folder,_ = file.split('/')
            folder_path = f'{tblogs_download_folder}/{folder}'
            os.makedirs(folder_path, exist_ok=True)
            print(f'Downloading {file}')
            run.download_file(file, folder_path)
