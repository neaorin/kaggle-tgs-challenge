from azureml.train.dnn import TensorFlow
from azureml.train.estimator import Estimator
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', help='JSON file containing run configuration')
parser.add_argument('--entry-script', help='Entry Python script to run')
parser.add_argument('--experiment-name', help='The experiment name to use')

args = parser.parse_args()

if not args.config_file:
    print('The --config-file parameter is missing, exiting.')
    sys.exit(1)

entry_script = args.entry_script if args.entry_script else 'train-fold.py'
expname = args.experiment_name if args.experiment_name else 'tests'

with open(args.config_file) as config_file:
    config = json.load(config_file)


# check workspace 
ws = Workspace.from_config('azureml/config.json')
print(f'Using Azure ML Workspace {ws.name} in location {ws.location}')

# default data store
ds = ws.get_default_datastore()

try:
  compute_target = ComputeTarget(workspace=ws, name='tgschallenge')
except ComputeTargetException:
  compute_config = BatchAiCompute.provisioning_configuration(vm_size='STANDARD_NC6', vm_priority='lowpriority', autoscale_enabled=True, cluster_min_nodes=0, cluster_max_nodes=4)
  compute_target = ComputeTarget.create(ws, 'tgschallenge', compute_config)
  compute_target.wait_for_completion(show_output=True)
  print(compute_target.get_status())

script_params = {
    '--config-file': args.config_file,
    '--data-folder': ds.as_mount(),
    '--outputs-folder': 'outputs'
}


cv_folds = int(config['crossvalidation']['folds']) if config['crossvalidation'] is not None else None
exp = Experiment(workspace=ws, name=f'{expname}')

if cv_folds is not None:
    for i in range(cv_folds):
        script_params['--cv-currentfold'] = str(i)
        est = TensorFlow(
            source_directory='.', 
            script_params=script_params, 
            entry_script=entry_script,
            compute_target=compute_target, 
            use_gpu=True,
            use_docker=True, 
            pip_requirements_file_path='requirements.txt'
            )
        run = exp.submit(config=est)
        print(f'Experiment {expname} submitted for fold {i}.')   
else:
    est = TensorFlow(
        source_directory='.', 
        script_params=script_params, 
        entry_script=entry_script,
        compute_target=compute_target, 
        use_gpu=True,
        use_docker=True, 
        pip_requirements_file_path='requirements.txt'
    )
    run = exp.submit(config=est)
    print(f'Experiment {expname} submitted.')