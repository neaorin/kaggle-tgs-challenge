import azureml
from azureml.core import Workspace, Run
from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

# check Azure ML SDK version
print("Azure ML SDK Version: ", azureml.core.VERSION)

# check workspace 
ws = Workspace.from_config('aml_config/config.json')
print(f'Using Azure ML Workspace {ws.name} in location {ws.location}')

# check and create the Batch AI Compute cluster
try:
  compute_target = ComputeTarget(workspace=ws, name='tgschallenge')
  print('The BatchAI cluster already exists.')
except ComputeTargetException:
  compute_config = BatchAiCompute.provisioning_configuration(vm_size='STANDARD_NC6', vm_priority='dedicated', autoscale_enabled=True, cluster_min_nodes=0, cluster_max_nodes=4)
  compute_target = ComputeTarget.create(ws, 'tgschallenge', compute_config)
  compute_target.wait_for_completion(show_output=True)
  print(compute_target.get_status())