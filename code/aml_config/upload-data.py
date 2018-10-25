import azureml
from azureml.core import Workspace, Run

# check Azure ML SDK version
print("Azure ML SDK Version: ", azureml.core.VERSION)

# check workspace 
ws = Workspace.from_config('aml_config/config.json')
print(f'Using Azure ML Workspace {ws.name} in location {ws.location}')

# upload the training data
ds = ws.get_default_datastore()
print(f'Default datastore type: {ds.datastore_type}, account name: {ds.account_name}, container name: {ds.container_name}')
ds.upload(src_dir='../data', target_path='data', overwrite=False, show_progress=True)

