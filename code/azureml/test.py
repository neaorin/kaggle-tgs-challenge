import azureml
from azureml.core import Workspace, Run

# check Azure ML SDK version
print("Azure ML SDK Version: ", azureml.core.VERSION)

# check workspace 
ws = Workspace.from_config('azureml/config.json')
print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

# upload data
ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)
#ds.upload(src_dir='./data', target_path='data', overwrite=True, show_progress=True)

def log_metric(label, value):
    try:
        run = Run.get_submitted_run()
        run.log(label, value)
    except:
        print(f'{label}: {value}')


log_metric('Accuracy', 0.99)