import numpy as np
import matplotlib
import matplotlib.pyplot as plt
 
import azureml
from azureml.core import Workspace, Run

from azureml.core import Workspace
from azureml.core.model import Model
import os
ws = Workspace.from_config()
model = Model(ws, 'model_1')

model.download(target_dir=os.getcwd(), exist_ok=True)

# verify the downloaded model file
file_path = os.path.join(os.getcwd(), "trained_models\model_n_estimators_7.pkl")

os.stat(file_path)


# Testing score.py

# Load test data:
from data_management import load_dataset
import configuracion
import score

data = load_dataset(file_name=configuracion.TRAINING_DATA_FILE)
data = data.iloc[:8,:]
data = data.to_json()

score.init()
pred = score.run(data)
print(pred)


# Create environment file
'''
- Add package requeriments in this file -> myenv.yml
'''

from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

with open("myenv.yml", "w") as f:
    f.write(myenv.serialize_to_string())


# Create a configuration file
from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "titanic",  
                                                     "method": "sklearn"},
                                               description='predict titanic survival')


# Deployment
from azureml.core import Workspace
from azureml.core.model import Model
import os
from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(runtime= "python", 
                                   entry_script="score.py",
                                   conda_file="myenv.yml")

ws = Workspace.from_config()
model = Model(ws, 'model_2')                                   

service = Model.deploy(workspace=ws, 
                       name='titanic-predictor-svc',
                       models=[model], 
                       inference_config=inference_config,
                       deployment_config=aciconfig)

service.wait_for_deployment(show_output=True)
    


#####################################################
#######################################################

from azureml.core import Workspace

ws = Workspace.from_config()

from azureml.core.model import Model
import sklearn


model = Model.register(model_path = "finalmodel2.pkl",
                       model_name = "finalmodel2.pkl",
                       tags = {'area': "diabetes", 'type': "regression", 'version': '1'},
                       description = "Ridge regression model to predict diabetes",
                       workspace = ws)


from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies.create(conda_packages=['numpy','scikit-learn'])

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())

from azureml.core.image import Image, ContainerImage

image_config = ContainerImage.image_configuration(runtime= "python",
                                 execution_script="score.py",
                                 conda_file="myenv.yml",
                                 tags = {'area': "diabetes", 'type': "regression"},
                                 description = "Image with ridge regression model")


image = Image.create(name = "myimagefinal",
                     # this is the model object. note you can pass in 0-n models via this list-type parameter
                     # in case you need to reference multiple models, or none at all, in your scoring script.
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)

image.wait_for_creation(show_output = True)


from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {'area': "diabetes", 'type': "regression"}, 
                                               description = 'Predict diabetes using regression model')

from azureml.core.webservice import Webservice

aci_service_name = 'my-aci-service-final'
print(aci_service_name)
aci_service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                           image = image,
                                           name = aci_service_name,
                                           workspace = ws)
aci_service.wait_for_deployment(True)
print(aci_service.state)
















