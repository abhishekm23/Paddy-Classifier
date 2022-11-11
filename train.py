from utils import PaddyModel
import argparse
import os 

weights_help = '''
Model weights to use. Can be either best.h5 or last.h5, defaults to best.h5
best.h5 - The weights from the chosen version having minimum validation loss
last.h5 - The weights from the chosen version on the last epoch
'''

version_help = 'The mlflow run id you want to use'

epochs_help = '''
The number of epochs to train the model. defaults to None. 
If the epochs is None, it trains the model for 100 epochs using early stopping
'''

lr_help = 'The learning rate to use while training the model, defaults to 0.01'

batch_size_help = 'The batch size to use while training the model, defaults to 16 for computational efficiency'

size_help = '''
The size of the image to be used while training the model, defaults to 331. 
NOTE - The size should be greater than 331 as NASNet uses 331 as the minimum input size
'''

aug_path_help = 'The path to augmentations.yaml file'

data_path_help = 'The path to data.yaml file'

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='best.h5', help=weights_help)
parser.add_argument('--run_id', type=int, default=None, help=version_help)
parser.add_argument('--epochs', type=int, default=None, help=epochs_help)
parser.add_argument('--lr', type=float, default=1e-3, help=lr_help)
parser.add_argument('--batch_size', type=int, default=16, help=batch_size_help)
parser.add_argument('--size', type=int, default=331, help=size_help)
parser.add_argument('--data_path', type=str, default='data.yaml', help=data_path_help)
parser.add_argument('--aug_path', type=str, default='augmentations.yaml', help=aug_path_help)

args = parser.parse_args()

weights = args.weights
run_id = args.run_id 
epochs = args.epochs 
lr = args.lr 
batch_size = args.batch_size
size = args.size 
data_path = args.data_path
aug_path = args.aug_path

RETRAIN = False
BASE_PATH = os.path.join('mlruns', '0')

model = PaddyModel(data_path=data_path, aug_path=aug_path)

if run_id is not None: 
    reload_path = os.path.join(BASE_PATH, run_id, 'artifacts', weights)

else: 
    RETRAIN = True

if RETRAIN:
    model.train(epochs=epochs, lr=lr, batch_size=batch_size)
else:
    model.train(epochs=epochs, lr=lr, batch_size=batch_size, reload_path=reload_path)