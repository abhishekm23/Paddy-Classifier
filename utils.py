import yaml 
import pandas as pd 
import matplotlib.pyplot as plt 
import keras.applications as applications 
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, GlobalAveragePooling2D,Concatenate, Dense
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import CategoricalAccuracy, AUC
from tensorflow_addons.metrics import F1Score
from keras.losses import CategoricalCrossentropy
from keras.optimizer_v2.adam import Adam
import numpy as np 
import os 
import re 
from sklearn.model_selection import StratifiedKFold, train_test_split
from glob import glob
import mlflow 

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

run = mlflow.start_run()


class PaddyModel:
    def __init__(self, num_classes=10, target_size=331, data_path='data.yaml', aug_path = 'augmentations.yaml'):
        self.TARGET_SIZE = target_size
        self.NUM_CLASSES = num_classes
        self.DATA_PATH = data_path
        self.AUG_PATH = aug_path
        
    def create_data(self, batch_size = 16,inference=False, size=331, kfold=False):
        self.TARGET_SIZE = size
        with open(self.DATA_PATH,'r') as f:
            data = yaml.safe_load(f)
        train_path = data['train']
        
        with open(self.AUG_PATH, 'r') as f:
            augmentations = yaml.safe_load(f)
        if 'rescale' in augmentations['train']:
            augmentations['train']['rescale'] = 1/float(augmentations['train']['rescale'])
        if augmentations['test'] is not None and 'rescale' in augmentations['test']:
            augmentations['test']['rescale'] = 1/float(augmentations['test']['rescale'])
            
        train_images = glob(f'{train_path}/*/*.jpg')
        
        df = pd.DataFrame({
            'X':train_images
        })
        
        df['y'] = df['X'].apply(lambda x: x.split(os.path.sep)[-2])
        test_size = int(len(df)*0.1)
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['y'])
        train_df, val_df = train_test_split(train_df, test_size=test_size, stratify=train_df['y'])
        
        train_gen = ImageDataGenerator(**augmentations['train'])
        if augmentations['test'] is not None:
            test_gen = ImageDataGenerator(**augmentations['test'])
        else:
            test_gen = ImageDataGenerator()
                   
        self.flow_specs = {
            'x_col':'X',
            'y_col':'y',
            'target_size':(self.TARGET_SIZE, self.TARGET_SIZE),
            'class_mode':'categorical',
            'batch_size':batch_size 
        }
        
        if inference:
            test_set = test_gen.flow_from_dataframe(dataframe = test_df,**self.flow_specs, shuffle=False)
            return test_set
        if kfold:
            return test_df
        else:
            train_set = train_gen.flow_from_dataframe(dataframe = train_df,**self.flow_specs, shuffle=True)
            val_set = train_gen.flow_from_dataframe(dataframe = val_df,**self.flow_specs, shuffle=True)
            return train_set, val_set

    
    def create_model(self,lr=1e-3):
        pretrained_specs =  {
            'weights':'imagenet',
            'include_top':False
        }
        
        xception_base = applications.xception.Xception(**pretrained_specs)
        inception_v3_base = applications.inception_v3.InceptionV3(**pretrained_specs)
        inception_resnet_v2_base = applications.inception_resnet_v2.InceptionResNetV2(**pretrained_specs)
        nasnet_base = applications.nasnet.NASNetLarge(**pretrained_specs)
        
        
        xception_base.trainable = False 
        inception_v3_base.trainable = False 
        inception_resnet_v2_base.trainable = False 
        nasnet_base.trainable = False 
          
        inputs = Input(shape=(self.TARGET_SIZE, self.TARGET_SIZE, 3))
        
        # Xception 
        x1 = applications.xception.preprocess_input(inputs)
        x1 = xception_base(x1)
        x1 = GlobalAveragePooling2D()(x1)
        
        # Inception V3 
        x2 = applications.inception_v3.preprocess_input(inputs)
        x2 = inception_v3_base(x2)
        x2 = GlobalAveragePooling2D()(x2)
        
        # Inception Resnet V22
        x3 = applications.inception_resnet_v2.preprocess_input(inputs)        
        x3 = inception_resnet_v2_base(inputs)
        x3 = GlobalAveragePooling2D()(x3)
        
        # Nasnet
        x4 = applications.nasnet.preprocess_input(inputs)
        x4 = nasnet_base(x4)
        x4 = GlobalAveragePooling2D()(x4)
        
        x = Concatenate()([x1,x2,x3,x4])        
        outputs = Dense(units=self.NUM_CLASSES, activation='softmax')(x)
        self.model = Model(inputs=[inputs], outputs=[outputs])
        
        metrics = [
            CategoricalAccuracy(),
            AUC(),
            F1Score(num_classes=self.NUM_CLASSES)
        ]
        
        lossfn = CategoricalCrossentropy()
        opt = Adam(learning_rate=lr)
        self.model.compile(loss=lossfn, optimizer=opt, metrics=metrics)
        
    def train(self, epochs=None,lr=1e-3, reload_id = None, batch_size=16, size=331):
        
        base_path = os.path.join('mlruns', '0')
        if epochs is None:
            TRAIN_EPOCHS = 100
        else:
            TRAIN_EPOCHS = epochs
            
        mlflow.log_param(key='batch_size', value=batch_size)
        mlflow.log_param(key='learning_rate', value=lr)
        mlflow.log_param(key='img_size', value=size)    
        mlflow.log_param(key='epochs', value=TRAIN_EPOCHS)
    
        train_set, val_set = self.create_data(batch_size=batch_size, size=size)
        if reload_id is None: 
            self.create_model(lr=lr)
        else: 
            reload_path = os.path.join(base_path, reload_id, '0', 'artifacts', 'best.h5')
            self.model = load_model(reload_path, custom_objects={'f1_score':F1Score(num_classes = self.NUM_CLASSES)})
 
        run_id = run.info.id
        model_save_path = os.path.join(base_path, run_id, '0', 'artifacts')
        callbacks = [
            ModelCheckpoint(os.path.join(model_save_path, 'best.h5'),save_best_only=True),
            EarlyStopping(min_delta=1e-4, patience=3)
        ]
        history = self.model.fit(train_set, steps_per_epoch=len(train_set),
                                validation_data = val_set, validation_steps=len(val_set),
                                callbacks=callbacks, epochs=TRAIN_EPOCHS)
        
        mlflow.keras.log_model(self.model, 
                               artifact_path=os.path.join(
                                   model_save_path, 'last.h5')
                               )
        
        best_model = load_model(os.path.join(model_save_path, 'best.h5'))
        # Plotting convergence curves 
        self.plot_convergence_curve(history)
        # Performing K-fold validation
        self.perform_kfold(best_model)
    
    def plot_convergence_curve(self, history):
        history_df = pd.DataFrame(history.history)
        history_df['f1_score'] = history_df['f1_score'].apply(np.mean)
        history_df['val_f1_score'] = history_df['val_f1_score'].apply(np.mean)
        history_df.columns = history_df.columns.to_series().apply(lambda x: re.sub('_[0-9]', '', x))
        
        fig, ax = plt.subplots(figsize=(16,16), nrows=2, ncols=2)
        
        history_df[['loss','val_loss']].plot(
            ax=ax[0,0], title='Loss Plot', xlabel='Epochs', ylabel='Loss')
        
        history_df[['categorical_accuracy','val_categorical_accuracy']].plot(
            ax=ax[0,1],xlabel='Epochs',ylabel='Categorical Accuracy', title='Categorical Accuracy plot')
        
        history_df[['f1_score','val_f1_score']].plot(
            ax=ax[1,0], xlabel='Epochs',ylabel='F1 Score', title='F1 Score Plot')
        
        history_df[['auc','val_auc']].plot(
            ax=ax[1,1], xlabel='Epochs',ylabel='AUC', title='AUC Plot')
        
        mlflow.log_figure(fig, 'convergence_plot.png')
    
    def perform_kfold(self,model, n_splits=10):
        test_df = self.create_data(kfold=True)
        test_gen = ImageDataGenerator(rescale=1/255.0)
        kf = StratifiedKFold(n_splits=n_splits)
        performance = {
            'Loss':[],
            'Accuracy':[],
            'AUC':[],
            'F1Score':[]
        }
        for fold, (_, test_idx) in enumerate(kf.split(X=test_df['X'], y=test_df['y'])):
            print(f'Evaluating fold {fold+1}')
            test_subset = test_df.iloc[test_idx, :]
            test_data = test_gen.flow_from_dataframe(dataframe=test_subset, **self.flow_specs, shuffle=False)
            test_loss, test_acc, test_auc, test_f1 = model.evaluate(test_data)               
            performance['Loss'].append(test_loss)
            performance['Accuracy'].append(test_acc)
            performance['AUC'].append(test_auc)
            performance['F1Score'].append(np.mean(test_f1))
        
        
        mlflow.log_metrics(performance)
        
        