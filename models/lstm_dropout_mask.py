import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from keras.layers import Dense, LSTM, LayerNormalization, Add
from keras.models import Sequential
from keras.regularizers import l2
from keras.activations import softmax

from focal_loss import sparse_categorical_focal_loss # pip install git+https://github.com/artemmavrin/focal-loss.git
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import wandb
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import Multi_class_evaluation

import itertools

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


hyperparameters_grid = [
    {'learning_rate': 0.001, 'epochs': 400,  'gamma': 2, 'batch_size': 512, 'class_weight':[1,20,10]}
]

# data loader ----------------------------------------
class Data:
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size

    def create_data(self):
        with open("/disk/cshen/vaccine/data/vaccine_train_lstm.pkl", "rb") as f:
            loaded_dt = pickle.load(f)

        X = loaded_dt['inputs'].astype(np.float32)     # all as float
        y = loaded_dt['outputs'].astype(np.float32)
        X_len = [float(i) for i in loaded_dt['length']]
        self.total_samples, self.sequence_length, self.input_dim = loaded_dt['inputs'].shape
        self.num_classes = len(set(y.reshape(-1)))

        full_dataset = tf.data.Dataset.from_tensor_slices((X, y, X_len))
        full_dataset = full_dataset.shuffle(buffer_size=self.total_samples, reshuffle_each_iteration=False)

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_ds = full_dataset.take(train_size)
        val_ds = full_dataset.skip(train_size)
        train_ds = train_ds.batch(self.batch_size)
        val_ds = val_ds.batch(self.batch_size)

        return train_ds, val_ds

    def create_model(self,lstm_units, dropout_rate, recurrent_dropout_rate, penalty=0.001):
    
        model = Sequential()
        model.add(LSTM(units=lstm_units,
                        input_shape=(self.sequence_length, self.input_dim), 
                        kernel_regularizer=l2(penalty),
                        recurrent_regularizer=l2(penalty),
                        dropout=dropout_rate, 
                        recurrent_dropout=recurrent_dropout_rate, 
                        return_sequences=True))
        model.add(LSTM(units=lstm_units, 
                        kernel_regularizer=l2(penalty),
                        recurrent_regularizer=l2(penalty),
                        dropout=dropout_rate, 
                        recurrent_dropout=recurrent_dropout_rate, 
                        return_sequences=True))
        model.add(Dense(units=self.num_classes, 
                        kernel_initializer='glorot_normal'))
        return model

# focal loss -----------------------------------
@tf.function
def focal_loss(model, x, y, x_len, max_sequence, gamma, class_weight, plot = False):

    # https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_categorical_focal_loss.py        
    masking = tf.sequence_mask(x_len, maxlen=max_sequence, dtype=tf.float32)
    logits = model(x)
    fc_loss = sparse_categorical_focal_loss(y, logits, from_logits=True, gamma=gamma, class_weight=class_weight) * masking
    sequence_loss = tf.reduce_mean(tf.reduce_sum(fc_loss, axis=1) / x_len)

    if plot == False:
        return sequence_loss
    
    if plot == True:
        probs_time = tf.nn.softmax(logits)
        labels = tf.math.argmax(probs_time, axis=2)

        cast_label = tf.cast(tf.math.equal(labels,2), masking.dtype)
        num_grp2 = tf.math.reduce_sum(cast_label * masking, axis=0)

        cast_y = tf.cast(tf.math.equal(y,2), masking.dtype)
        num_grp2_real = tf.math.reduce_sum(cast_y * masking, axis=0)

        num_patients = tf.math.reduce_sum(masking, axis=0)
        num_prob = tf.math.reduce_sum(probs_time[:,:,2], axis = 0)
        
        return sequence_loss, (num_grp2, num_grp2_real, num_patients, num_prob)
    
# model ----------------------------------------
def train(hyperparams, current_output_dir):
    tr_loss_hist = []
    val_loss_hist = []
    break_times = 0

    class config:
        learning_rate = hyperparams['learning_rate']
        epochs = hyperparams['epochs']
        batch_size = hyperparams['batch_size']
        gamma = hyperparams['gamma']
        loss_type = 'focal'
        class_weight = hyperparams['class_weight']
    
    # read data 
    vaccine = Data(config.batch_size)
    train_ds, val_ds = vaccine.create_data()
    sequence_length = vaccine.sequence_length

    # create model 
    model = vaccine.create_model(lstm_units=128, dropout_rate=0.3, recurrent_dropout_rate=0.3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    print(model.summary())

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}")
        obs_group2, real_group2, num_patients_all, sum_prob = 0, 0, 0, 0
        L2_train_value = []
        L2_valid_value = []

        # Training Phase
        loop = tqdm(train_ds, desc='Train')
        for x_mb, y_mb, x_mb_len in loop:
            with tf.GradientTape() as tape:
                tr_loss, (num_group2, num_group2_real, num_patients, num_prob) = focal_loss(
                    model, x_mb, y_mb, x_mb_len, 
                    max_sequence=sequence_length, gamma=config.gamma, 
                    class_weight=config.class_weight, plot = True)
            grads = tape.gradient(tr_loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            
            obs_group2 += num_group2
            real_group2 += num_group2_real
            num_patients_all += num_patients
            sum_prob += num_prob

        L2 = np.linalg.norm((obs_group2[20:]-real_group2[20:])/num_patients_all[20:])
        L2_train_value.append(L2)
        
        # Validation Phase
        avg_val_loss, avg_val_f1, avg_val_auc = 0.0, 0.0, 0.0
        obs_group2, real_group2, num_patients_all, sum_prob = 0, 0, 0, 0
        cm_all = np.zeros((3,3))

        loop2 = tqdm(val_ds, desc = 'Val')
        for x_val, y_val, x_val_len in loop2:
            val_loss, (num_group2, num_group2_real, num_patients, num_prob) = focal_loss(
                model, x_val, y_val, x_val_len, 
                max_sequence=sequence_length, gamma=config.gamma,
                class_weight=config.class_weight, plot = True
                )            

            obs_group2 += num_group2
            real_group2 += num_group2_real
            num_patients_all += num_patients
            sum_prob += num_prob

        L2 = np.linalg.norm((obs_group2[20:]-real_group2[20:])/num_patients_all[20:])
        L2_valid_value.append(L2) 

        plt.plot(obs_group2[20:]/num_patients_all[20:], label='Predicted', marker='o', linestyle='-')
        plt.plot(real_group2[20:]/num_patients_all[20:], label='Real', marker='x', linestyle='--')
        plt.legend()
        filename = os.path.join(current_output_dir,f"valid_L2.png")
        plt.savefig(filename)
        plt.show()
        plt.close()

        with open(os.path.join(current_output_dir, "l2_train_values.txt"), "a") as f:
            for value in L2_train_value:
                f.write(f"{value}\n")
    
        with open(os.path.join(current_output_dir, "l2_valid_values.txt"), "a") as f:
            for value in L2_valid_value:
                f.write(f"{value}\n")

        model_checkpoint_path = os.path.join(current_output_dir, 'model_checkpoint.h5')
        model.save(model_checkpoint_path)

    return break_times

def create_unique_dir(base_dir, hyperparams):
    """ Creates a unique directory path for a given set of hyperparameters. """
    dir_name = '__'.join(f"{key}_{value}" for key, value in hyperparams.items())
    unique_dir = os.path.join(base_dir, dir_name)
    if not os.path.exists(unique_dir):
        os.makedirs(unique_dir)
    return unique_dir

if __name__ == '__main__':
    base_output_dir = "./vaccine/"
    import time
    t1 = time.time()
    print('Training start ========================')

    for hyperparams in hyperparameters_grid:
        current_output_dir = create_unique_dir(base_output_dir, hyperparams)
        print(f"Training with hyperparameters: {hyperparams}")
        print(f"Output directory: {current_output_dir}")
        train(hyperparams, current_output_dir)

    np.seterr(divide = 'ignore') 

    print('Training finished =======================')
    elapsed_time_seconds = time.time() - t1
    elapsed_time_hours = elapsed_time_seconds / 3600
    print(f'Elapsed time (hrs): {elapsed_time_hours:.2f}')





