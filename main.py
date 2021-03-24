# This is the project repo for Crop Yield Forecase

# Imports required for the project
import numpy as np
import time

import tensorflow as tf

# IDE show a runtime error, to ignore the error and run this line is used
np.seterr(divide='ignore', invalid='ignore')


def import_dataset():
    dataset = np.load('/home/isuruthiwa/Downloads/soybean_samples.npz')
    return dataset['data']


def clean_dataset(dataset):
    cleaned_dataset = X[X[:, 1] <= 2017]
    cleaned_dataset = cleaned_dataset[:, 3:]
    return cleaned_dataset


def calc_basic_stats(dataset):
    M = np.mean(dataset, axis=0, keepdims=True)
    S = np.std(dataset, axis=0, keepdims=True)
    return M, S


def run_model(E_t1, E_t2, E_t3, E_t4, E_t5, E_t6, S_t1, S_t2, S_t3, S_t4, S_t5, S_t6, S_t7, S_t8, S_t9, S_t10, S_t11,
              P_t, Ybar, f, is_training, num_units, num_layers, dropout):
    return



def process_data():
    with tf.device('/cpu:0'):
        E_t1 = tf.keras.Input(shape=(None, 52, 1), dtype=tf.dtypes.float32, name='E_t1')
        E_t2 = tf.keras.Input(shape=(None, 52, 1), dtype=tf.dtypes.float32, name='E_t2')
        E_t3 = tf.keras.Input(shape=(None, 52, 1), dtype=tf.dtypes.float32, name='E_t3')
        E_t4 = tf.keras.Input(shape=(None, 52, 1), dtype=tf.dtypes.float32, name='E_t4')
        E_t5 = tf.keras.Input(shape=(None, 52, 1), dtype=tf.dtypes.float32, name='E_t5')
        E_t6 = tf.keras.Input(shape=(None, 52, 1), dtype=tf.dtypes.float32, name='E_t6')

        S_t1 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t1')
        S_t2 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t2')
        S_t3 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t3')
        S_t4 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t4')
        S_t5 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t5')
        S_t6 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t6')
        S_t7 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t7')
        S_t8 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t8')
        S_t9 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t9')
        S_t10 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t10')
        S_t11 = tf.keras.Input(shape=(None, 10, 1), dtype=tf.dtypes.float32, name='S_t11')

        P_t = tf.keras.Input(shape=(None, 16, 1), dtype=tf.dtypes.float32, name='P_t')

        Ybar = tf.keras.Input(shape=[None, 5, 1], dtype=tf.dtypes.float32, name='Ybar')

        Y_t = tf.keras.Input(shape=[None, 1], dtype=tf.dtypes.float32, name='Y_t')

        Y_t_2 = tf.keras.Input(shape=[None, 4], dtype=tf.dtypes.float32, name='Y_t_2')

        is_training = tf.keras.Input(dtype=tf.dtypes.bool)
        lr = tf.keras.Input(shape=[], dtype=tf.dtypes.float32, name='learning_rate')
        dropout = tf.keras.Input(tf.float32, name='dropout')

        f = 3





X = import_dataset()
X_tr = clean_dataset(X)
M, S = calc_basic_stats(X_tr)

print(X.shape)
print(X_tr.shape)

X[:, 3:] = (X[:, 3:] - M) / S

X = np.nan_to_num(X)

index_low_yield = X[:, 2] < 5
print('low yield observations', np.sum(index_low_yield))

print(X[index_low_yield][:, 1])
X = X[np.logical_not(index_low_yield)]

Index = X[:, 1] == 2018  # validation year
print(Index.shape)

print('Std %.2f and mean %.2f  of test ' % (np.std(X[Index][:, 2]), np.mean(X[Index][:, 2])))
print("train data", np.sum(np.logical_not(Index)))
print("test data", np.sum(Index))

Max_it = 350000  # 150000 could also be used with early stopping
learning_rate = 0.0003  # Learning rate
batch_size_tr = 25  # training batch size
le = 0.0  # Weight of loss for prediction using times before final time steps
l = 1.0  # Weight of loss for prediction using final time step
num_units = 64  # Number of hidden units for LSTM cells
num_layers = 1  # Number of layers of LSTM cell
