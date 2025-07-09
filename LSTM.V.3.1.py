import pandas as pd
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Multiply, Lambda, Input, Layer, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.feature_selection import mutual_info_regression

##############################################################################
######################### GPU CONFIGURATION ##################################
##############################################################################

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Devices:", tf.config.list_physical_devices())

###########################################################################
######################## CONSISTENCY IN PREDICTIONS #######################
###########################################################################

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISM'] = '1'

random.seed(seed_value)                
np.random.seed(seed_value)            
tf.random.set_seed(seed_value) 
tf.config.experimental.enable_op_determinism()

#############################################################################
######################## DATA IMPORT AND MANIPULATION #######################
#############################################################################

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "KO.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File is not on the path: {file_path}")

df_model = pd.read_csv(file_path)
df_model = df_model.set_index("DATE")
df_model.index = pd.to_datetime(df_model.index)

# Feature engineering
df_model["CLOSE/CLOSE"] = (df_model["PX_LAST"] / df_model["PX_LAST"].shift(1)) - 1
df_model["OPEN/OPEN"] = (df_model["PX_OPEN"] / df_model["PX_OPEN"].shift(1)) - 1
df_model["HIGH/HIGH"] = (df_model["PX_HIGH"] / df_model["PX_HIGH"].shift(1)) - 1
df_model["LOW/LOW"] = (df_model["PX_LOW"] / df_model["PX_LOW"].shift(1)) - 1
df_model["VOLUME/VOLUME"] = (df_model["PX_VOLUME"] / df_model["PX_VOLUME"].shift(1)) - 1

df_model["TARGET"] = (df_model["PX_LAST"] / df_model["PX_OPEN"]) - 1
df_model["HIGH/OPEN"] = (df_model["PX_HIGH"] / df_model["PX_OPEN"]) - 1
df_model["LOW/OPEN"] = (df_model["PX_LOW"] / df_model["PX_OPEN"]) - 1
df_model["CLOSE/HIGH"] = (df_model["PX_LAST"] / df_model["PX_HIGH"]) - 1
df_model["CLOSE/LOW"] = (df_model["PX_LAST"] / df_model["PX_LOW"]) - 1
df_model["MAXRANGE"] = (df_model["PX_HIGH"] / df_model["PX_LOW"]) - 1
df_model["OVERNIGHT"] = (df_model["PX_OPEN"].shift(-1) / df_model["PX_LAST"]) - 1
df_model["OVERNIGHT"] = df_model["OVERNIGHT"].shift(1)

df_model = df_model.drop(columns=["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME"])
df_model = df_model.dropna()

###########################################
############ DATA FRAME SET UP ############
###########################################

df_model_original = df_model.copy()
df_model['TARGET'] = df_model['TARGET'].shift(-1)
df_model = df_model.dropna()

#######################################################################
###################### CONFIGURABLE PARAMETERS ########################
#######################################################################

# Quantum Parameters
quantum_units = 128  # Dimensions in quantum attention layer. Increase for complex patterns (risk overfitting)
hamiltonian_matrix = tf.eye(quantum_units, dtype=tf.float32)  # Quantum system representation
phase_lambda = 0.0001  # MAE weight in loss. Increase for robust outlier handling
quantum_noise_level = 0.0001  # Training noise std. Increase for stronger regularization
wave_init_scale = 0.1  # Wave initializer magnitude. Larger values = bigger initial weights
energy_levels_list = [64, 128, 256]  # Hidden unit candidates for quantum optimization
energy_probabilities = [0.3, 0.4, 0.3]  # Selection probs before sqrt (sum should be 1)

# Architecture Parameters
lstm_layers = 2  # Stacked LSTM layers. More layers = higher abstraction (risk overfitting)
lstm_units = 128  # Units per LSTM layer. Increase for model capacity
activation_function = 'tanh'  # LSTM activation. Try 'relu' for positive outputs
output_activation = 'linear'  # Final activation. Keep linear for regression
bidirectional_merge_mode = 'concat'  # Bidirectional merging: 'sum', 'mul' or 'ave'
weight_initializer = 'glorot_uniform'  # Weight init: 'he_normal' for ReLU
phase_initializer = 'random_normal'  # Phase init for quantum layers

# Training Parameters
learning_rate = 0.005  # Initial learning rate. Increase for faster training
fine_tune_learning_rate = 0.0005  # Post-freeze learning rate. Keep lower
training_epochs = 200  # Initial training cycles. Increase for convergence
fine_tune_epochs = 200  # Fine-tuning cycles. Monitor for overfitting
batch_size = 204800  # Samples per batch. Smaller = noisier updates
clip_value = 1.1  # Gradient clipping. Decrease if gradients explode
alpha_init = 0.95  # Student-teacher mix. Higher = more student influence

# Regularization & Data Parameters
temporal_curvature_weight = 0.15  # Temporal smoothness. Increase for less jitter
gradient_penalty_weight = 0.1  # WGAN-style penalty (not used in current setup)
wormhole_aug_tolerance = 0.1  # Similarity threshold. Lower = stricter matches
mi_conv_window_size = 7  # MI smoothing window. Larger = smoother lags
min_lookback_window = 7  # Minimum sequence length. Increase for longer memory
lookback_lag_multiplier = 0.5  # Optimal lag scaling. Adjust with MI results
quantile_min = 0.0001  # Robust scaling lower bound. Decrease for outlier resistance
quantile_max = 0.999  # Upper scaling bound. Increase to ignore upper outliers

#######################################################################
######################## QUANTUM COMPONENTS ###########################
#######################################################################

class QuantumMinMaxScaler:
    def __init__(self):
        self.data_min_ = None
        self.data_max_ = None
        self.target_min_ = None
        self.target_max_ = None

    def fit_transform(self, X, y):
        self.data_min_ = np.quantile(X, quantile_min, axis=0)
        self.data_max_ = np.quantile(X, quantile_max, axis=0)
        X_scaled = (X - self.data_min_) / (self.data_max_ - self.data_min_ + 1e-8)
        
        self.target_min_ = np.quantile(y, quantile_min)
        self.target_max_ = np.quantile(y, quantile_max)
        y_scaled = (y - self.target_min_) / (self.target_max_ - self.target_min_ + 1e-8)
        
        return X_scaled, y_scaled

    def inverse_transform(self, y_scaled):
        return y_scaled * (self.target_max_ - self.target_min_) + self.target_min_

class QuantumAttention(Layer):
    def build(self, input_shape):
        self.phase_weights = self.add_weight(
            shape=(input_shape[-1], quantum_units),
            initializer=weight_initializer,
            dtype=tf.float32,
            trainable=True
        )
        
    def call(self, inputs):
        phase_shift = tf.tensordot(inputs, self.phase_weights, axes=1)
        attention_scores = tf.tensordot(phase_shift, hamiltonian_matrix, axes=1)
        attention_weights = tf.nn.softmax(attention_scores)
        return attention_weights * inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        return super().get_config()

class WaveDynamicsInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        x = np.linspace(0, 2*np.pi, shape[0])
        y = np.linspace(0, 2*np.pi, shape[1])
        yy, xx = np.meshgrid(y, x)
        return tf.cast(wave_init_scale * np.sin(xx + yy), dtype=dtype)

class PhaseFieldDense(Layer):
    def build(self, input_shape):
        self.phase = self.add_weight(
            shape=(input_shape[-1],),
            initializer=phase_initializer,
            dtype=tf.float32,
            trainable=True
        )
        
    def call(self, inputs):
        return inputs * tf.math.sin(self.phase)
    
    def get_config(self):
        return super().get_config()

class TensorQuantumNoise(Layer):
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(tf.shape(inputs)) * quantum_noise_level
            return inputs + noise
        return inputs
    
    def get_config(self):
        return super().get_config()

def quantum_phase_loss(y_true, y_pred):
    error = y_true - y_pred
    loss = tf.reduce_mean(tf.square(error)) + phase_lambda * tf.reduce_mean(tf.abs(error))
    return loss

#######################################################################
###################### OPTIMIZATION & REGULARIZATION ##################
#######################################################################

def quantum_parameter_optimization():
    probabilities = np.sqrt(energy_probabilities)
    probabilities /= np.sum(probabilities)
    return int(np.random.choice(energy_levels_list, p=probabilities))

lstm_units = quantum_parameter_optimization()

def temporal_curvature_regularizer(y_pred):
    laplacian = tf.reduce_mean(tf.square(y_pred[:, 2:] - 2*y_pred[:, 1:-1] + y_pred[:, :-2]))
    return temporal_curvature_weight * laplacian

def wormhole_augmentation(X_sequences, Y_sequences):
    augmented_X, augmented_Y = [], []
    for idx, seq in enumerate(X_sequences):
        similarity_mask = np.all(np.isclose(seq[-1], X_sequences[:, -1], 
                                         atol=wormhole_aug_tolerance), axis=1)
        similar_indices = np.where(similarity_mask)[0]
        
        if len(similar_indices) > 0:
            wormhole_idx = np.random.choice(similar_indices)
            augmented_seq = np.concatenate([seq, X_sequences[wormhole_idx]], axis=0)[-min_lookback_window:]
            augmented_X.append(augmented_seq)
            augmented_Y.append(Y_sequences[wormhole_idx])
    
    if len(augmented_X) > 0:
        return np.stack(augmented_X), np.array(augmented_Y)
    else:
        return np.zeros((0, *X_sequences.shape[1:])), np.zeros((0,))

#######################################################################
######################### DATA PREPROCESING ###########################
#######################################################################

X = df_model.drop(columns = ['TARGET'])
Y = df_model['TARGET']
n_features = X.shape[1]

mi_scores = mutual_info_regression(X, Y)
optimal_lag = np.argmax(np.convolve(mi_scores, 
                                  np.ones(mi_conv_window_size)/mi_conv_window_size, 
                                  mode='valid')) + 1
lookback_window = int(max(min_lookback_window, optimal_lag * lookback_lag_multiplier))

scaler = QuantumMinMaxScaler()
X_scaled, Y_scaled = scaler.fit_transform(X.values, Y.values)

X_sequences = []
Y_sequences = []
for i in range(len(X_scaled) - lookback_window + 1):
    X_sequences.append(X_scaled[i:i+lookback_window])
    Y_sequences.append(Y_scaled[i+lookback_window-1])

X_train = np.array(X_sequences)
Y_train = np.array(Y_sequences)

if len(X_train) > 0:
    X_aug, Y_aug = wormhole_augmentation(X_train, Y_train)
    if len(X_aug) > 0:
        X_train = np.concatenate([X_train, X_aug], axis=0)
        Y_train = np.concatenate([Y_train, Y_aug])

#######################################################################
######################## HYBRID MODEL ARCHITECTURE ####################
#######################################################################

def create_models():
    input_layer = Input(shape=(lookback_window, n_features))
    x = TensorQuantumNoise()(input_layer)
    
    # Student (Causal)
    lstm_out = LSTM(lstm_units, return_sequences=True, activation=activation_function)(x)
    for _ in range(lstm_layers-1):
        lstm_out = LSTM(lstm_units, return_sequences=True, activation=activation_function)(lstm_out)
    
    # Teacher (Non-Causal)
    teacher_out = Bidirectional(LSTM(lstm_units), 
                              merge_mode=bidirectional_merge_mode)(x)
    
    # Quantum Attention
    attention = QuantumAttention()(lstm_out)
    context = Multiply()([lstm_out, attention])
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    
    # Dimensional alignment
    phase_context = Dense(lstm_units)(context_vector)
    phase_teacher = Dense(lstm_units)(teacher_out)
    
    # Learnable combination
    alpha = tf.Variable(alpha_init, trainable=True)
    combined = alpha * phase_context + (1-alpha) * phase_teacher
    
    # Final output
    output = Dense(1, activation=output_activation,
                  kernel_initializer=WaveDynamicsInitializer())(combined)
    
    return Model(inputs=input_layer, outputs=output)

model = create_models()

optimizer = Adam(learning_rate=learning_rate, clipvalue=clip_value)
model.compile(optimizer=optimizer, loss=quantum_phase_loss)

#######################################################################
######################## MODEL TRAINING ###############################
#######################################################################

history = model.fit(X_train, Y_train,
                  epochs=training_epochs,
                  batch_size=batch_size,
                  verbose=1)

for layer in model.layers:
    if 'lstm' in layer.name.lower() and not isinstance(layer, Bidirectional):
        layer.trainable = False
        
model.compile(optimizer=Adam(fine_tune_learning_rate), loss=quantum_phase_loss)
history = model.fit(X_train, Y_train,
                  epochs=fine_tune_epochs,
                  batch_size=batch_size,
                  verbose=1)

#######################################################################
####################### FINAL PREDICTION ##############################
#######################################################################

X_next = df_model_original.drop(columns = ['TARGET']).iloc[-lookback_window:]
X_next_scaled = (X_next.values - scaler.data_min_) / (scaler.data_max_ - scaler.data_min_ + 1e-8)
X_next = X_next_scaled.reshape((1, lookback_window, n_features))

prediction_scaled = model.predict(X_next)
final_prediction = scaler.inverse_transform(prediction_scaled)

print(f"\nOut-of-sample prediction for t+1: {final_prediction[0][0]:.4f}")