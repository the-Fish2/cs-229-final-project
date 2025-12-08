# %% [markdown]
# # Conv1D Model Training Notebook

# %% [code]
# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl
import function_utils  # your module to load data

# Check available devices
print("Devices:", tf.config.list_physical_devices())

# %% [markdown]
# ## LOAD DATA

# %% [code]
# Load X and y
X = function_utils.load_sympy_to_np_array('SE_comp_1_3.pkl')
y = np.loadtxt('y_comp_1_3.npy', delimiter=',')

# Normalize X per sample
X_norm = np.zeros_like(X, dtype=np.float32)
for i in range(X.shape[0]):
    std = np.std(X[i])
    if std == 0:
        X_norm[i] = 0
    else:
        X_norm[i] = (X[i] - np.mean(X[i])) / std

# Shuffle dataset
n_samples = X_norm.shape[0]
perm = np.random.permutation(n_samples)
X_shuffled = X_norm[perm]
y_shuffled = y[perm]

# Split train/validation
train_frac = 0.8
n_train = int(train_frac * n_samples)

X_train = X_shuffled[:n_train]
y_train = y_shuffled[:n_train]
X_val   = X_shuffled[n_train:]
y_val   = y_shuffled[n_train:]

# Add channel dimension for Conv1D
X_train_in = X_train[..., np.newaxis]  # (batch, 5000, 1)
X_val_in   = X_val[..., np.newaxis]

# Targets as float32 and shape (batch,1)
y_train_in = y_train[:, np.newaxis].astype(np.float32)
y_val_in   = y_val[:, np.newaxis].astype(np.float32)

# %% [markdown]
# ## BUILD MODEL

# %% [code]
SAMPLES = X_train_in.shape[1]

model = tf.keras.Sequential([
    tfl.Input(shape=(SAMPLES, 1)),

    tfl.SeparableConv1D(32, 7, padding="same", dilation_rate=1),
    tfl.LeakyReLU(alpha=0.1),
    tfl.Dropout(0.1),

    tfl.SeparableConv1D(32, 7, padding="same", dilation_rate=2),
    tfl.LeakyReLU(alpha=0.1),
    tfl.Dropout(0.1),

    tfl.SeparableConv1D(32, 7, padding="same", dilation_rate=4),
    tfl.LeakyReLU(alpha=0.1),
    tfl.Dropout(0.1),

    tfl.SeparableConv1D(32, 7, padding="same", dilation_rate=8),
    tfl.LeakyReLU(alpha=0.1),
    tfl.Dropout(0.1),

    tfl.Conv1D(16, 1, padding="same"),
    tfl.LeakyReLU(alpha=0.1),

    tfl.GlobalAveragePooling1D(),
    tfl.Dense(1),  # regression output
])

# %% [markdown]
# ## COMPILE MODEL

# %% [code]
# Optimizer with gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss='mae',  # mean absolute error
    metrics=[]
)

model.summary()

# %% [markdown]
# ## TRAIN MODEL

# %% [code]
history = model.fit(
    X_train_in, y_train_in,
    validation_data=(X_val_in, y_val_in),
    epochs=50,
    batch_size=64,
    shuffle=True
)

# %% [markdown]
# ## PLOT LOSS

# %% [code]
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## EVALUATE & PREDICT

# %% [code]
# Example prediction
ind = 0
plt.plot(function_utils.XS, X_val[ind])
plt.title(f"True y: {y_val[ind]}")
plt.show()

y_pred_val = model.predict(X_val_in[:10])
print("Predictions:", y_pred_val.flatten())
