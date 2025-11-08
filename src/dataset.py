import tensorflow as tf
import numpy as np
import gdown


# training datase
!gdown --id 1sasl3dhaHDOIS294jEfOKXqhpEBhRfd1
!gdown --id 1B3df03o4HnNwD8BJ-fFWiRVO_0ZXXwec
train_x = np.load("train_x.npy", allow_pickle=True)
train_y = np.load("train_y.npy", allow_pickle=True)
def data_generator(train_x, train_y):
    for tx_file, ty_file in zip(train_x, train_y):
        yield (tx_file, ty_file)
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_x, train_y),
    output_signature=(
        tf.TensorSpec(shape=(24000,), dtype=tf.float32),
        tf.TensorSpec(shape=(2, 24000,), dtype=tf.float32)
    )
)
SHUFFLE_BUFFER_SIZE = 5000
dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
batched_dataset = dataset.batch(5, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
for x_batch, y_batch in batched_dataset:
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    break

# validation dataset
!gdown --id 1234shglaHGSO4dfhcnkj49430KD4iJUg
!gdown --id 1vzqD7-tBdjfaddjgkPQLzB6SBVGVkr0n
val_x = np.load("val_x.npy", allow_pickle=True)
val_y = np.load("val_y.npy", allow_pickle=True)
def data_generator(val_x, val_y):
    for vx_file, vy_file in zip(val_x, val_y):
        yield (vx_file, vy_file)
val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_x, val_y),
    output_signature=(
        tf.TensorSpec(shape=(24000,), dtype=tf.float32),
        tf.TensorSpec(shape=(2, 24000,), dtype=tf.float32)
    )
)
val_batched_dataset = val_dataset.batch(5, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
for x_batch, y_batch in val_batched_dataset:
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    break


# test dataset
!gdown --id 2kdsgfakhashflahdfhd-aIXTSoG9jWbE
!gdown --id 2akdhglio34aKFKSdjgafahjkvyXoucUl
test_x = np.load("test_x.npy", allow_pickle=True)
test_y = np.load("test_y.npy", allow_pickle=True)
def data_generator(test_x, test_y):
    for tx_file, ty_file in zip(test_x, test_y):
        yield (tx_file, ty_file)
test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_x, test_y),
    output_signature=(
        tf.TensorSpec(shape=(24000,), dtype=tf.float32),
        tf.TensorSpec(shape=(2, 24000,), dtype=tf.float32)
    )
)
test_batched_dataset = test_dataset.batch(5, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
for x_batch, y_batch in test_batched_dataset:
    print("x_batch shape:", x_batch.shape)
    print("y_batch shape:", y_batch.shape)
    break