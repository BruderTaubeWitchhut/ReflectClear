import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from tensorflow_addons.optimizers import AdamW

# Constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 4
EPOCHS = 50
BUFFER_SIZE = 1000
LEARNING_RATE = 0.0002
BETA_1 = 0.5

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Function to apply Spectral Normalization
def spectral_norm(layer):
    return tf.keras.layers.experimental.SpectralNormalization(layer)

# Build the Discriminator with Spectral Normalization
def build_discriminator():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    x = spectral_norm(layers.Conv2D(64, (4, 4), strides=2, padding='same'))(inputs)
    x = layers.LeakyReLU(0.2)(x)
    
    x = spectral_norm(layers.Conv2D(128, (4, 4), strides=2, padding='same'))(x)
    x = layers.InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = spectral_norm(layers.Conv2D(256, (4, 4), strides=2, padding='same'))(x)
    x = layers.InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = spectral_norm(layers.Conv2D(512, (4, 4), strides=2, padding='same'))(x)
    x = layers.InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    return models.Model(inputs=inputs, outputs=x)

# Build the Generator with Instance Normalization
def build_generator():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    
    # Encoder
    x = layers.Conv2D(64, (7, 7), padding='same')(inputs)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    
    # Residual Blocks
    for _ in range(9):
        res = layers.Conv2D(256, (3, 3), padding='same')(x)
        res = layers.InstanceNormalization()(res)
        res = layers.ReLU()(res)
        res = layers.Conv2D(256, (3, 3), padding='same')(res)
        res = layers.InstanceNormalization()(res)
        x = layers.add([x, res])
    
    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = layers.InstanceNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(IMG_CHANNELS, (7, 7), padding='same', activation='tanh')(x)
    return models.Model(inputs=inputs, outputs=x)

# Build models
generator_AtoB = build_generator()
generator_BtoA = build_generator()
discriminator_A = build_discriminator()
discriminator_B = build_discriminator()

# Define optimizers
optimizer_G = AdamW(learning_rate=LEARNING_RATE, beta_1=BETA_1, weight_decay=1e-4)
optimizer_D = AdamW(learning_rate=LEARNING_RATE, beta_1=BETA_1, weight_decay=1e-4)

# Loss functions
adv_loss = tf.keras.losses.MeanSquaredError()
cycle_loss = tf.keras.losses.MeanAbsoluteError()

# Training step
@tf.function
def train_step(real_A, real_B):
    with tf.GradientTape(persistent=True) as tape:
        fake_B = generator_AtoB(real_A)
        fake_A = generator_BtoA(real_B)
        
        cycled_A = generator_BtoA(fake_B)
        cycled_B = generator_AtoB(fake_A)
        
        disc_real_A = discriminator_A(real_A)
        disc_fake_A = discriminator_A(fake_A)
        disc_real_B = discriminator_B(real_B)
        disc_fake_B = discriminator_B(fake_B)
        
        gen_AB_loss = adv_loss(tf.ones_like(disc_fake_B), disc_fake_B)
        gen_BA_loss = adv_loss(tf.ones_like(disc_fake_A), disc_fake_A)
        total_cycle_loss = cycle_loss(real_A, cycled_A) + cycle_loss(real_B, cycled_B)
        total_gen_loss = gen_AB_loss + gen_BA_loss + 10 * total_cycle_loss
        
        disc_A_loss = adv_loss(tf.ones_like(disc_real_A), disc_real_A) + adv_loss(tf.zeros_like(disc_fake_A), disc_fake_A)
        disc_B_loss = adv_loss(tf.ones_like(disc_real_B), disc_real_B) + adv_loss(tf.zeros_like(disc_fake_B), disc_fake_B)
    
    gen_gradients = tape.gradient(total_gen_loss, generator_AtoB.trainable_variables + generator_BtoA.trainable_variables)
    disc_A_gradients = tape.gradient(disc_A_loss, discriminator_A.trainable_variables)
    disc_B_gradients = tape.gradient(disc_B_loss, discriminator_B.trainable_variables)
    
    optimizer_G.apply_gradients(zip(gen_gradients, generator_AtoB.trainable_variables + generator_BtoA.trainable_variables))
    optimizer_D.apply_gradients(zip(disc_A_gradients, discriminator_A.trainable_variables))
    optimizer_D.apply_gradients(zip(disc_B_gradients, discriminator_B.trainable_variables))

# Testing function
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (image / 127.5) - 1
    return tf.expand_dims(image, axis=0)

def denormalize_image(image):
    return (image + 1) / 2

def test_reflection_removal(image_path):
    input_image = preprocess_image(image_path)
    reflection_removed_image = generator_AtoB(input_image, training=False)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(denormalize_image(tf.squeeze(input_image)))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Reflection Removed")
    plt.imshow(denormalize_image(tf.squeeze(reflection_removed_image)))
    plt.axis('off')
    plt.show()
