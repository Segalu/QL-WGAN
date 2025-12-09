import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import matplotlib.pyplot as plt

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
print(f"✅ RTX 4070: {gpus}")

GRID_SIZE = 8
N_CRITIC = 5
LAMBDA_GP = 10.0


class ShotDataset:
    def __init__(self, num_sequences=500):
        self.states, self.real_policies = self.generate_shot_sequences(num_sequences)

    def generate_shot_sequences(self, num_sequences):
        print(f"Generating {num_sequences} sequences...")
        states, policies = [], []
        for _ in tqdm(range(num_sequences)):
            shots = self.generate_game()
            for i in range(min(10, len(shots))):
                state_shots = shots[:i + 1]
                state = self.state_to_input(state_shots)
                policy = self.create_policy_map(state_shots)
                states.append(state)
                policies.append(policy)
        return np.array(states, dtype=np.float32), np.array(policies, dtype=np.float32)

    def generate_game(self):
        player_grid = np.random.randint(0, 4, (GRID_SIZE, GRID_SIZE)).astype(np.float32)
        player_grid[player_grid == 0] = 0
        shots = []
        while len(shots) < 12:
            row, col = np.random.randint(0, GRID_SIZE, 2)
            if not any(s['row'] == row and s['col'] == col for s in shots):
                hit = player_grid[row, col] > 0
                shots.append({'row': row, 'col': col, 'hit': hit})
        return shots

    def state_to_input(self, shots):
        shots_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        hits_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for shot in shots:
            shots_grid[shot['row'], shot['col']] = 1.0
            if shot['hit']:
                hits_grid[shot['row'], shot['col']] = 1.0
        return np.stack([shots_grid, hits_grid], axis=-1)

    def create_policy_map(self, shots):
        """SMART EXPERT POLICY: Hunt hits + explore smartly"""
        policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        available = []

        # Calculate smart scores for each available position
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if not any(s['row'] == r and s['col'] == c for s in shots):
                    score = self.expert_position_score(shots, r, c)
                    policy[r, c] = score
                    available.append((r, c))

        # Normalize to probability distribution
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        return policy

    def expert_position_score(self, shots, r, c):
        """Expert scoring: hunt hits + prefer edges"""
        score = 1.0  # Base score

        # 1. HUNT NEAR HITS (highest priority)
        recent_hits = [s for s in shots[-5:] if s['hit']]
        for hit in recent_hits:
            dist = np.sqrt((r - hit['row']) ** 2 + (c - hit['col']) ** 2)
            score += 10.0 / (dist + 1)  # Strong hunt bonus

        # 2. EDGE PREFERENCE (ships often on edges)
        edge_bonus = min(r, c, GRID_SIZE - 1 - r, GRID_SIZE - 1 - c)
        score += edge_bonus * 0.5

        # 3. EARLY EXPLORATION (spread out first)
        if len(shots) < 8:
            center_dist = np.sqrt((r - 4) ** 2 + (c - 4) ** 2)
            score += center_dist * 0.3

        return max(0.01, score)  # Minimum score

def build_shot_generator():
    inputs = keras.Input(shape=(GRID_SIZE, GRID_SIZE, 2), dtype=tf.float32)
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(GRID_SIZE * GRID_SIZE)(x)
    x = layers.Softmax()(x)
    outputs = layers.Reshape((GRID_SIZE, GRID_SIZE))(x)
    return keras.Model(inputs, outputs, name="shot_generator")


def build_shot_discriminator():
    inputs = keras.Input(shape=(GRID_SIZE, GRID_SIZE, 3), dtype=tf.float32)
    x = layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs, name="shot_discriminator")


@tf.function
def gradient_penalty(discriminator, real, fake):
    batch_size = tf.shape(real)[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0, dtype=tf.float32)
    interpolated = epsilon * tf.cast(real, tf.float32) + (1 - epsilon) * tf.cast(fake, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    grads = tape.gradient(pred, [interpolated])[0]
    grads = tf.cast(grads, tf.float32)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-8)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp


def train_shot_gan(epochs=30, batch_size=128):  # RTX 4070 optimized
    dataset = ShotDataset(num_sequences=800)
    states, real_policies = dataset.states, dataset.real_policies
    real_policies_3d = real_policies[..., np.newaxis]

    generator = build_shot_generator()
    discriminator = build_shot_discriminator()

    g_optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    d_optimizer = keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    print("🚀 RTX 4070 WGAN-GP Training...")
    for epoch in tqdm(range(epochs), desc="GPU Epochs"):
        indices = np.random.permutation(len(states))
        states_shuf = states[indices]
        policies_shuf = real_policies_3d[indices]

        for i in range(0, len(states), batch_size):
            batch_states = states_shuf[i:i + batch_size]
            batch_policies = policies_shuf[i:i + batch_size]

            # Discriminator N_CRITIC times
            for _ in range(N_CRITIC):
                with tf.GradientTape() as d_tape:
                    fake_policies = generator(batch_states, training=True)
                    fake_policies_3d = fake_policies[..., tf.newaxis]

                    real_input = tf.concat([batch_states, batch_policies], -1)
                    fake_input = tf.concat([batch_states, fake_policies_3d], -1)

                    d_real = discriminator(real_input, training=True)
                    d_fake = discriminator(fake_input, training=True)

                    gp = gradient_penalty(discriminator, real_input, fake_input)
                    d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + LAMBDA_GP * gp

                d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            # Generator
            with tf.GradientTape() as g_tape:
                fake_policies = generator(batch_states, training=True)
                fake_policies_3d = fake_policies[..., tf.newaxis]
                fake_input = tf.concat([batch_states, fake_policies_3d], -1)
                g_loss = -tf.reduce_mean(discriminator(fake_input, training=False))

            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    return generator, discriminator


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    generator, discriminator = train_shot_gan(epochs=30, batch_size=128)

    generator.save_weights("models/shot_generator_rtx4070.weights.h5")
    print("✅ RTX 4070 STABLE GAN SAVED!")

    # Test
    test_state = np.random.uniform(0, 1, (1, GRID_SIZE, GRID_SIZE, 2)).astype(np.float32)
    test_policy = generator.predict(test_state, verbose=0)[0]
    plt.imshow(test_policy, cmap='hot')
    plt.title('RTX 4070 Shot Policy')
    plt.savefig("models/shot_policy_rtx4070.png")
    plt.show()