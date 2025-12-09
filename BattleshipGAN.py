import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
GRID_SIZE = 8
CELL_SIZE = 60
GRID_PADDING = 10

COLORS = {
    'bg': (26, 26, 46),
    'bg_light': (22, 33, 62),
    'primary': (15, 52, 96),
    'accent': (233, 69, 96),
    'water': (74, 144, 226),
    'ship': (44, 62, 80),
    'hit': (231, 76, 60),
    'miss': (149, 165, 166),
    'text': (236, 240, 241),
    'button': (52, 152, 219),
    'button_hover': (41, 128, 185),
    'train_button': (155, 89, 182),
    'success': (46, 204, 113),
    'warning': (241, 196, 15)
}

# Ships configuration
SHIPS = [
    {'name': 'Carrier', 'size': 4},
    {'name': 'Battleship', 'size': 3},
    {'name': 'Destroyer', 'size': 2}
]


class BattleshipGAN:
    def __init__(self, grid_size=8):
        self.GRID_SIZE = grid_size
        self.SHIPS = SHIPS

        # Game grids
        self.player_grid = np.zeros((grid_size, grid_size), dtype=int)
        self.ai_grid = np.zeros((grid_size, grid_size), dtype=int)
        self.player_shots = []
        self.ai_shots = []

        # GAN models
        self.generator = None
        self.discriminator = None
        self.gan_loaded = False

        # Stats (no Q-table, just game stats)
        self.games_played = 0
        self.ai_stats = {'hits': 0, 'misses': 0, 'accuracy': 0.0}

    def build_shot_generator(self):
        """EXACT MATCH with hunter training"""
        return tf.keras.Sequential([
            tf.keras.Input((self.GRID_SIZE, self.GRID_SIZE, 2)),
            tf.keras.layers.Conv2D(64, 3, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(128, 3, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dense(self.GRID_SIZE * self.GRID_SIZE),
            tf.keras.layers.Softmax(),
            tf.keras.layers.Reshape((self.GRID_SIZE, self.GRID_SIZE))
        ])

    def load_shot_hunter_gan(self, gen_path='models/shot_generator_hunter.h5'):
        """Load PERFECT ship hunter GAN"""
        try:
            print("🎯 Loading PERFECT SHIP HUNTER...")
            self.shot_generator = self.build_shot_generator()
            self.shot_generator.load_weights(gen_path)
            self.shot_generator.trainable = False
            self.shot_gan_loaded = True
            self.shot_gan_type = "hunter"  # Mark as hunter
            print("✅ SHIP HUNTER LOADED! 🎯")
            return True
        except Exception as e:
            print(f"❌ Hunter load failed: {e}")
            self.shot_gan_loaded = False
            return False

    def build_shot_discriminator(self):
        return keras.Sequential([
            layers.Input(shape=(self.GRID_SIZE, self.GRID_SIZE, 3)),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name='shot_discriminator')

    def load_shot_gan_models(self, shot_gen_path='models/shot_generator.weights.h5',
                             shot_disc_path='models/shot_discriminator.weights.h5'):
        try:
            print("Loading Shot Policy GAN...")
            self.shot_generator = self.build_shot_generator()
            self.shot_discriminator = self.build_shot_discriminator()

            self.shot_generator.load_weights(shot_gen_path)
            self.shot_discriminator.load_weights(shot_disc_path)

            self.shot_generator.trainable = False
            self.shot_discriminator.trainable = False
            self.shot_gan_loaded = True
            print("✅ Shot GAN loaded!")
            return True
        except Exception as e:
            print(f"❌ Shot GAN error: {e}")
            self.shot_gan_loaded = False
            return False

    def load_stable_shot_gan(self, gen_path='models/shot_generator_rtx4070.weights.h5'):
        """Load the trained WGAN-GP shot policy generator for smart hunting"""
        try:
            print("Loading stable WGAN-GP ship hunter GAN generator...")
            self.shot_generator = self.build_shot_generator()  # Must match training architecture
            self.shot_generator.load_weights(gen_path)
            self.shot_generator.trainable = False
            self.shot_gan_loaded = True
            print("✅ Hunter GAN generator loaded successfully!")
            return True
        except Exception as e:
            print(f"Failed to load hunter GAN generator: {e}")
            self.shot_gan_loaded = False
            return False

    def ai_choose_shot_hunter(self, shots):
        """GAN‑driven hunter: GAN decides, heuristic only refines around hits."""
        taken = {(s['row'], s['col']) for s in shots}
        available = [(r, c)
                     for r in range(self.GRID_SIZE)
                     for c in range(self.GRID_SIZE)
                     if (r, c) not in taken]
        if not available:
            return None

        # 1) GAN probabilities for all cells
        if self.shot_gan_loaded:
            state = self.state_to_input(shots)
            prob_map = self.shot_generator.predict(state, verbose=0)[0]  # (8,8)
        else:
            prob_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)

        # Mask already-shot cells
        gan_scores = np.array([prob_map[r, c] for (r, c) in available], dtype=np.float32)
        gan_scores = np.clip(gan_scores, 1e-6, None)

        # 2) Small heuristic boost ONLY near hits (not global edge bias)
        all_hits = [s for s in shots if s['hit']]
        heur_boost = np.ones_like(gan_scores, dtype=np.float32)

        if all_hits:
            for i, (r, c) in enumerate(available):
                bonus = 0.0
                for h in all_hits:
                    dist = abs(r - h['row']) + abs(c - h['col'])
                    if dist == 1:
                        bonus += 3.0
                    elif dist == 2:
                        bonus += 1.0
                heur_boost[i] += bonus

        # 3) GAN is main term, heuristic is mild multiplier
        combined = gan_scores * heur_boost  # multiplicative, GAN shape preserved
        combined_sum = combined.sum()
        if combined_sum <= 0:
            # Fallback: uniform over available
            idx = np.random.choice(len(available))
            return available[idx]

        probs = combined / combined_sum

        # 4) Sample only among top‑K GAN cells to avoid “bottom bias”
        K = min(5, len(available))
        top_idx = np.argpartition(probs, -K)[-K:]
        top_probs = probs[top_idx]
        top_probs /= top_probs.sum()
        chosen_local = np.random.choice(len(top_idx), p=top_probs)
        chosen_idx = top_idx[chosen_local]

        return available[chosen_idx]

    def generate_hit_priority(self, shots, available):
        """Strongly favor shots near all hits"""
        scores = []
        all_hits = [s for s in shots if s['hit']]

        for r, c in available:
            score = 1.0
            for hit in all_hits:
                dist = abs(r - hit['row']) + abs(c - hit['col'])
                if dist == 0:
                    score += 50.0  # very strong boost on exact hit cell unlikely but for completeness
                elif dist == 1:
                    score += 40.0  # very strong bonus to immediate neighbors
                elif dist == 2:
                    score += 20.0  # strong bonus a bit farther out
                elif dist == 3:
                    score += 5.0  # small bonus up to 3 cells away
            scores.append(score)
        return scores

    def generate_pattern_priority(self, shots, available):
        """Prioritize completing ship patterns (horizontal/vertical)"""
        scores = []
        for r, c in available:
            score = 1.0

            # Check horizontal ship patterns
            for ship_len in [2, 3, 4]:
                # Left extension
                if c + ship_len <= self.GRID_SIZE:
                    hit_count = sum(1 for i in range(ship_len) if
                                    any(s['row'] == r and s['col'] == c + i and s['hit'] for s in shots))
                    if hit_count > 0:
                        score += hit_count * 2.0

                # Right extension
                if c - ship_len + 1 >= 0:
                    hit_count = sum(1 for i in range(ship_len) if
                                    any(s['row'] == r and s['col'] == c - i and s['hit'] for s in shots))
                    if hit_count > 0:
                        score += hit_count * 2.0

            # Vertical patterns (same logic)
            for ship_len in [2, 3, 4]:
                if r + ship_len <= self.GRID_SIZE:
                    hit_count = sum(1 for i in range(ship_len) if
                                    any(s['row'] == r + i and s['col'] == c and s['hit'] for s in shots))
                    if hit_count > 0:
                        score += hit_count * 2.0

            scores.append(score)
        return scores

    def hunter_heuristic_scores(self, shots, available):
        """Local around‑hit heuristic ONLY, no edge bias."""
        scores = []
        all_hits = [s for s in shots if s['hit']]

        for r, c in available:
            score = 1.0
            for h in all_hits:
                dist = abs(r - h['row']) + abs(c - h['col'])
                if dist == 1:
                    score += 3.0
                elif dist == 2:
                    score += 1.0
            scores.append(score)

        return scores

    def state_to_input(self, shots):
        """Convert shots log into GAN input state tensor (1, 8, 8, 2)"""
        shots_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        hits_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)

        for shot in shots:
            shots_grid[shot['row'], shot['col']] = 1.0
            if shot['hit']:
                hits_grid[shot['row'], shot['col']] = 1.0

        return np.stack([shots_grid, hits_grid], axis=-1)[None, ...]

    def ai_choose_shot_from_stable_gan(self, shots):
        """Use stable WGAN-GP generator for shot policy"""
        if not hasattr(self, 'shot_gan_loaded') or not self.shot_gan_loaded:
            return self.get_random_valid_shot(shots)

        # Generator input: state only
        state_input = self.state_to_input(shots)
        prob_map = self.shot_generator.predict(state_input, verbose=0)[0]

        # Sample from available positions weighted by GAN probabilities
        taken = {(s['row'], s['col']) for s in shots}
        available = [(r, c) for r in range(self.GRID_SIZE)
                     for c in range(self.GRID_SIZE) if (r, c) not in taken]

        if not available:
            return None

        # Weight by GAN probabilities (minimum 0.01 to avoid zero probs)
        scores = [max(0.01, prob_map[r, c]) for r, c in available]
        probs = np.array(scores) / np.sum(scores)

        idx = np.random.choice(len(available), p=probs)
        return available[idx]

    def ai_choose_shot_from_gan_safe(self, shots):
        """Safe GAN shot selection avoiding repeated shots"""
        try:
            if not hasattr(self, 'shot_gan_loaded') or not self.shot_gan_loaded:
                return self.get_random_valid_shot(shots)

            state_input = self.state_to_input(shots)
            prob_map = self.shot_generator.predict(state_input, verbose=0)[0]

            # Create list of available positions excluding previous shots
            available = []
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if not any(s['row'] == r and s['col'] == c for s in shots):
                        score = prob_map[r, c]
                        available.append((score, r, c))

            if not available:
                # No valid moves left - fallback or raise game over
                return None

            scores = [score for score, _, _ in available]
            total = sum(scores)
            if total > 0:
                probs = [s / total for s in scores]
                idx = np.random.choice(len(available), p=probs)
                _, row, col = available[idx]
                return (row, col)

        except Exception as e:
            print(f"GAN shot error: {e}, falling back to heuristic")

        # Fallback to heuristic ensuring no repeats
        return self.get_random_valid_shot(shots)

    def get_random_valid_shot(self, shots):
        """Safe random shot excluding previous shots"""
        taken = {(s['row'], s['col']) for s in shots}
        available = [(r, c) for r in range(self.GRID_SIZE)
                     for c in range(self.GRID_SIZE)
                     if (r, c) not in taken]
        return random.choice(available) if available else None

    def generate_shot_distribution(self, shots):
        """Generate probability map for next shot"""
        if not hasattr(self, 'shot_gan_loaded') or not self.shot_gan_loaded:
            return None

        state_input = self.state_to_input(shots)
        with tf.device('/CPU:0'):
            prob_map = self.shot_generator.predict(state_input, verbose=0)[0]

        # Mask already shot positions
        shot_positions = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        for shot in shots:
            shot_positions[shot['row'], shot['col']] = 1.0
        prob_map = prob_map * (1 - shot_positions)

        # Renormalize
        prob_map = prob_map / (np.sum(prob_map) + 1e-8)
        return prob_map

    def can_place_ship(self, grid, row, col, size, horizontal):
        if horizontal:
            if col + size > self.GRID_SIZE: return False
            for i in range(size):
                if grid[row][col + i] != 0: return False
        else:
            if row + size > self.GRID_SIZE: return False
            for i in range(size):
                if grid[row + i][col] != 0: return False
        return True

    def place_ship(self, grid, ship_id, row, col, size, horizontal):
        if horizontal:
            for i in range(size):
                grid[row][col + i] = ship_id
        else:
            for i in range(size):
                grid[row + i][col] = ship_id

    def place_ships_randomly(self, grid):
        """Fallback for player ships or when GAN not loaded"""
        grid.fill(0)
        for idx, ship in enumerate(self.SHIPS):
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                horizontal = random.choice([True, False])
                row = random.randint(0, self.GRID_SIZE - 1)
                col = random.randint(0, self.GRID_SIZE - 1)
                if self.can_place_ship(grid, row, col, ship['size'], horizontal):
                    self.place_ship(grid, idx + 1, row, col, ship['size'], horizontal)
                    placed = True
                attempts += 1

    def build_generator(self):
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_dim=100),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1024, activation='relu'),
            layers.Dense(self.GRID_SIZE * self.GRID_SIZE, activation='sigmoid'),
            layers.Reshape((self.GRID_SIZE, self.GRID_SIZE))
        ], name='generator')
        return model

    def build_discriminator(self):
        model = keras.Sequential([
            layers.Conv2D(64, 5, strides=2, padding='same', activation='relu',
                          input_shape=(self.GRID_SIZE, self.GRID_SIZE, 1)),
            layers.Dropout(0.3),
            layers.Conv2D(128, 5, strides=2, padding='same', activation='relu'),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        return model

    def load_gan_models(self, generator_path='models/generator.weights.h5', discriminator_path='models/discriminator.weights.h5'):
        """Load pretrained GAN weights - FIXED version"""
        try:
            print("Loading GAN models...")
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            # Load weights
            self.generator.load_weights(generator_path)
            self.discriminator.load_weights(discriminator_path)

            self.generator.trainable = False
            self.discriminator.trainable = False
            self.gan_loaded = True
            print("✅ GAN models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading GAN: {e}")
            self.gan_loaded = False
            return False

    def generate_ai_board_with_gan(self):
        """Generate challenging AI ship layout using GAN"""
        self.ai_grid.fill(0)

        if not self.gan_loaded:
            print("GAN not loaded, using random placement")
            self.place_ships_randomly(self.ai_grid)
            return

        # Generate multiple candidates
        best_grid = None
        best_score = -1

        for attempt in range(3):  # 3 attempts
            noise = tf.random.normal([1, 100])
            generated = self.generator.predict(noise, verbose=0)[0]
            candidate_grid = self.postprocess_gan_output(generated)
            score = self.evaluate_layout_difficulty(candidate_grid)

            if score > best_score:
                best_score = score
                best_grid = candidate_grid.copy()

        self.ai_grid = best_grid if best_grid is not None else self.place_ships_randomly(self.ai_grid)

    def postprocess_gan_output(self, generated):
        """Convert GAN probabilities to valid ship layout"""
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)

        # Threshold and sort cells by probability
        ship_mask = generated > 0.3
        flat_probs = generated.flatten()
        indices = np.argsort(flat_probs)[::-1]

        ship_cells_needed = sum(ship['size'] for ship in self.SHIPS)
        available_cells = []

        for idx in indices:
            if len(available_cells) >= ship_cells_needed:
                break
            row, col = divmod(idx, self.GRID_SIZE)
            if ship_mask[row, col]:
                available_cells.append((row, col))

        # Place ships greedily on highest probability cells
        ship_id = 1
        for ship in self.SHIPS:
            placed = False
            for _ in range(50):  # Try 50 placements
                if len(available_cells) < ship['size']:
                    break

                # Pick random high-probability region
                start_idx = random.randint(0, len(available_cells) - ship['size'])
                candidates = available_cells[start_idx:start_idx + ship['size']]

                # Check if can place horizontally or vertically
                row, col = candidates[0]
                horizontal_ok = all(self.can_place_ship(grid, row, col + i, 1, True)
                                    for i in range(ship['size']))
                vertical_ok = all(self.can_place_ship(grid, row + i, col, 1, False)
                                  for i in range(ship['size']))

                if horizontal_ok:
                    self.place_ship(grid, ship_id, row, col, ship['size'], True)
                    placed = True
                    break
                elif vertical_ok:
                    self.place_ship(grid, ship_id, row, col, ship['size'], False)
                    placed = True
                    break

            if placed:
                ship_id += 1

        # Fill any missing ships randomly
        total_placed = np.sum(grid > 0)
        total_needed = sum(ship['size'] for ship in self.SHIPS)
        if total_placed < total_needed:
            self.fill_remaining_ships(grid)

        return grid

    def fill_remaining_ships(self, grid):
        """Place any missing ships randomly"""
        placed_count = {i: 0 for i in range(1, len(self.SHIPS) + 1)}
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if grid[r, c] > 0:
                    placed_count[grid[r, c]] += 1

        for ship_id, ship in enumerate(self.SHIPS, 1):
            remaining = ship['size'] - placed_count[ship_id]
            for _ in range(remaining):
                self.place_ships_randomly(grid)  # Simple fallback

    def evaluate_layout_difficulty(self, grid):
        """Score layout based on how challenging it is"""
        ship_positions = np.argwhere(grid > 0)
        if len(ship_positions) == 0:
            return 0

        # Distance from center (edge placement = higher score)
        center_dist = np.mean(np.sqrt((ship_positions[:, 0] - 4) ** 2 + (ship_positions[:, 1] - 4) ** 2))

        # Clustering score (ships together = harder to find all)
        clustering = -np.std(ship_positions, axis=0).mean()

        return center_dist + clustering * 0.5

def build_generator(self):
    """Generator: noise -> 8x8 ship layout"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(self.GRID_SIZE * self.GRID_SIZE, activation='sigmoid'),
        layers.Reshape((self.GRID_SIZE, self.GRID_SIZE))
    ])
    return model

def build_discriminator(self):
    """Discriminator: 8x8 layout -> real/fake probability"""
    model = keras.Sequential([
        layers.Flatten(input_shape=(self.GRID_SIZE, self.GRID_SIZE)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def load_gan_models(self, generator_path='generator.weights.h5', discriminator_path='discriminator.weights.h5'):
    """Load pretrained GAN weights"""
    try:
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator.load_weights(generator_path)
        self.discriminator.load_weights(discriminator_path)

        self.generator.trainable = False
        self.discriminator.trainable = False

        self.gan_loaded = True
        return True
    except:
        print("GAN models not found, using random placement")
        self.gan_loaded = False
        return False

def generate_ai_board_with_gan(self):
    """Generate challenging AI ship layout using GAN"""
    if not self.gan_loaded:
        self.place_ships_randomly(self.ai_grid)
        return

    # Generate multiple layouts and pick the best
    best_grid = None
    best_score = -1

    for _ in range(10):  # Try 10 generations
        noise = np.random.normal(0, 1, (1, 100))
        generated = self.generator.predict(noise, verbose=0)[0]

        # Post-process to valid ship layout
        candidate_grid = self.postprocess_gan_output(generated)

        # Score based on "difficulty" (spread out, hard-to-find ships)
        score = self.evaluate_layout_difficulty(candidate_grid)

        if score > best_score:
            best_score = score
            best_grid = candidate_grid.copy()

    self.ai_grid = best_grid

def postprocess_gan_output(self, generated):
    """Convert GAN probabilities to valid ship layout"""
    grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)

    # Threshold and assign ships greedily
    ship_mask = generated > 0.3  # Threshold

    for ship_idx, ship in enumerate(self.SHIPS):
        # Find largest connected region for this ship
        ship_cells = self.find_largest_region(ship_mask)
        if len(ship_cells) >= ship['size']:
            # Place ship on best cells
            for i, (r, c) in enumerate(ship_cells[:ship['size']]):
                grid[r][c] = ship_idx + 1
            ship_mask[ship_cells[:ship['size']]] = False  # Remove used cells

    # Fill remaining with random valid ships if needed
    self.fill_remaining_ships(grid)
    return grid

def evaluate_layout_difficulty(self, grid):
    """Score layout based on how hard it is to find ships"""
    ship_positions = np.argwhere(grid > 0)
    if len(ship_positions) == 0:
        return 0

    # Reward spread-out ships, clustering, edge placement
    center_dist = np.mean(np.sqrt((ship_positions[:, 0] - 4) ** 2 + (ship_positions[:, 1] - 4) ** 2))
    clustering_score = -np.std(ship_positions, axis=0).mean()  # Reward clustering

    return center_dist + clustering_score

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.action = action
        self.is_hovered = False

    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, COLORS['text'], self.rect, 2, border_radius=8)

        text_surface = font.render(self.text, True, COLORS['text'])
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)


    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                self.action()

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Final Project: Battleship with GAN")
        self.clock = pygame.time.Clock()
        self.running = True

        # Fonts
        self.title_font = pygame.font.Font(None, 56)
        self.header_font = pygame.font.Font(None, 32)
        self.text_font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)

        # Game state
        self.rl_game = BattleshipGAN(grid_size=GRID_SIZE)
        self.game_active = False
        self.status_message = "Click 'New Game' to start!"
        self.training = False
        self.training_progress = 0
        self.training_total = 0

        self.manual_placement = False  # default auto placement
        self.placement_mode = False
        self.current_ship_index = 0
        self.placement_horizontal = True  # True = horizontal, False = vertical

        # Grid positions
        self.player_grid_x = 100
        self.player_grid_y = 200
        self.ai_grid_x = 600
        self.ai_grid_y = 200

        # Create buttons
        self.create_buttons()

    def load_shot_gan(self):
        """Wrapper for shot GAN loading"""
        success = self.rl_game.load_shot_gan_models()
        if success:
            self.status_message = "Shot GAN loaded! AI uses smart shot policies!"
        else:
            self.status_message = "❌ Shot GAN load failed - using heuristic shots"

    def can_place_ship(self, grid, row, col, size, horizontal):
        if horizontal:
            if col + size > self.GRID_SIZE:
                return False
            for i in range(size):
                if grid[row][col + i] != 0:
                    return False
        else:
            if row + size > self.GRID_SIZE:
                return False
            for i in range(size):
                if grid[row + i][col] != 0:
                    return False
        return True

    def place_ship(self, grid, ship_id, row, col, size, horizontal):
        if horizontal:
            for i in range(size):
                grid[row][col + i] = ship_id
        else:
            for i in range(size):
                grid[row + i][col] = ship_id

    def place_ships_randomly(self, grid):
        for idx, ship in enumerate(self.SHIPS):
            placed = False
            while not placed:
                horizontal = random.choice([True, False])
                row = random.randint(0, self.GRID_SIZE - 1)
                col = random.randint(0, self.GRID_SIZE - 1)

                if self.can_place_ship(grid, row, col, ship['size'], horizontal):
                    self.place_ship(grid, idx + 1, row, col, ship['size'], horizontal)
                    placed = True

    def create_buttons(self):
        button_x = 1100
        button_y = 200
        button_width = 250
        button_height = 50
        spacing = 60

        self.buttons = [
            Button(button_x, button_y, button_width, button_height,
                   "New Game", COLORS['accent'], COLORS['hit'], self.new_game),
            Button(button_x, button_y + spacing, button_width, button_height,
                   "Load GAN Models", COLORS['success'], COLORS['primary'], self.load_gan),
            Button(button_x, button_y + spacing * 2, button_width, button_height,
                   "Regen AI Fleet", COLORS['train_button'], COLORS['primary'], self.regen_ai_fleet),
            Button(button_x, button_y + spacing * 3, button_width, button_height,
                   "Load Shot GAN", COLORS['train_button'], COLORS['primary'], self.load_shot_gan),
            # In your Game class
            Button(button_x, button_y + spacing * 4, button_width, button_height, "Load Hunter GAN",
                   COLORS['success'], COLORS['primary'], lambda: self.load_hunter_gan()),
            Button(button_x, button_y + spacing * 5, button_width, button_height, "Toggle Manual Placement",
                   COLORS['train_button'], COLORS['primary'], self.toggle_manual_placement)
        ]

    def toggle_manual_placement(self):
        self.manual_placement = not self.manual_placement
        if self.manual_placement:
            self.status_message = "Manual ship placement enabled. Place your ships!"
            self.placement_mode = True
            self.current_ship_index = 0
            self.placement_horizontal = True
            # Clear player grid for manual placement
            self.rl_game.player_grid.fill(0)
        else:
            self.status_message = "Auto ship placement enabled. Placing ships..."
            self.placement_mode = False
            # Automatically place ships
            self.rl_game.place_ships_randomly(self.rl_game.player_grid)
            self.game_active = True  # Game starts immediately

    def load_hunter_gan(self):
        """Load + verify hunter works"""
        success = self.rl_game.load_shot_hunter_gan('models/shot_generator_rtx4070.weights.h5')
        if success:
            self.status_message = "HUNTER GAN LOADED! AI will HUNT ships!"
            print("GAN Status:", self.rl_game.shot_gan_loaded)  # Debug
        else:
            self.status_message = "❌ GAN load failed - check models/ folder"

    def load_gan(self):
        if self.rl_game.load_gan_models():
            self.status_message = "GAN models loaded successfully!"
        else:
            self.status_message = "❌ GAN model files not found"

    def regen_ai_fleet(self):
        self.rl_game.ai_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.rl_game.generate_ai_board_with_gan()
        self.status_message = "New GAN-generated AI fleet!"

    def new_game(self):
        self.rl_game.player_grid.fill(0)
        self.rl_game.ai_grid.fill(0)
        self.rl_game.player_shots.clear()
        self.rl_game.ai_shots.clear()
        self.rl_game.ai_stats = {"hits": 0, "misses": 0, "accuracy": 0.0}

        if self.manual_placement:
            self.placement_mode = True
            self.current_ship_index = 0
            self.placement_horizontal = True
            self.game_active = False
            self.status_message = "Manual placement mode: place your ships!"
        else:
            self.placement_mode = False
            self.rl_game.place_ships_randomly(self.rl_game.player_grid)
            self.rl_game.generate_ai_board_with_gan()
            self.game_active = True
            self.status_message = "Auto placement: game started!"

    def handle_player_placement(self, row, col, mouse_buttons):
        """Handle placing player's ships before game starts."""
        # Right click toggles orientation
        if mouse_buttons[2]:  # right mouse button
            self.placement_horizontal = not self.placement_horizontal
            orient = "horizontal" if self.placement_horizontal else "vertical"
            self.status_message = f"Orientation: {orient}"
            return

        # Left click places the current ship if valid
        if not mouse_buttons[0]:  # no left button
            return

        if self.current_ship_index >= len(self.rl_game.SHIPS):
            return

        ship = self.rl_game.SHIPS[self.current_ship_index]
        size = ship["size"]

        if self.rl_game.can_place_ship(self.rl_game.player_grid,
                                       row, col, size, self.placement_horizontal):
            self.rl_game.place_ship(self.rl_game.player_grid,
                                    ship_id=self.current_ship_index + 1,
                                    row=row, col=col,
                                    size=size,
                                    horizontal=self.placement_horizontal)
            self.current_ship_index += 1

            if self.current_ship_index < len(self.rl_game.SHIPS):
                next_ship = self.rl_game.SHIPS[self.current_ship_index]
                self.status_message = f"Placed {ship['name']}. Now place {next_ship['name']} (size {next_ship['size']})."
            else:
                self.placement_mode = False
                self.game_active = True
                self.status_message = "All ships placed! Start firing on the enemy grid."
        else:
            self.status_message = "Invalid placement (overlap or out of bounds). Try another cell."

    def check_ai_win(self):
        """Check if AI sank all player ships"""
        total_ships = np.sum(self.rl_game.player_grid > 0)
        ai_hits = sum(s['hit'] for s in self.rl_game.ai_shots)
        return ai_hits >= total_ships

    def draw_grid(self, x, y, grid, shots, hide_ships=False):
        # Draw header
        for i in range(GRID_SIZE):
            text = self.small_font.render(str(i), True, COLORS['text'])
            text_rect = text.get_rect(center=(x + i * CELL_SIZE + CELL_SIZE // 2, y - 20))
            self.screen.blit(text, text_rect)

            text = self.small_font.render(str(i), True, COLORS['text'])
            text_rect = text.get_rect(center=(x - 20, y + i * CELL_SIZE + CELL_SIZE // 2))
            self.screen.blit(text, text_rect)

        # Draw cells
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(x + j * CELL_SIZE, y + i * CELL_SIZE,
                                   CELL_SIZE - GRID_PADDING, CELL_SIZE - GRID_PADDING)

                shot = next((s for s in shots if s['row'] == i and s['col'] == j), None)

                if shot:
                    if shot['hit']:
                        color = COLORS['hit']
                        text = "S"
                    else:
                        color = COLORS['miss']
                        text = "•"
                    pygame.draw.rect(self.screen, color, rect, border_radius=5)

                    if text == "S":
                        text_surface = self.text_font.render("X", True, COLORS['text'])
                    else:
                        text_surface = self.text_font.render(text, True, COLORS['text'])
                    text_rect = text_surface.get_rect(center=rect.center)
                    self.screen.blit(text_surface, text_rect)
                elif not hide_ships and grid[i][j] > 0:
                    pygame.draw.rect(self.screen, COLORS['ship'], rect, border_radius=5)
                    text_surface = self.text_font.render("S", True, COLORS['text'])
                    text_rect = text_surface.get_rect(center=rect.center)
                    self.screen.blit(text_surface, text_rect)
                else:
                    pygame.draw.rect(self.screen, COLORS['water'], rect, border_radius=5)

                pygame.draw.rect(self.screen, COLORS['bg_light'], rect, 2, border_radius=5)

    def handle_grid_click(self, pos):
        if not self.game_active:
            return

        x, y = pos

        if self.player_grid_x <= x < self.player_grid_x + GRID_SIZE * CELL_SIZE and \
                self.player_grid_y <= y < self.player_grid_y + GRID_SIZE * CELL_SIZE:
            col = (x - self.player_grid_x) // CELL_SIZE
            row = (y - self.player_grid_y) // CELL_SIZE
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                if self.placement_mode:
                    self.handle_player_placement(row, col, pygame.mouse.get_pressed())
                return  # do not treat as shot when in placement area

        # Check if click is on AI grid
        if (self.ai_grid_x <= x <= self.ai_grid_x + GRID_SIZE * CELL_SIZE and
                self.ai_grid_y <= y <= self.ai_grid_y + GRID_SIZE * CELL_SIZE):

            col = (x - self.ai_grid_x) // CELL_SIZE
            row = (y - self.ai_grid_y) // CELL_SIZE

            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                self.player_shoot(row, col)

    def player_shoot(self, row, col):
        """Player shoots → AI responds"""
        if self.placement_mode:
            self.status_message = "Place all your ships first."
            return

        # Check already shot
        if any(s['row'] == row and s['col'] == col for s in self.rl_game.player_shots):
            self.status_message = "Already fired here!"
            return

        # PLAYER SHOOT
        ai_hit = self.rl_game.ai_grid[row][col] > 0
        self.rl_game.player_shots.append({'row': row, 'col': col, 'hit': ai_hit})

        if ai_hit:
            self.status_message = f"HIT at ({row},{col})! Fire again!"
            pygame.time.wait(400)

            # PLAYER WIN CHECK
            total_ships = np.sum(self.rl_game.ai_grid > 0)
            if sum(s['hit'] for s in self.rl_game.player_shots) >= total_ships:
                self.game_active = False
                self.status_message = "VICTORY! You sank all enemy ships!"
                return
        else:
            self.status_message = f"❌ Miss at ({row},{col}). AI's turn..."
            pygame.time.wait(500)
            self.ai_shoot()  # ← CRITICAL: AI TAKES TURN!

    def ai_shoot(self):
        """🎯 PERFECT HUNTER AI - Called after player shoots"""
        ai_row, ai_col = self.rl_game.ai_choose_shot_hunter(self.rl_game.ai_shots)

        if ai_row is None:
            self.game_active = False
            self.status_message = "Player wins - AI out of moves!"
            return

        # AI SHOOTS
        ai_hit = self.rl_game.player_grid[ai_row][ai_col] > 0
        self.rl_game.ai_shots.append({'row': ai_row, 'col': ai_col, 'hit': ai_hit})

        # UPDATE STATS
        if ai_hit:
            self.rl_game.ai_stats['hits'] += 1
            self.status_message = f"HUNTER HIT ({ai_row},{ai_col})! AI shoots again!"
            pygame.time.wait(500)

            # WIN CHECK
            total_ships = np.sum(self.rl_game.player_grid > 0)
            if sum(s['hit'] for s in self.rl_game.ai_shots) >= total_ships:
                self.game_active = False
                self.status_message = "HUNTER AI DESTROYS FLEET!"
                return

            # HUNTER STRIKES AGAIN ON HIT!
            self.ai_shoot()
        else:
            self.rl_game.ai_stats['misses'] += 1
            total = self.rl_game.ai_stats['hits'] + self.rl_game.ai_stats['misses']
            accuracy = (self.rl_game.ai_stats['hits'] / total * 100) if total > 0 else 0
            self.rl_game.ai_stats['accuracy'] = accuracy
            self.status_message = f"❌ Hunter miss ({ai_row},{ai_col}). Accuracy: {accuracy:.1f}%"

    def draw(self):
        self.screen.fill(COLORS['bg'])

        # Title
        title_text = self.title_font.render("Final project: Battleship with GAN",
                                            True, COLORS['accent'])
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
        self.screen.blit(title_text, title_rect)

        # Status message
        status_text = self.text_font.render(self.status_message, True, COLORS['text'])
        status_rect = status_text.get_rect(center=(SCREEN_WIDTH // 2, 120))
        self.screen.blit(status_text, status_rect)

        # Grid labels
        player_label = self.header_font.render("YOUR FLEET", True, COLORS['text'])
        player_rect = player_label.get_rect(
            center=(self.player_grid_x + GRID_SIZE * CELL_SIZE // 2, 150)
        )
        self.screen.blit(player_label, player_rect)

        ai_label = self.header_font.render("ENEMY WATERS", True, COLORS['text'])
        ai_rect = ai_label.get_rect(
            center=(self.ai_grid_x + GRID_SIZE * CELL_SIZE // 2, 150)
        )
        self.screen.blit(ai_label, ai_rect)

        if self.placement_mode:
            orient = "Horizontal" if self.placement_horizontal else "Vertical"
            hint = f"Placing {self.rl_game.SHIPS[self.current_ship_index]['name']} ({orient})"
            hint_surface = self.small_font.render(hint, True, COLORS["warning"])
            self.screen.blit(hint_surface, (self.player_grid_x, self.player_grid_y - 30))

        # Draw grids
        self.draw_grid(self.player_grid_x, self.player_grid_y,
                       self.rl_game.player_grid, self.rl_game.ai_shots, hide_ships=False)
        self.draw_grid(self.ai_grid_x, self.ai_grid_y,
                       self.rl_game.ai_grid, self.rl_game.player_shots, hide_ships=True)

        # Draw buttons
        for button in self.buttons:
            button.draw(self.screen, self.text_font)

        # Draw stats panel
        stats_x = 1100
        stats_y = 600

        stats_title = self.header_font.render("AI STATISTICS", True, COLORS['text'])
        self.screen.blit(stats_title, (stats_x, stats_y))

        stats_y += 50
        stats_lines = [
            f"Games Played: {self.rl_game.games_played}",
            f"",
            f"GAM Stats:",
            f"GAM Loaded: {'Yes' if self.rl_game.gan_loaded else 'No'}",
            f"AI Hits: {self.rl_game.ai_stats['hits']}",
            f"AI Accuracy: {self.rl_game.ai_stats['accuracy']:.1f}%",
        ]

        for line in stats_lines:
            text = self.small_font.render(line, True, COLORS['text'])
            self.screen.blit(text, (stats_x, stats_y))
            stats_y += 25

        # Draw training progress
        if self.training:
            progress_x = 1100
            progress_y = 450
            progress_width = 250
            progress_height = 30

            pygame.draw.rect(self.screen, COLORS['bg_light'],
                             (progress_x, progress_y, progress_width, progress_height),
                             border_radius=5)

            progress = self.training_progress / self.training_total
            filled_width = int(progress_width * progress)
            pygame.draw.rect(self.screen, COLORS['accent'],
                             (progress_x, progress_y, filled_width, progress_height),
                             border_radius=5)

            progress_text = self.small_font.render(
                f"Training: {self.training_progress}/{self.training_total}",
                True, COLORS['text']
            )
            text_rect = progress_text.get_rect(center=(progress_x + progress_width // 2,
                                                       progress_y - 15))
            self.screen.blit(progress_text, text_rect)

        # MyInfo
        infoLines = [
            "Subject: Artificial Intelligence",
            "Teacher: Alejandra Hernandez Sanchez",
            "Student: Carlos Sebastian Garcia Luna Garcia"
        ]
        stats_y = 700
        for line in infoLines:
            info_text = self.small_font.render(line, True, COLORS['warning'])
            info_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, stats_y))
            self.screen.blit(info_text, info_rect)
            stats_y += 25

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_grid_click(event.pos)

                # Handle button events
                for button in self.buttons:
                    button.handle_event(event)

            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

def sample_top_k(available, scores, k=3):
        top_indices = np.argpartition(scores, -k)[-k:]
        top_scores = scores[top_indices]
        probs = top_scores / np.sum(top_scores)
        chosen_idx = np.random.choice(top_indices, p=probs)
        return available[chosen_idx]
def fill_remaining_ships(self, grid):
    """Fallback: place any missing ships randomly"""
    placed_ships = {i + 1: 0 for i in range(len(self.SHIPS))}
    for r in range(self.GRID_SIZE):
        for c in range(self.GRID_SIZE):
            if grid[r][c] > 0:
                placed_ships[grid[r][c]] += 1

    for ship_idx, ship in enumerate(self.SHIPS):
        ship_id = ship_idx + 1
        if placed_ships[ship_id] < ship['size']:
            self.place_remaining_ship(grid, ship_id, ship['size'] - placed_ships[ship_id])



if __name__ == "__main__":
    game = Game()
    game.run()