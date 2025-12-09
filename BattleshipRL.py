import pygame
import numpy as np
import random
import pickle
import os

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
GRID_SIZE = 8
CELL_SIZE = 60
GRID_PADDING = 10

# Colors
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


class BattleshipRL:
    def __init__(self, grid_size=8):
        self.GRID_SIZE = grid_size
        self.SHIPS = SHIPS

        # Initialize game state
        self.player_grid = np.zeros((grid_size, grid_size), dtype=int)
        self.ai_grid = np.zeros((grid_size, grid_size), dtype=int)
        self.player_shots = []
        self.ai_shots = []

        # Q-learning parameters
        self.q_table = {}
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.games_played = 0
        self.ai_stats = {'hits': 0, 'misses': 0, 'accuracy': 0.0}

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

    def get_q_value(self, state, action):
        key = f"{state}-{action}"
        return self.q_table.get(key, 0.0)

    def update_q_value(self, state, action, reward):
        key = f"{state}-{action}"
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward - current_q)
        self.q_table[key] = new_q

    def get_ai_state(self):
        recent_hits = [s for s in self.ai_shots if s['hit']][-3:]
        if len(recent_hits) == 0:
            return 'exploring'
        last_hit = recent_hits[-1]
        return f"hunting-{last_hit['row']}-{last_hit['col']}"

    def get_ai_state_from_shots(self, shots):
        recent_hits = [s for s in shots if s['hit']][-3:]
        if not recent_hits:
            return 'exploring'
        last_hit = recent_hits[-1]
        return f"hunting-{last_hit['row']}-{last_hit['col']}"

    def ai_choose_action(self, grid, shots):
        state = self.get_ai_state_from_shots(shots)
        epsilon = max(0.1, 1 - self.games_played * 0.1)

        if random.random() < epsilon:
            return self.get_random_valid_shot(shots)
        else:
            recent_hits = [s for s in shots if s['hit']][-2:]
            candidates = []

            if recent_hits:
                for hit in recent_hits:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        r, c = hit['row'] + dr, hit['col'] + dc
                        if 0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE:
                            if not any(s['row'] == r and s['col'] == c for s in shots):
                                candidates.append((r, c))

            if not candidates:
                return self.get_random_valid_shot(shots)

            best_action = None
            best_value = float('-inf')

            for r, c in candidates:
                action = f"{r}-{c}"
                q_value = self.get_q_value(state, action)
                if q_value > best_value:
                    best_value = q_value
                    best_action = (r, c)

            if best_action:
                return best_action
            else:
                return random.choice(candidates)

    def get_random_valid_shot(self, shots):
        while True:
            row = random.randint(0, self.GRID_SIZE - 1)
            col = random.randint(0, self.GRID_SIZE - 1)
            if not any(s['row'] == row and s['col'] == col for s in shots):
                return (row, col)

    def simulate_training_game(self):
        training_player_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        training_ai_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)

        self.place_ships_randomly(training_player_grid)
        self.place_ships_randomly(training_ai_grid)

        training_ai_shots = []
        training_player_shots = []

        player_ship_cells = np.sum(training_player_grid > 0)
        ai_ship_cells = np.sum(training_ai_grid > 0)

        while True:
            player_row, player_col = self.get_random_valid_shot(training_player_shots)
            player_hit = training_ai_grid[player_row][player_col] > 0
            training_player_shots.append({'row': player_row, 'col': player_col, 'hit': player_hit})

            if sum(s['hit'] for s in training_player_shots) == ai_ship_cells:
                return {'ai_won': False}

            state = self.get_ai_state_from_shots(training_ai_shots)
            ai_row, ai_col = self.ai_choose_action(training_player_grid, training_ai_shots)

            ai_hit = training_player_grid[ai_row][ai_col] > 0
            action = f"{ai_row}-{ai_col}"
            reward = 10 if ai_hit else -1

            self.update_q_value(state, action, reward)
            training_ai_shots.append({'row': ai_row, 'col': ai_col, 'hit': ai_hit})

            if sum(s['hit'] for s in training_ai_shots) == player_ship_cells:
                return {'ai_won': True}

    def save_q_table(self, filename='battleship_qtable.pkl'):
        data = {'q_table': self.q_table, 'games_played': self.games_played}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        return True

    def load_q_table(self, filename='battleship_qtable.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.q_table = data['q_table']
            self.games_played = data['games_played']
            return True
        return False


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
        pygame.display.set_caption("Evaluation 3: Battleship with RL and Q-Learning")
        self.clock = pygame.time.Clock()
        self.running = True

        # Fonts
        self.title_font = pygame.font.Font(None, 56)
        self.header_font = pygame.font.Font(None, 32)
        self.text_font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)

        # Game state
        self.rl_game = BattleshipRL()
        self.game_active = False
        self.status_message = "Click 'New Game' to start!"
        self.training = False
        self.training_progress = 0
        self.training_total = 0

        # Grid positions
        self.player_grid_x = 100
        self.player_grid_y = 200
        self.ai_grid_x = 600
        self.ai_grid_y = 200

        # Create buttons
        self.create_buttons()

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
                   "Train 10,000 Games", COLORS['train_button'], COLORS['primary'],
                   lambda: self.start_training(10_000)),
            Button(button_x, button_y + spacing * 2, button_width, button_height,
                   "Train 100,000 Games", COLORS['train_button'], COLORS['primary'],
                   lambda: self.start_training(100_000)),
            Button(button_x, button_y + spacing * 3, button_width, button_height,
                   "Save Q-Table", COLORS['success'], COLORS['primary'], self.save_qtable),
            Button(button_x, button_y + spacing * 4, button_width, button_height,
                   "Load Q-Table", COLORS['warning'], COLORS['primary'], self.load_qtable),
        ]

    def new_game(self):
        self.game_active = True
        self.rl_game.player_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.rl_game.ai_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.rl_game.place_ships_randomly(self.rl_game.player_grid)
        self.rl_game.place_ships_randomly(self.rl_game.ai_grid)
        self.rl_game.player_shots = []
        self.rl_game.ai_shots = []
        self.rl_game.ai_stats = {'hits': 0, 'misses': 0, 'accuracy': 0.0}
        self.status_message = "Game started! Click on enemy waters to fire!"

    def start_training(self, num_games):
        if self.training:
            return

        self.training = True
        self.training_progress = 0
        self.training_total = num_games
        self.game_active = False
        self.status_message = f"Training AI with {num_games} games..."

        wins = 0
        for i in range(num_games):
            result = self.rl_game.simulate_training_game()
            self.rl_game.games_played += 1
            if result['ai_won']:
                wins += 1

            self.training_progress = i + 1

            # Update display every 10 games
            if (i + 1) % 10 == 0:
                self.draw()
                pygame.display.flip()

        self.training = False
        self.training_progress = 0
        self.status_message = f"Training complete! AI won {wins}/{num_games} ({wins / num_games * 100:.1f}%)"

    def save_qtable(self):
        if self.rl_game.save_q_table():
            self.status_message = "Q-Table saved successfully!"

    def load_qtable(self):
        if self.rl_game.load_q_table():
            self.status_message = f"Q-Table loaded! Games: {self.rl_game.games_played}"
        else:
            self.status_message = "❌ Q-Table file not found"

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

        # Check if click is on AI grid
        if (self.ai_grid_x <= x <= self.ai_grid_x + GRID_SIZE * CELL_SIZE and
                self.ai_grid_y <= y <= self.ai_grid_y + GRID_SIZE * CELL_SIZE):

            col = (x - self.ai_grid_x) // CELL_SIZE
            row = (y - self.ai_grid_y) // CELL_SIZE

            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                self.player_shoot(row, col)

    def player_shoot(self, row, col):
        # Check if already shot
        if any(s['row'] == row and s['col'] == col for s in self.rl_game.player_shots):
            self.status_message = "Already fired here! Try another cell."
            return

        # Player shoots
        hit = self.rl_game.ai_grid[row][col] > 0
        self.rl_game.player_shots.append({'row': row, 'col': col, 'hit': hit})

        if hit:
            self.status_message = f"HIT at ({row}, {col})! Fire again!"
        else:
            self.status_message = f"Miss at ({row}, {col}). AI's turn..."

        # Check player win
        player_ship_cells = np.sum(self.rl_game.ai_grid > 0)
        if sum(s['hit'] for s in self.rl_game.player_shots) == player_ship_cells:
            self.game_active = False
            self.rl_game.games_played += 1
            self.status_message = "VICTORY! You sank all enemy ships!"
            return

        # AI's turn (only if player missed)
        if not hit:
            pygame.time.wait(500)  # Small delay
            self.ai_shoot()

    def ai_shoot(self):
        state = self.rl_game.get_ai_state()
        ai_row, ai_col = self.rl_game.ai_choose_action(
            self.rl_game.player_grid, self.rl_game.ai_shots
        )

        ai_hit = self.rl_game.player_grid[ai_row][ai_col] > 0
        action = f"{ai_row}-{ai_col}"
        reward = 10 if ai_hit else -1

        self.rl_game.update_q_value(state, action, reward)
        self.rl_game.ai_shots.append({'row': ai_row, 'col': ai_col, 'hit': ai_hit})

        # Update stats
        self.rl_game.ai_stats['hits'] += 1 if ai_hit else 0
        self.rl_game.ai_stats['misses'] += 0 if ai_hit else 1
        self.rl_game.ai_stats['accuracy'] = (
                self.rl_game.ai_stats['hits'] / len(self.rl_game.ai_shots) * 100
        )

        if ai_hit:
            self.status_message = f"AI HIT your ship at ({ai_row}, {ai_col})!"
            self.ai_shoot()
        else:
            self.status_message = f"AI missed at ({ai_row}, {ai_col}). Your turn!"

        # Check AI win
        player_ship_cells = np.sum(self.rl_game.player_grid > 0)
        if sum(s['hit'] for s in self.rl_game.ai_shots) == player_ship_cells:
            self.game_active = False
            self.rl_game.games_played += 1
            self.status_message = "AI WINS! All your ships were sunk!"

    def draw(self):
        self.screen.fill(COLORS['bg'])

        # Title
        title_text = self.title_font.render("Evaluation 3: Battleship with RL and Q-Learning",
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
        stats_y = 500

        stats_title = self.header_font.render("AI STATISTICS", True, COLORS['text'])
        self.screen.blit(stats_title, (stats_x, stats_y))

        stats_y += 50
        stats_lines = [
            f"Games Played: {self.rl_game.games_played}",
            f"",
            f"Current Game:",
            f"AI Hits: {self.rl_game.ai_stats['hits']}",
            f"AI Misses: {self.rl_game.ai_stats['misses']}",
            f"AI Accuracy: {self.rl_game.ai_stats['accuracy']:.1f}%",
            f"",
            f"Q-Learning:",
            f"Q-Table Size: {len(self.rl_game.q_table)}",
            f"Epsilon: {max(0.1, 1 - self.rl_game.games_played * 0.1):.2%}",
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

if __name__ == "__main__":
    game = Game()
    game.run()