import pygame, random, sys, time
from pygame.surfarray import array3d
from pygame import display

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)


class SnakeEnv(gym.Env):

    metadata = {'render.modes': ['human'],
                'render_modes': ['rgb'],
                'render_fps': 10}

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.frame_size_x = 200
        self.frame_size_y = 200
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.frame_size_x, self.frame_size_y), dtype=np.int
        )
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
        self.reset()
        self.STEP_LIMIT = 1000
        self.sleep = 0

    def reset(self):
        """
        Resets the game along with the default snake size, screen and spawn the food
        :return:
        """
        self.game_window.fill(BLACK)
        self.snake_pos = [100, 50]
        self.snake_body = [
            [100, 50],
            [90, 50],
            [80, 50]
        ]
        self.food_pos = self.spawn_food()
        self.food_spawn = True

        self.direction = "RIGHT"
        self.action = self.direction

        self.score = 0
        self.steps = 0

        img = array3d(display.get_surface())
        img = np.swapaxes(img, 0, 1)
        return img

    @staticmethod
    def change_direction(action, direction):
        if action == 0 and direction != 1:
            direction = 0

        if action == 1 and direction != 0:
            direction = 1

        if action == 2 and direction != 3:
            direction = 2

        if action == 3 and direction != 2:
            direction = 3

        return direction

    @staticmethod
    def move(direction, snake_pos):

        if direction == "UP":
            snake_pos[1] -= 10

        if direction == "DOWN":
            snake_pos[1] += 10

        if direction == "LEFT":
            snake_pos[0] -= 10

        if direction == "RIGHT":
            snake_pos[0] += 10

        return snake_pos

    def spawn_food(self):
        """
        # ex : [100, 50]
        # SOLID : I wanted to add a logic to check if the snake_pos == food_pos
        # But remember, it violates SINGLE RESPONSIBILITY PRINCIPLE
        # We will handle the same in other function. This function just spawns the food.
        """
        return [random.randrange(1, (self.frame_size_x // 10) * 10),
                random.randrange(1, (self.frame_size_y // 10) * 10)]

    def eat(self):
        """
        We are basically checking if the current Food position will match the current Snake Pos
        If YES, food has been eaten
        """
        #         print("Snake Body {} : Food Pos {}".format(self.snake_body[0], self.food_pos))
        #         return (self.snake_body[0][0] == self.food_pos[0]) and (self.snake_body[0][1] == self.food_pos[1])
        condition_1 = True if self.food_pos[0] in range(self.snake_pos[0] - 10, self.snake_pos[0] + 10) else False
        condition_2 = True if self.food_pos[1] in range(self.snake_pos[1] - 10, self.snake_pos[1] + 10) else False
        return condition_1 and condition_2

    def step(self, action):
        """
        This is the classic OpenAI step method that takes an action as argument and performs it on the Environment and
        send back the new state
        Question that needs to be answered : What happens when your agent performs the action on the environment
        here : img = observation
        :return: img, reward, done, info
        """
        scoreholder = self.score
        reward = 0

        self.direction = SnakeEnv.change_direction(action, self.direction)
        self.snake_pos = SnakeEnv.move(self.direction, self.snake_pos)
        self.snake_body.insert(0, list(self.snake_pos))

        reward = self.food_handler() # this is basically a function to calculate reward

        # how do we update the env after applying the action
        # viz. if snake moved up, updated status is that snake moved one step up
        self.update_game_state()

        reward, done = self.game_over(reward)
        img = self.get_image_array_from_game()  # Get observations

        info = {'score': self.score}
        self.steps += 1
        time.sleep(self.sleep)

        return img, reward, done, info

    def get_image_array_from_game(self):
        img = array3d(display.get_surface())
        img = np.swapaxes(img, 0, 1)
        return img

    def food_handler(self):
        if self.eat():
            self.score += 1
            reward = 1
            self.food_spawn = False
        else:
            self.snake_body.pop()
            reward = 0

        if not self.food_spawn:
            self.food_pos = self.spawn_food()
        self.food_spawn = True

        return reward

    def update_game_state(self):
        self.game_window.fill(BLACK)
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))

        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))

    def display_score(self, color, font, size):
        """
        Few basic things :
        1. Place a rectangle to show the score
        2. use system font to display/print/notify the score
        3. Place the display surface on the game window
        4. Place this widget on the game_window
        """
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render("Score : {}".format(self.score), True, color)

        score_rect = score_surface.get_rect()
        score_rect.midtop = (self.frame_size_x / 10, 15)
        self.game_window.blit(score_surface, score_rect)

    def game_over(self, reward):
        """
        2 Conditions
        1. If snake touches the BOX boundary
        2. If snake touches its own body
        """
        if self.snake_pos[0] < 0 or self.snake_pos[0] > (self.frame_size_x - 10):
            return -1, True
        if self.snake_pos[1] < 0 or self.snake_pos[1] > (self.frame_size_y - 10):
            return -1, True

        # Snake touching its own body
        # Quick Explanation of the above logic
        # Block is any part of the snake body except its head.
        # If snake's head position is anywhere equal to a block of its own body, snake has touched its own body.
        # End game.
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return -1, True

        if self.steps >= self.STEP_LIMIT:
            return 0, True

        return reward, False

    def render(self, mode="human"):
        if mode == "human":
            display.update()

    def close(self):
        pass

    def end_game(self):
        message = pygame.font.SysFont('arial', 45)
        message_surface = message.render('GAME HAS ENDED', True, RED)
        message_rect = message_surface.get_rect()
        message_rect.midtop = (self.frame_size_x / 2, self.frame_size_y / 4)

        # Below code will do the following
        # Fill the screen black, clearing the snake
        # Display the message that Game has Ended
        # Display the score
        # Wait and then end the game and session
        self.game_window.fill(BLACK)
        self.game_window.blit(message_surface, message_rect)
        self.display_score(RED, 'times', 20)
        pygame.display.flip()
        time.sleep(3)
        pygame.quit()
        sys.exit()