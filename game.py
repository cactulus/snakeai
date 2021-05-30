import math
import pygame
import random
import numpy as np

UP = (0, -1)
DOWN = (0, 1)
RIGHT = (1, 0)
LEFT = (-1, 0)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class Snake():
    def __init__(self, simulate=False):
        super().__init__()
        if not simulate:
            pygame.init()
            self.font = pygame.font.SysFont("arial", 22)
        self.w = 800
        self.h = 800
        self.ts = self.w // 20
        self.tiles = self.w // self.ts
        self.simulate = simulate
        if not simulate:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()
        self.tail = []
        self.tail_length = 1
        self.px = 5
        self.py = 5
        self.fx = 15
        self.fy = 15
        self.dir = RIGHT

    # move = [up, down, left, right]
    def step(self, move):
        new_dir = None
        game_over = False
        score = 0
        
        if np.array_equal(move, [1, 0, 0, 0]):
            new_dir = UP
        if np.array_equal(move, [0, 1, 0, 0]):
            new_dir = DOWN
        if np.array_equal(move, [0, 0, 1, 0]):
            new_dir = RIGHT
        if np.array_equal(move, [0, 0, 0, 1]):
            new_dir = LEFT

        if self.dir == UP and new_dir == DOWN:
            score = -10
            game_over = True
        elif self.dir == DOWN and new_dir == UP:
            score = -10
            game_over = True
        elif self.dir == LEFT and new_dir == RIGHT:
            score = -10
            game_over = True
        elif self.dir == RIGHT and new_dir == LEFT:
            score = -10
            game_over = True

        self.dir = new_dir

        self.px += self.dir[0]
        self.py += self.dir[1]

        self.tail.append((self.px, self.py))

        while len(self.tail) > self.tail_length:
            self.tail.pop(0)

        if not self.simulate:
            self.display.fill(BLACK)

        if self.px < 0 or self.px >= self.tiles:
            score = -10
            game_over = True

        if self.py < 0 or self.py >= self.tiles:
            score = -10
            game_over = True

        for t_i in range(len(self.tail)-1):
            t = self.tail[t_i]
            if self.px == t[0] and self.py == t[1]:
                score = -10
                game_over = True
                break

        if not self.simulate:
            for t_i in range(len(self.tail)):
                t = self.tail[t_i]
                tx = t[0]
                ty = t[1]
                pygame.draw.rect(self.display, GREEN, pygame.Rect(tx * self.ts, ty * self.ts, self.ts, self.ts))

        if not self.simulate:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(self.px * self.ts, self.py * self.ts, self.ts, self.ts))
            pygame.draw.rect(self.display, RED, pygame.Rect(self.fx * self.ts, self.fy * self.ts, self.ts, self.ts))

            text = self.font.render("Score: " + str(self.tail_length), True, WHITE)
            self.display.blit(text, [0, 0])

            pygame.display.flip()

        if self.px == self.fx and self.py == self.fy:
            score = 10
            self.tail_length += 1
            self.replace_food()

        if not self.simulate:
            self.clock.tick(20)
        return score, game_over, self.tail_length

    def reset(self):
        self.tail = []
        self.tail_length = 1
        self.px = 5
        self.py = 5
        self.fx = 15
        self.fy = 15
        self.dir = RIGHT

    def replace_food(self):
        self.fx = math.floor(random.random() * self.tiles)
        self.fy = math.floor(random.random() * self.tiles)
        if self.is_body(self.fx, self.fy):
            self.replace_food()
        if self.fx == self.px and self.fy == self.py:
            self.replace_food()

    def is_body(self, x, y):
        for t_i in range(len(self.tail)):
            t = self.tail[t_i]
            tx = t[0]
            ty = t[1]
            if x == tx and y == ty:
                return True
        return False

    def is_danger_up(self):
        return self.py - 1 < 0 or self.is_body(self.px, self.py - 1)
        
    def is_danger_down(self):
        return self.py + 1 >= self.tiles or self.is_body(self.px, self.py + 1)
        
    def is_danger_left(self):
        return self.px - 1 < 0 or self.is_body(self.px - 1, self.py)
        
    def is_danger_right(self):
        return self.px + 1 >= self.tiles or self.is_body(self.px + 1, self.py)
