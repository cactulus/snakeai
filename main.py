from array import array
import random
import numpy as np
from game import Snake, UP, DOWN, LEFT, RIGHT

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

LEARN_RATE = 0.001
GAMMA = 0.9

class SnakeDeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self):
        torch.save(self, "model.pth")

class SnakeOptimiser:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), LEARN_RATE)
        self.criterion = nn.MSELoss()

    def step(self, state, move, score, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        move = torch.tensor(move, dtype=torch.long)
        score = torch.tensor(score, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            move = torch.unsqueeze(move, 0)
            score = torch.unsqueeze(score, 0)
            game_over = (game_over, )

        pred = self.model(state)
        target = pred.clone()
        for i in range(len(game_over)):
            n = score[i]
            if not game_over[i]:
                n = score[i] + GAMMA * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(move[i]).item()] = n

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class SnakeAI:
    def __init__(self, model):
        self.games = 0
        self.model = model
        self.opt = SnakeOptimiser(self.model)
        self.states = deque(maxlen=1000000)

    def get_state(self, game):
        state = [
            game.is_danger_up(),
            game.is_danger_down(),
            game.is_danger_left(),
            game.is_danger_right(),

            game.dir == UP,
            game.dir == DOWN,
            game.dir == LEFT,
            game.dir == RIGHT,

            game.fy < game.py,
            game.fy > game.py,
            game.fx < game.px,
            game.fx > game.px
        ]
        return np.array(state, dtype=int)

    def remember(self, state, move, score, sn, game_over):
        self.states.append((state, move, score, sn, game_over))

    def practice(self, state, move, score, sn, game_over):
        self.opt.step(state, move, score, sn, game_over)

    def train(self):
        if len(self.states) > 4000:
            sample = random.sample(self.states, 4000)
        else:
            sample = self.states
        states, moves, rewards, states_new, game_overs = zip(*sample)
        self.opt.step(states, moves, rewards, states_new, game_overs)

    def make_move(self, state):
        move = [0, 0, 0, 0]
        if random.randint(0, 200) < 100 - self.games:
            move[random.randint(0, 3)] = 1
        else:
            p = self.model(torch.tensor(state, dtype=torch.float))
            mi = torch.argmax(p).item()
            move[mi] = 1
        return move

    def make_ai_move(self, state):
        move = [0, 0, 0, 0]
        prediction = self.model(torch.tensor(state, dtype=torch.float))
        mi = torch.argmax(prediction).item()
        move[mi] = 1
        return move

def train(simulate):
    model = SnakeDeepNet()

    ai = SnakeAI(model)
    game = Snake(simulate=simulate)
    max_score = 0

    while True:
        state = ai.get_state(game)
        move = ai.make_move(state)
        score, game_over, end_score = game.step(move)
        sn = ai.get_state(game)

        ai.practice(state, move, score, sn, game_over)
        ai.remember(state, move, score, sn, game_over)

        if game_over:
            game.reset()
            ai.games += 1
            ai.train()

            if end_score > max_score:
                max_score = end_score
                ai.model.save()

            print("Run", ai.games, "Score", end_score, "Record", max_score)

def test():
    model = torch.load("model.pth")
    model.eval()
    max_score = 0
    games_count = 0

    ai = SnakeAI(model)
    game = Snake()

    while True:
        state = ai.get_state(game)
        move = ai.make_ai_move(state)
        _, game_over, end_score = game.step(move)

        if game_over:
            game.reset()
            games_count += 1

            if end_score > max_score:
                max_score = end_score
            print("Run", games_count, "Score", end_score, "Max Score", max_score)

if __name__ == "__main__":
    test()
    #train(simulate=False)