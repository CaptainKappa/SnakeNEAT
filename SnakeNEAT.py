import pygame
import pygame.font
import random
import os
import neat
import math

pygame.font.init()
WIN_WIDTH = 600
WIN_HEIGHT = 300
BLOCK_SIZE = 10
EVAL_FONT = pygame.font.SysFont("Power Red and Green", 20)
gen = 0

class Snake:
    COLOR_FILLED = (0, 0, 0)
    COLOR_BORDER = (99, 163, 96)


    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = 1
        self.body = []
        self.xForce = 0
        self.yForce = 0
        self.direction = None


    def move(self):
        self.x += self.xForce * BLOCK_SIZE
        self.y += self.yForce * BLOCK_SIZE


    def changeForce(self,input):
        if input == 0 and self.direction != "down":
            self.xForce, self.yForce = 0, -1
            self.direction = "up"
        elif input == 1 and self.direction != "up":
            self.xForce, self.yForce = 0, 1
            self.direction = "down"
        elif input == 2 and self.direction != "right":
            self.xForce, self.yForce = -1, 0
            self.direction = "left"
        elif input == 3 and self.direction != "left": 
            self.xForce, self.yForce = 1, 0
            self.direction = "right"


    def draw(self, win):
        snakeHead = []
        snakeHead.append(self.x)
        snakeHead.append(self.y)
        self.body.append(snakeHead)

        if len(self.body) > self.length:
            del self.body[0]

        for element in self.body:
            pygame.draw.rect(win, self.COLOR_FILLED, [element[0], element[1], BLOCK_SIZE, BLOCK_SIZE])
            pygame.draw.rect(win, self.COLOR_BORDER, [element[0], element[1], BLOCK_SIZE, BLOCK_SIZE], 1)


    def colApple(self, apple):
        if self.x == apple.x and self.y == apple.y:
            apple.ate = True
            self.length += 1
            return True
        else:
            return False


    def colWall(self):
        if self.x >= 281 or self.x <= 9 or self.y >= 281 or self.y <= 9:
            return True


    def colBody(self):
        for i in range(len(self.body) - 1):
            if self.x == self.body[i][0] and self.y == self.body[i][1]:
                return True

    def getInputs(self,apple):
        distWallStraight = 0
        distWallRight = 0
        distWallLeft = 0

        if self.direction == "down":
            distWallStraight = abs(300 - self.y)
            distWallRight = abs(0 - self.x)
            distWallLeft = abs(300 - self.x)

        if self.direction == "up":
            distWallStraight = abs(0 - self.y)
            distWallRight = abs(300 - self.x)
            distWallLeft = abs(0 - self.x)

        if self.direction == "right":
            distWallStraight = abs(300 - self.x)
            distWallRight = abs(300 - self.y)
            distWallLeft = abs(0 - self.y)

        if self.direction == "left":
            distWallStraight = abs(0 - self.x)
            distWallRight = abs(0 - self.y)
            distWallLeft = abs(300 - self.y)

        distApple = math.sqrt(pow((self.x - apple.x),2) + pow((self.y - apple.y),2))

        return [distWallStraight, distWallRight, distWallLeft, distApple]


class Apple:
    COLOR_FILLED = (184, 44, 44)
    COLOR_BORDER = (99, 163, 96)
    

    def __init__(self,snakeBody):
        while True:
            self.x = random.randint(1,((WIN_WIDTH/2)/BLOCK_SIZE) - 2) * BLOCK_SIZE
            self.y = random.randint(1,(WIN_HEIGHT/BLOCK_SIZE) - 2) * BLOCK_SIZE
        
            if not self.checkOccupied(snakeBody):
                break



    def draw(self, win):
        pygame.draw.rect(win, self.COLOR_FILLED, [self.x, self.y, BLOCK_SIZE, BLOCK_SIZE])
        pygame.draw.rect(win, self.COLOR_BORDER, [self.x, self.y, BLOCK_SIZE, BLOCK_SIZE], 1)

    def checkOccupied(self,snakeBody):
        for element in snakeBody:
            if element[0] == self.x and element[1] == self.y:
                return True
            else:
                return False


# RUNNING GAME

def draw_window_play(win, snake, apple, evalList):
    win.fill((99, 163, 96))
    snake.draw(win)
    apple.draw(win)
    pygame.draw.lines(win, (0, 0, 0), False, [(300,0), (300,300)])

    scoreLabel = EVAL_FONT.render("Current Score: " + str(evalList[0]), 1, (0, 0 ,0))
    popLabel = EVAL_FONT.render("Current Population: " + str(evalList[1]), 1, (0, 0, 0))
    genLabel = EVAL_FONT.render("Current Generation: " + str(evalList[2]), 1, (0, 0 ,0))

    win.blit(scoreLabel, (350, 50))
    win.blit(genLabel, (350, 150))
    win.blit(popLabel, (350, 250))
    
    pygame.display.flip()
    pygame.display.update()


def main(genomes, config):
    global gen
    pop = 0
    nets = []
    ge = []
    snakes = []
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    clock = pygame.time.Clock()


    for index, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snake = Snake(150, 150)
        snakes.append(snake)
        g.fitness = 0
        ge.append(g)

        apple = Apple(snake.body)

        score = 0

        run = True
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                    quit()
           
            inputs = snake.getInputs(apple)
            outputs = net.activate(inputs)

            direction = outputs.index(max(outputs))

            snake.changeForce(direction)

            snake.move()
        
            if snake.colApple(apple):
                apple = Apple(snake.body)
                score += 1
                g.fitness += 100
            elif snake.colWall():
                run = False
                g.fitness -= 50
            elif snake.colBody():
                run = False
                g.fitness -= 50

            evalList = [score,pop,gen]

            draw_window_play(win, snake, apple, evalList)
            g.fitness += 0.1
       
        pop += 1

    gen += 1

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
  
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 30)


if __name__ == "__main__":
    local_dir = os.path.dirname("__file__")
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
