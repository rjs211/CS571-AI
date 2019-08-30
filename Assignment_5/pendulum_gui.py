import sys
import pygame
from pygame.locals import USEREVENT, QUIT
from math import sin, cos, radians
from fuzzy_pendulum import FuzzyPendulum

pygame.init()
refresh_rate = 1
bob_size = 15
pos_theta_eps = [3, 0, 2, 5]
pos_omega_eps = [4, 0, 4, 8]
pos_alpha_eps = [1, 0, 2, 4, 3, 5, 6]
window = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Fuzzy Logic Guided Inverted Pendulum")
screen = pygame.display.get_surface()
screen.fill((255, 255, 255))
pivot = (400, 720)
pendulum_length = 280


class BobMass(pygame.sprite.Sprite):
    """
    Class for defining the pendulum object and rendering its motion
    under the given physical conditions.
    """
    def __init__(self, model):
        """
        Function for initializing the pendulum class.
        """
        pygame.sprite.Sprite.__init__(self)
        self.theta = 0.8
        self.omega = -3
        self.model = model
        self.rect = pygame.Rect(int(pivot[0] - pendulum_length * cos(self.theta)),
                                int(pivot[1] - pendulum_length * sin(self.theta)), 1, 1)
        self.draw()

    def recompute_angle(self):
        """
        Function for recomputing the angle that the pendulum makes
        with the vertical.
        """
        t = refresh_rate / 1000
        self.theta, self.omega = self.model.get_new_theta_omega(self.theta, self.omega, t)
        #print(self.theta, self.omega)
        self.rect = pygame.Rect(pivot[0] - pendulum_length * sin(self.theta),
                                pivot[1] - pendulum_length * cos(self.theta), 1, 1)

    def draw(self):
        """
        Function for drawing the pendulum in its current state.
        """
        pygame.draw.circle(screen, (0, 0, 0), pivot, 5, 0)
        pygame.draw.circle(screen, (0, 0, 0), self.rect.center, bob_size, 0)
        pygame.draw.aaline(screen, (0, 0, 0), pivot, self.rect.center)
        pygame.draw.line(screen, (0, 0, 0), (0, pivot[1]), (800, pivot[1]))

    def update(self):
        """
        Function for updating the current physical state of
        the pendulum.
        """
        self.recompute_angle()
        screen.fill((255, 255, 255))
        self.draw()


def event_input(events):
    """
    Function for registering user events in the event queue
    and updating the pendulum accordingly.
    """
    for event in events:
        if event.type == QUIT:
            sys.exit(0)
        elif event.type == tick:
            bob.update()


physics = FuzzyPendulum(use_gravity = True, pos_theta_eps = pos_theta_eps, pos_omega_eps = pos_omega_eps, pos_alpha_eps = pos_alpha_eps)
bob = BobMass(physics)
clock = pygame.time.Clock()
tick = USEREVENT
pygame.time.set_timer(tick, refresh_rate)


while True:
    event_input(pygame.event.get())
    pygame.display.flip()