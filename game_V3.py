import pygame
import cv2
import numpy as np
import random
import math
import json
import os
from enum import Enum
import mediapipe as mp

pygame.init()
pygame.mixer.init()


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GameState(Enum):
    MENU = 1
    CALIBRATION = 2
    SETTINGS = 3
    CAR_SELECT = 4
    TRACK_SELECT = 5
    GAME = 6
    PAUSE = 7
    GAME_OVER = 8

class Car:
    def __init__(self, x, y, color=(255, 0, 0)):
        self.x = x
        self.y = y
        self.speed = 0
        self.max_speed = 8
        self.acceleration = 0.3
        self.friction = 0.9
        self.turn_speed = 5
        self.width = 40
        self.height = 20
        self.color = color
        self.angle = 0
        self.boost_timer = 0
        self.shield_timer = 0
        
    def update(self, action, steering_angle):
       
        if action == "accelerate":
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        elif action == "brake":
            self.speed = max(self.speed - self.acceleration * 2, 0)
        elif action == "coast":
           
            self.speed *= 0.98
            
       
        if self.speed > 0:
           
            steering_factor = min(self.speed / self.max_speed, 1.0)
            self.angle += (steering_angle * 0.3 * steering_factor)
            
       
        self.speed *= self.friction
        
     
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        
      
        self.x = max(self.width//2, min(800 - self.width//2, self.x))
        self.y = max(self.height//2, min(600 - self.height//2, self.y))
        
       
        if self.boost_timer > 0:
            self.boost_timer -= 1
        if self.shield_timer > 0:
            self.shield_timer -= 1

    def draw(self, screen):
       
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        
        if self.shield_timer > 0:
            color = (0, 255, 255)  # Cyan for shield
        elif self.boost_timer > 0:
            color = (255, 255, 0)  # Yellow for boost
        else:
            color = self.color
            
        pygame.draw.rect(car_surface, color, (0, 0, self.width, self.height))
        pygame.draw.rect(car_surface, (100, 100, 100), (0, 0, self.width, self.height), 2)
        
       
        pygame.draw.rect(car_surface, (200, 200, 200), (5, 5, 10, 10))
        pygame.draw.rect(car_surface, (200, 200, 200), (25, 5, 10, 10))
        
       
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        screen.blit(rotated_car, rect)

class Obstacle:
    def __init__(self, x, y, obstacle_type="cone"):
        self.x = x
        self.y = y
        self.type = obstacle_type
        self.width = 30
        self.height = 30
        self.active = True
        
    def draw(self, screen):
        if not self.active:
            return
            
        if self.type == "cone":
            color = (255, 165, 0)  # Orange
            pygame.draw.polygon(screen, color, [(self.x, self.y), 
                                               (self.x-15, self.y+30), 
                                               (self.x+15, self.y+30)])
        elif self.type == "pothole":
            color = (50, 50, 50)  # Dark gray
            pygame.draw.ellipse(screen, color, (self.x-15, self.y-15, 30, 30))
        elif self.type == "roadblock":
            color = (139, 69, 19)  # Brown
            pygame.draw.rect(screen, color, (self.x-20, self.y-10, 40, 20))

class Collectible:
    def __init__(self, x, y, collectible_type="coin"):
        self.x = x
        self.y = y
        self.type = collectible_type
        self.active = True
        self.animation_timer = 0
        
    def update(self):
        self.animation_timer += 1
        
    def draw(self, screen):
        if not self.active:
            return
            
        offset = math.sin(self.animation_timer * 0.1) * 3
        
        if self.type == "coin":
            color = (255, 215, 0)  # Gold
            pygame.draw.circle(screen, color, (int(self.x), int(self.y + offset)), 10)
            pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y + offset)), 10, 2)
        elif self.type == "boost":
            color = (255, 0, 255)  # Magenta
            pygame.draw.polygon(screen, color, [(self.x, self.y + offset - 10),
                                               (self.x - 8, self.y + offset + 10),
                                               (self.x + 8, self.y + offset + 10)])
        elif self.type == "shield":
            color = (0, 255, 255)  # Cyan
            pygame.draw.circle(screen, color, (int(self.x), int(self.y + offset)), 12, 3)

    