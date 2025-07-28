import pygame
import cv2
import numpy as np
import random
import math
import json
import os
from enum import Enum
import mediapipe as mp
import time


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

class GestureDetector:
    def __init__(self, sensitivity=1.0):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.finger_count = 0
        self.hand_angle = 0
        self.is_fist = False
        self.is_open_hand = False
        self.gesture_history = []
        self.angle_history = []
        self.smoothing_window = 8
        self.sensitivity = sensitivity
        self.hand_center = None
        
    def count_fingers(self, landmarks):
        # Finger tip and pip landmarks
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        
        fingers = []
        
        # Thumb (different logic due to orientation)
        if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return sum(fingers)
    
    def calculate_hand_angle(self, landmarks):
        
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        # Calculate angle of hand relative to vertical
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        
        angle = math.degrees(math.atan2(dx, dy))/10#priyanshu
        
        
        angle = max(-90, min(90, angle * self.sensitivity)) #priyanshu 
        
        return angle
    
  
    def detect_hand_state(self, landmarks):
        palm_center = landmarks[9] 
        finger_tips = [4, 8, 12, 16, 20]

        distances = []
        for tip_idx in finger_tips:
            tip = landmarks[tip_idx]
            distance = math.sqrt((tip.x - palm_center.x)**2 + (tip.y - palm_center.y)**2)
            distances.append(distance)

        avg_distance = sum(distances) / len(distances)

        if avg_distance < 0.15:
            return "fist"
        elif avg_distance > 0.25:
            return "open"
        else:
            return "neutral"

    
    def detect_gesture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                
               
                finger_count = self.count_fingers(landmarks)
                
               
                hand_state = self.detect_hand_state(landmarks)
                
               
                angle = self.calculate_hand_angle(landmarks)
                
             
                self.gesture_history.append(hand_state)
                self.angle_history.append(angle)
                
                if len(self.gesture_history) > self.smoothing_window:
                    self.gesture_history.pop(0)
                if len(self.angle_history) > self.smoothing_window:
                    self.angle_history.pop(0)
                
              
                self.is_fist = self.gesture_history.count("fist") > self.smoothing_window // 2
                self.is_open_hand = self.gesture_history.count("open") > self.smoothing_window // 2
                
                
             
                self.hand_angle = sum(self.angle_history) / len(self.angle_history)
                
              
                self.hand_center = (int(landmarks[9].x * frame.shape[1]), 
                                  int(landmarks[9].y * frame.shape[0]))
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw angle indicator
                if self.hand_center:
                    cv2.circle(frame, self.hand_center, 10, (0, 255, 0), -1)
                    
                    # Draw angle line
                    angle_rad = math.radians(self.hand_angle)
                    end_x = int(self.hand_center[0] + 50 * math.sin(angle_rad))
                    end_y = int(self.hand_center[1] - 50 * math.cos(angle_rad))
                    cv2.line(frame, self.hand_center, (end_x, end_y), (255, 0, 0), 3)
                
        else:
            # No hand detected
            self.is_fist = False
            self.is_open_hand = False
            self.hand_angle = 0
            self.hand_center = None
            
        return self.is_fist, self.is_open_hand, self.hand_angle
    
    def get_action_and_steering(self):
        if self.is_fist:
            return "brake", self.hand_angle
        elif self.is_open_hand:
            return "accelerate", self.hand_angle
        else:
            return "coast", self.hand_angle

class InclusiveVelocity:
    def __init__(self):
        self.frame_count = 0
        self.last_gesture_time = 0  # for gesture rate limiting


        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Inclusive Velocity - Gesture Racing")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Game state
        self.state = GameState.MENU
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
        # for gesture‑based menu/settings nav
        self.nav_last_pos = None
        self.nav_cooldown = 0
        self.nav_threshold = 50    # pixels
        self.nav_cooldown_time = 10 # frames
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.gesture_detector = GestureDetector(sensitivity=1.0)
        
      
        self.obstacles = []
        self.collectibles = []
        
        # Game variables
        self.score = 0
        self.distance = 0
        self.coins = 0
        self.game_speed = 2
        self.spawn_timer = 0
        self.last_action = "coast"
        self.current_steering = 0
        
        # settings
        self.high_contrast = False
        self.audio_feedback = True
        self.gesture_sensitivity = 1.0
        # Car-physics adjustable parameters 
        self.default_friction = 0.9        
        self.default_acceleration = 0.3    
        self.default_max_speed = 8         # top speed normally
        self.default_boost_duration = 120  # frames of boost when collected
      
        self.default_turn_speed = 5        # not currently used directly in Car.update but can incorporate if desired
        # Ranges for safety:
       

        # SETTINGS menu state
        self.settings_options = ["Friction", "Acceleration", "Max Speed", "Boost Duration", "Turn Sensitivity", "Back"]
        self.settings_selected = 0
        
        # UI elements
        self.menu_options = ["Start Game", "Calibration", "Settings", "Quit"]
        self.wheel_img = pygame.image.load("steering_wheel.png").convert_alpha()
        self.wheel_img = pygame.transform.smoothscale(self.wheel_img, (80, 80))  # Resize

        self.selected_option = 0
        
       
        self.load_settings()
        
   
    def load_settings(self):
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r") as f:
                    settings = json.load(f)
                    self.high_contrast = settings.get("high_contrast", False)
                    self.audio_feedback = settings.get("audio_feedback", True)
                    self.gesture_sensitivity = settings.get("gesture_sensitivity", 1.0)
                    # Load car-physics settings (with safe defaults if missing)
                    self.default_friction = settings.get("default_friction", self.default_friction)
                    self.default_acceleration = settings.get("default_acceleration", self.default_acceleration)
                    self.default_max_speed = settings.get("default_max_speed", self.default_max_speed)
                    self.default_boost_duration = settings.get("default_boost_duration", self.default_boost_duration)
                    self.default_turn_speed = settings.get("default_turn_speed", self.default_turn_speed)
        except Exception as e:
            print(f"Error loading settings: {e}")

    
  
    def save_settings(self):
        settings = {
            "high_contrast": self.high_contrast,
            "audio_feedback": self.audio_feedback,
            "gesture_sensitivity": self.gesture_sensitivity,
            # Car-physics settings
            "default_friction": self.default_friction,
            "default_acceleration": self.default_acceleration,
            "default_max_speed": self.default_max_speed,
            "default_boost_duration": self.default_boost_duration,
            "default_turn_speed": self.default_turn_speed
        }
        try:
            with open("settings.json", "w") as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving settings: {e}")

    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.state == GameState.MENU:
                    if event.key == pygame.K_UP:
                        self.selected_option = (self.selected_option - 1) % len(self.menu_options)
                    elif event.key == pygame.K_DOWN:
                        self.selected_option = (self.selected_option + 1) % len(self.menu_options)
                    elif event.key == pygame.K_RETURN:
                        self.handle_menu_selection()
                elif self.state == GameState.GAME:
                    if event.key == pygame.K_ESCAPE:
                        self.state = GameState.PAUSE
                elif self.state == GameState.PAUSE:
                    if event.key == pygame.K_ESCAPE:
                        self.state = GameState.GAME
                elif self.state == GameState.GAME_OVER:
                    if event.key == pygame.K_RETURN:
                        self.reset_game()
                        self.state = GameState.MENU
                elif self.state == GameState.SETTINGS:
                    if event.key == pygame.K_UP:
                        # Move selection up
                        self.settings_selected = (self.settings_selected - 1) % len(self.settings_options)
                    elif event.key == pygame.K_DOWN:
                        # Move selection down
                        self.settings_selected = (self.settings_selected + 1) % len(self.settings_options)
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_MINUS:
                        # Decrease selected parameter
                        self.adjust_setting(-1)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        # Increase selected parameter
                        self.adjust_setting(1)
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        # On Enter or ESC, go back to menu and save settings
                        self.save_settings()
                        self.state = GameState.MENU

    
    def handle_menu_selection(self):
        if self.selected_option == 0:  # Start Game
            self.reset_game()
            self.state = GameState.GAME
        elif self.selected_option == 1:  # Calibration
            self.state = GameState.CALIBRATION
        elif self.selected_option == 2:  # Settings
            self.state = GameState.SETTINGS
        elif self.selected_option == 3:  # Quit
            self.running = False
 
    def handle_menu_gestures(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.gesture_detector.detect_gesture(frame)
        center = self.gesture_detector.hand_center
        results = self.gesture_detector.hands.process(rgb_frame)

        # 1) Always handle left‑swipe first (quit/back)
        if center and self.nav_cooldown == 0 and self.nav_last_pos:
            dx = center[0] - self.nav_last_pos[0]
            if dx < -2 * self.nav_threshold:
                self.running = False
                self.nav_cooldown = self.nav_cooldown_time
                self.nav_last_pos = center
                return

        # 2) Menu nav & selection
        if center and self.nav_cooldown == 0:
            # a) Fist = SELECT (highest priority)
            if self.gesture_detector.is_fist:
                self.handle_menu_selection()
                self.nav_cooldown = self.nav_cooldown_time

            # b) Else, tilt up/down to move selection
            else:
                if self.nav_last_pos:
                    dy = center[1] - self.nav_last_pos[1]
                    dx = center[0] - self.nav_last_pos[0]

                    # Vertical motion
                    if abs(dy) > abs(dx) and abs(dy) > 10:
                        if dy < 0:
                            self.selected_option = (self.selected_option - 1) % len(self.menu_options)
                        else:
                            self.selected_option = (self.selected_option + 1) % len(self.menu_options)
                        self.nav_cooldown = self.nav_cooldown_time

            self.nav_last_pos = center

        # 3) Cooldown tick
        if self.nav_cooldown > 0:
            self.nav_cooldown -= 1

  
    def handle_settings_gestures(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.gesture_detector.detect_gesture(frame)
        center = self.gesture_detector.hand_center
        results = self.gesture_detector.hands.process(rgb_frame)

        # 1) Left‑swipe to go back (highest priority)
        if center and self.nav_cooldown == 0 and self.nav_last_pos:
            dx = center[0] - self.nav_last_pos[0]
            if dx < -2 * self.nav_threshold:
                self.save_settings()
                self.state = GameState.MENU
                self.nav_cooldown = self.nav_cooldown_time
                self.nav_last_pos = center
                return

        # 2) Settings nav & adjust
        if center and self.nav_cooldown == 0:
            # a) Fist = enter
            if self.gesture_detector.is_fist:
                if self.settings_options[self.settings_selected] == "Back":
                    self.save_settings()
                    self.state = GameState.MENU
                else:
                    self.adjust_setting(+1)
                self.nav_cooldown = self.nav_cooldown_time

            # b) Else, tilt up/down to move selection
            else:
                if self.nav_last_pos:
                    dy = center[1] - self.nav_last_pos[1]
                    dx = center[0] - self.nav_last_pos[0]

                    if abs(dy) > abs(dx) and abs(dy) > self.nav_threshold:
                        if dy < 0:
                            self.settings_selected = (self.settings_selected - 1) % len(self.settings_options)
                        else:
                            self.settings_selected = (self.settings_selected + 1) % len(self.settings_options)
                        self.nav_cooldown = self.nav_cooldown_time

            self.nav_last_pos = center

        # 3) Cooldown tick
        if self.nav_cooldown > 0:
            self.nav_cooldown -= 1

   


  
    def reset_game(self):
        car = Car(400, 500, self.RED)
        car.friction = self.default_friction
        car.acceleration = self.default_acceleration
        car.max_speed = self.default_max_speed
        car.turn_speed = self.default_turn_speed  # if you're using this
        self.car = car

        self.obstacles = []
        self.collectibles = []
        self.score = 0
        self.distance = 0
        self.coins = 0
        self.game_speed = 2
        self.spawn_timer = 0

    
    def update_game(self):
        # Get camera frame
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror image
            is_fist, is_open_hand, hand_angle = self.gesture_detector.detect_gesture(frame)
        
            self.last_frame = frame.copy()

            action, steering_angle = self.gesture_detector.get_action_and_steering()
            
           
            self.gesture_detector.sensitivity = self.gesture_sensitivity
            
         
            if action != self.last_action and self.audio_feedback:
               
                pass
            
            self.last_action = action
            self.current_steering = steering_angle
            self.car.update(action, steering_angle)
            
            # Update distance and score
            self.distance += self.car.speed * 0.1
            self.score = int(self.distance + self.coins * 10)
            
            # Spawn obstacles and collectibles
            self.spawn_timer += 1
            if self.spawn_timer > max(30 - self.game_speed, 10):
                self.spawn_objects()
                self.spawn_timer = 0
            
            # Update game objects
            for obstacle in self.obstacles[:]:
                obstacle.y += self.game_speed
                if obstacle.y > 650:
                    self.obstacles.remove(obstacle)
                    
                # Collision detection
                if (abs(obstacle.x - self.car.x) < 30 and 
                    abs(obstacle.y - self.car.y) < 30 and 
                    self.car.shield_timer <= 0):
                    if obstacle.type == "pothole":
                        self.car.speed *= 0.5  # Slow down
                    else:
                        self.state = GameState.GAME_OVER
            
            for collectible in self.collectibles[:]:
                collectible.update()
                collectible.y += self.game_speed
                if collectible.y > 650:
                    self.collectibles.remove(collectible)
                    
                # Collection detection
                if (abs(collectible.x - self.car.x) < 25 and 
                    abs(collectible.y - self.car.y) < 25):
                    if collectible.type == "coin":
                        self.coins += 1
                  
                    elif collectible.type == "boost":
                        self.car.boost_timer = self.default_boost_duration
                       
                        boosted_speed = self.default_max_speed * 1.5
                       
                        boosted_speed = min(boosted_speed, 30)
                        self.car.max_speed = boosted_speed

                    elif collectible.type == "shield":
                        self.car.shield_timer = 180
                    collectible.active = False
                    self.collectibles.remove(collectible)
            
            # Increase difficulty
            if int(self.distance) % 100 == 0 and self.distance > 0:
                self.game_speed = min(self.game_speed + 0.1, 8)
            
          
            if self.car.boost_timer <= 0:
                self.car.max_speed = self.default_max_speed

    
    def spawn_objects(self):
        x = random.randint(50, 750)
        y = -50
        
        # 70% chance obstacle, 30% chance collectible
        if random.random() < 0.7:
            obstacle_type = random.choice(["cone", "pothole", "roadblock"])
            self.obstacles.append(Obstacle(x, y, obstacle_type))
        else:
            collectible_type = random.choice(["coin", "boost", "shield"])
            self.collectibles.append(Collectible(x, y, collectible_type))
    
    def draw_menu(self):
        bg_color = self.BLACK if self.high_contrast else (50, 50, 100)
        self.screen.fill(bg_color)
     
        title = self.font.render("INCLUSIVE VELOCITY", True, self.WHITE)
        title_rect = title.get_rect(center=(400, 100))
        self.screen.blit(title, title_rect)
        
        subtitle = self.small_font.render("Gesture-Controlled Racing", True, self.WHITE)
        subtitle_rect = subtitle.get_rect(center=(400, 140))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Menu options
        for i, option in enumerate(self.menu_options):
            color = self.GREEN if i == self.selected_option else self.WHITE
            text = self.font.render(option, True, color)
            text_rect = text.get_rect(center=(400, 250 + i * 60))
            self.screen.blit(text, text_rect)
        
        # Instructions
        instructions = [
            "Hand Gestures:",
            "✊ Fist = Brake",
            "✋ Open Hand = Accelerate", 
            "↔ Tilt Hand = Steer Left/Right",
            "👌 No Gesture = Coast",
            "",
            "Sensitivity can be adjusted in calibration"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.WHITE)
            self.screen.blit(text, (50, 450 + i * 25))
        ret, frame = self.cap.read()
      
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
          
            self.gesture_detector.detect_gesture(frame)
          
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0,1))
            small = pygame.transform.scale(frame_surface, (160, 120))
            # Position: bottom-right
            x = 800 - 170
            y = 600 - 130
            self.screen.blit(small, (x, y))
            pygame.draw.rect(self.screen, self.WHITE, (x, y, 160, 120), 2)
            lbl = self.small_font.render("Camera", True, self.WHITE)
            self.screen.blit(lbl, (x, y - 20))


    
    def draw_calibration(self):
        self.screen.fill(self.BLACK)
        
       
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            is_fist, is_open_hand, hand_angle = self.gesture_detector.detect_gesture(frame)
            
           
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
           
            scaled_frame = pygame.transform.scale(frame_surface, (400, 300))
            self.screen.blit(scaled_frame, (200, 50))
            
            # Display gesture status
            fist_color = self.GREEN if is_fist else self.WHITE
            open_color = self.GREEN if is_open_hand else self.WHITE
            
            fist_text = self.font.render(f"FIST (Brake): {'✓' if is_fist else '✗'}", True, fist_color)
            self.screen.blit(fist_text, (50, 380))
            
            open_text = self.font.render(f"OPEN HAND (Accelerate): {'✓' if is_open_hand else '✗'}", True, open_color)
            self.screen.blit(open_text, (50, 420))
            
            angle_text = self.font.render(f"Hand Angle (Steering): {hand_angle:.1f}°", True, self.WHITE)
            self.screen.blit(angle_text, (50, 460))
            
            # Visual steering indicator
            center_x = 400
            indicator_y = 500
            pygame.draw.line(self.screen, self.GRAY, (center_x - 100, indicator_y), (center_x + 100, indicator_y), 5)
            
          
            angle_pos = center_x + (hand_angle *0.2) 
            pygame.draw.circle(self.screen, self.GREEN, (int(angle_pos), indicator_y), 10)
            
           
            sens_text = self.font.render(f"Sensitivity: {self.gesture_sensitivity:.1f}", True, self.WHITE)
            self.screen.blit(sens_text, (50, 520))
            
            sens_help = self.small_font.render("Press +/- to adjust sensitivity", True, self.WHITE)
            self.screen.blit(sens_help, (50, 550))
        
      
        title = self.font.render("Gesture Calibration & Testing", True, self.WHITE)
        self.screen.blit(title, (225, 20))
        
        instructions = [
            "• Make a FIST to brake",
            "• Show OPEN HAND to accelerate", 
            "• TILT your hand left/right to steer",
            "• Ensure good lighting for best results",
            "",
            "",
            "",
            "Press ESC to return to menu"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, self.WHITE)
            self.screen.blit(text, (500, 380 + i * 25))
    
    def draw_settings(self):
        # Background
        bg_color = self.BLACK if self.high_contrast else (30, 30, 30)
        self.screen.fill(bg_color)
      


        title = self.font.render("Settings", True, self.WHITE)
        title_rect = title.get_rect(center=(400, 50))
        self.screen.blit(title, title_rect)

        for idx, opt in enumerate(self.settings_options):
            is_selected = (idx == self.settings_selected)
            color = self.GREEN if is_selected else self.WHITE

            # Determine display value
            if opt == "Friction":
                val = f"{self.default_friction:.2f}"
            elif opt == "Acceleration":
                val = f"{self.default_acceleration:.2f}"
            elif opt == "Max Speed":
                val = f"{self.default_max_speed:.1f}"
            elif opt == "Boost Duration":
                val = str(self.default_boost_duration)
            elif opt == "Turn Sensitivity":
                val = f"{self.default_turn_speed:.1f}"
            else:  # "Back"
                val = ""
            text = self.font.render(f"{opt}: {val}", True, color)
            # Position: vertically spaced
            self.screen.blit(text, (200, 150 + idx * 50))
    def adjust_setting(self, direction):
        """
        direction: +1 to increase, -1 to decrease.
        Adjust the currently selected setting within safe bounds.
        """
        opt = self.settings_options[self.settings_selected]
      
        if opt == "Friction":
           
            step = 0.01 * direction
            new = self.default_friction + step
            self.default_friction = max(0.7, min(0.99, new))
        elif opt == "Acceleration":
           
            step = 0.05 * direction
            new = self.default_acceleration + step
            self.default_acceleration = max(0.1, min(1.0, new))
        elif opt == "Max Speed":
           
            step = 1 * direction
            new = self.default_max_speed + step
            self.default_max_speed = max(4, min(20, new))
        elif opt == "Boost Duration":
          
            step = 30 * direction
            new = self.default_boost_duration + step
            self.default_boost_duration = max(60, min(300, new))
        elif opt == "Turn Sensitivity":
            # turn_speed or steering sensitivity [1, 10], step 0.5
            step = 0.5 * direction
            new = self.default_turn_speed + step
            self.default_turn_speed = max(1.0, min(10.0, new))
        elif opt == "Back":
           
            pass

       
        if self.state == GameState.SETTINGS and hasattr(self, 'car'):
          
            if opt == "Friction":
                self.car.friction = self.default_friction
            elif opt == "Acceleration":
                self.car.acceleration = self.default_acceleration
            elif opt == "Max Speed":
               
                self.car.max_speed = self.default_max_speed
            elif opt == "Turn Sensitivity":
                self.car.turn_speed = self.default_turn_speed
            

    def draw_game(self):
      
        bg_color = self.WHITE if self.high_contrast else (100, 150, 100)
        self.screen.fill(bg_color)
        
        # Draw road
        road_color = self.BLACK if self.high_contrast else (80, 80, 80)
        pygame.draw.rect(self.screen, road_color, (100, 0, 600, 600))
        
        # Road lines
        line_color = self.WHITE
        for y in range(0, 600, 40):
            pygame.draw.rect(self.screen, line_color, (395, y, 10, 20))
        
        # Draw game objects
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        
        for collectible in self.collectibles:
            collectible.draw(self.screen)
        
        self.car.draw(self.screen)
        
        # UI
        score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
        self.screen.blit(score_text, (10, 10))
        
        distance_text = self.font.render(f"Distance: {int(self.distance)}m", True, self.BLACK)
        self.screen.blit(distance_text, (10, 50))
        
        coins_text = self.font.render(f"Coins: {self.coins}", True, self.BLACK)
        self.screen.blit(coins_text, (10, 90))
        
        speed_text = self.font.render(f"Speed: {int(self.car.speed)}", True, self.BLACK)
        self.screen.blit(speed_text, (10, 130))
        
        # Gesture feedback
        gesture_status = ""
        if self.gesture_detector.is_fist:
            gesture_status = "FIST (Braking)"
            action_color = self.RED
        elif self.gesture_detector.is_open_hand:
            gesture_status = "OPEN HAND (Accelerating)"
            action_color = self.GREEN
        else:
            gesture_status = "NO GESTURE (Coasting)"
            action_color = self.GRAY
            
        action_text = self.font.render(f"Gesture: {gesture_status}", True, action_color)
        self.screen.blit(action_text, (10, 170))
        
        # Steering feedback
        steer_direction = "CENTER"
        if self.current_steering > 5:
            steer_direction = f"RIGHT ({self.current_steering:.1f}°)"
        elif self.current_steering < -5:
            steer_direction = f"LEFT ({self.current_steering:.1f}°)"
            
        steer_text = self.small_font.render(f"Steering: {steer_direction}", True, self.BLACK)
        self.screen.blit(steer_text, (10, 210))

        
        # Visual steering indicator
        pygame.draw.rect(self.screen, self.GRAY, (10, 230, 200, 20), 2)
        steer_pos = 110 + (self.current_steering * 2)  
        steer_pos = max(15, min(205, steer_pos))  
        pygame.draw.circle(self.screen, self.BLUE, (int(steer_pos), 240), 8)
        
       
        if self.car.boost_timer > 0:
            boost_text = self.small_font.render("BOOST ACTIVE!", True, (255, 255, 0))
            self.screen.blit(boost_text, (600, 10))
            
        if self.car.shield_timer > 0:
            shield_text = self.small_font.render("SHIELD ACTIVE!", True, (0, 255, 255))
            self.screen.blit(shield_text, (600, 35))
      
        wheel_center = (700, 500)
        rotated_wheel = pygame.transform.rotate(self.wheel_img, -self.current_steering)  # Negative to match direction
        rect = rotated_wheel.get_rect(center=wheel_center)
        self.screen.blit(rotated_wheel, rect)


        # Label
        label = self.small_font.render("Steering Wheel", True, self.BLACK)
        self.screen.blit(label, (wheel_center[0] - 50, wheel_center[1] + 50))
       
        if hasattr(self, 'last_frame'):
            frame = self.last_frame
          
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
           
            small = pygame.transform.scale(frame_surface, (160, 120))
           
            x = 800 - 170 
            y = 10        
            self.screen.blit(small, (x, y))
          
            border_color = self.BLACK if self.high_contrast else self.WHITE
            pygame.draw.rect(self.screen, border_color, (x, y, 160, 120), 2)
            # Label
            lbl = self.small_font.render("Camera", True, border_color)
            self.screen.blit(lbl, (x, y - 20))


    
    def draw_game_over(self):
        self.screen.fill(self.BLACK)
        
        # Game Over text
        game_over_text = self.font.render("GAME OVER", True, self.RED)
        game_over_rect = game_over_text.get_rect(center=(400, 200))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Final stats
        stats = [
            f"Final Score: {self.score}",
            f"Distance: {int(self.distance)}m", 
            f"Coins Collected: {self.coins}",
            "",
            "Press ENTER to return to menu"
        ]
        
        for i, stat in enumerate(stats):
            color = self.WHITE if stat else self.WHITE
            text = self.font.render(stat, True, color)
            text_rect = text.get_rect(center=(400, 280 + i * 40))
            self.screen.blit(text, text_rect)
    
    def draw_pause(self):
      
        self.draw_game()
        
       
        overlay = pygame.Surface((800, 600))
        overlay.set_alpha(128)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Pause text
        pause_text = self.font.render("PAUSED", True, self.WHITE)
        pause_rect = pause_text.get_rect(center=(400, 280))
        self.screen.blit(pause_text, pause_rect)
        
        resume_text = self.small_font.render("Press ESC to resume", True, self.WHITE)
        resume_rect = resume_text.get_rect(center=(400, 320))
        self.screen.blit(resume_text, resume_rect)
    
    def run(self):
        while self.running:
            self.handle_events()
            
          
            if self.state == GameState.MENU:
                self.handle_menu_gestures()
            elif self.state == GameState.SETTINGS:
                self.handle_settings_gestures()

            
            if self.state == GameState.GAME:
                self.update_game()
           
            if self.state == GameState.MENU:

                self.draw_menu()
            elif self.state == GameState.CALIBRATION:
                self.draw_calibration()
               
                keys = pygame.key.get_pressed()
                if keys[pygame.K_ESCAPE]:
                    self.state = GameState.MENU
                elif keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:  # Increase sensitivity
                    self.gesture_sensitivity = min(2.0, self.gesture_sensitivity + 0.1)
                elif keys[pygame.K_MINUS]:  # Decrease sensitivity
                    self.gesture_sensitivity = max(0.1, self.gesture_sensitivity - 0.1) #priyanshu
            elif self.state == GameState.GAME:
                self.draw_game()
                
            elif self.state == GameState.PAUSE:
                self.draw_pause()
            elif self.state == GameState.GAME_OVER:
                self.draw_game_over()
            elif self.state == GameState.SETTINGS:
                self.draw_settings()

            
            pygame.display.flip()
            self.clock.tick(60)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    try:
        game = InclusiveVelocity()
        game.run()
    except Exception as e:
        print(f"Error running game: {e}")
        print("Make sure you have a camera connected and the required libraries installed:")
        print("pip install pygame opencv-python mediapipe numpy")