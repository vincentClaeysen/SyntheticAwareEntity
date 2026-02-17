import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import time
import random

# ---------------------------------------------------------
# SYNTHETIC AWARENESS SYSTEM - "Gelly Mask"
# Version: Sleep/Dream/Nightmare + Configurable Background
# ---------------------------------------------------------

WIDTH, HEIGHT = 600, 600
FPS = 25 
PARTICLES_DENSITY = 2800 
TRANSITION_SENSORS = 10 
TRANSITION_STRESS = 3
MORPH_FRAMES_M = 100
STRESS_RETRACTION_PCT = 0.75
FEL_DECAY_STEP = 0.005

SLEEP_TRANSITION_DURATION = 2.0
WAKE_TRANSITION_DURATION = 1.0

# ============================================================
# CONFIGURATION FOND
# ============================================================
BACKGROUND_COLORS = [
    (0.0, 0.005, 0.02),
    (0.01, 0.02, 0.05)
]

# ============================================================
# PALETTES ÉMOTIONNELLES
# ============================================================

MEMBRANE_A = [(0.0, 0.4, 1.0), (0.5, 0.2, 0.8), (0.3, 0.0, 0.7)] 
CORE_A     = [(0.8, 0.1, 0.6), (0.5, 0.0, 0.5), (0.4, 0.1, 0.8)]
MEMBRANE_B = [(1.2, 0.8, 0.0), (1.5, 1.2, 0.2), (1.8, 0.6, 0.0)]
CORE_B     = [(1.3, 0.1, 0.1), (1.0, 0.3, 0.0), (0.8, 0.4, 0.0)]

MEMBRANE_GLACIER = [(0.6, 1.2, 3.5), (0.4, 1.0, 3.0), (0.3, 0.8, 2.5)]
CORE_GLACIER     = [(0.4, 1.0, 3.0), (0.3, 0.8, 2.5), (0.2, 0.6, 2.0)]
MEMBRANE_LAVA    = [(4.0, 0.4, 0.0), (3.5, 0.3, 0.0), (3.0, 0.2, 0.0)]
CORE_LAVA        = [(4.5, 0.3, 0.1), (4.0, 0.2, 0.0), (3.5, 0.1, 0.0)]

PAL_SMILE_MEM      = [(2.5, 2.0, 0.9), (2.2, 1.8, 0.7), (1.9, 1.5, 0.5)]
PAL_SMILE_CORE     = [(2.8, 2.2, 1.0), (2.5, 1.9, 0.8), (2.2, 1.6, 0.6)]
PAL_LAUGHTER_MEM   = [(3.2, 2.4, 0.8), (2.8, 2.0, 0.6), (2.4, 1.7, 0.9)]
PAL_LAUGHTER_CORE  = [(3.5, 2.6, 0.9), (3.0, 2.2, 0.7), (2.6, 1.8, 1.0)]
PAL_EUPHORIA_MEM   = [(3.5, 1.8, 3.0), (3.0, 2.0, 2.5), (2.5, 1.5, 2.0)]
PAL_EUPHORIA_CORE  = [(4.0, 2.0, 3.5), (3.5, 2.2, 3.0), (3.0, 1.8, 2.5)]
PAL_FELICITY_MEM   = [(3.5, 2.5, 1.0), (3.0, 2.2, 0.7), (2.5, 1.8, 0.4)]
PAL_FELICITY_CORE  = [(4.0, 2.8, 1.2), (3.5, 2.5, 0.9), (3.0, 2.0, 0.6)]
PAL_EXCITEMENT_MEM = [(3.0, 2.5, 2.5), (2.5, 2.0, 2.0), (2.0, 1.8, 1.8)]
PAL_EXCITEMENT_CORE= [(3.5, 3.0, 3.0), (3.0, 2.5, 2.5), (2.5, 2.0, 2.0)]

PAL_SADNESS_MEM    = [(0.4, 0.6, 1.8), (0.3, 0.5, 1.5), (0.2, 0.4, 1.2)]
PAL_SADNESS_CORE   = [(0.3, 0.5, 1.5), (0.2, 0.4, 1.2), (0.15, 0.3, 1.0)]
PAL_TEARS_MEM      = [(0.5, 0.9, 2.5), (0.4, 0.7, 2.2), (0.3, 0.6, 1.9)]
PAL_TEARS_CORE     = [(0.4, 0.7, 2.2), (0.3, 0.6, 1.9), (0.2, 0.5, 1.6)]
PAL_MELANCHOLY_MEM = [(0.7, 0.6, 1.8), (0.6, 0.5, 1.5), (0.5, 0.4, 1.2)]
PAL_MELANCHOLY_CORE= [(0.6, 0.5, 1.6), (0.5, 0.4, 1.3), (0.4, 0.3, 1.0)]
PAL_DEPRESSION_MEM = [(0.3, 0.3, 0.5), (0.2, 0.2, 0.4), (0.15, 0.15, 0.3)]
PAL_DEPRESSION_CORE= [(0.2, 0.2, 0.4), (0.15, 0.15, 0.3), (0.1, 0.1, 0.25)]

PAL_TERROR_MEM     = [(4.0, 4.0, 4.5), (3.5, 3.5, 4.0), (3.0, 3.0, 3.5)]
PAL_TERROR_CORE    = [(4.5, 4.5, 5.0), (4.0, 4.0, 4.5), (3.5, 3.5, 4.0)]
PAL_FEAR_MEM       = [(0.2, 0.2, 0.6), (0.15, 0.15, 0.5), (0.3, 0.3, 0.8)]
PAL_FEAR_CORE      = [(0.25, 0.2, 0.5), (0.15, 0.1, 0.4), (0.1, 0.05, 0.3)]
PAL_ANXIETY_MEM    = [(0.6, 1.0, 0.6), (0.5, 0.8, 0.5), (0.4, 0.6, 0.4)]
PAL_ANXIETY_CORE   = [(0.5, 0.8, 0.5), (0.4, 0.6, 0.4), (0.3, 0.5, 0.3)]
PAL_DESPAIR_MEM    = [(0.2, 0.1, 0.2), (0.15, 0.05, 0.15), (0.1, 0.02, 0.1)]
PAL_DESPAIR_CORE   = [(0.15, 0.05, 0.15), (0.1, 0.02, 0.1), (0.08, 0.0, 0.08)]

PAL_ANGER_MEM      = [(3.5, 0.4, 0.0), (3.2, 0.3, 0.0), (2.9, 0.5, 0.2)]
PAL_ANGER_CORE     = [(4.0, 0.3, 0.0), (3.7, 0.2, 0.0), (3.4, 0.4, 0.1)]
PAL_RAGE_MEM       = [(4.0, 0.5, 0.0), (3.5, 0.8, 0.3), (3.0, 0.4, 0.0)]
PAL_RAGE_CORE      = [(4.5, 0.4, 0.0), (4.0, 0.9, 0.4), (3.5, 0.3, 0.0)]

PAL_PAIN_MEM       = [(10.0, 10.0, 10.0), (9.0, 9.0, 9.0), (8.0, 8.0, 8.0)]
PAL_PAIN_CORE      = [(12.0, 12.0, 12.0), (11.0, 11.0, 11.0), (10.0, 10.0, 10.0)]

PAL_SLEEP_MEM      = [(0.5, 0.8, 2.0), (0.4, 0.6, 1.6), (0.3, 0.5, 1.3)]
PAL_SLEEP_CORE     = [(0.4, 0.6, 1.6), (0.3, 0.5, 1.3), (0.25, 0.4, 1.0)]

SCALE_MIN, SCALE_MAX = 1.0, 1.32
EXPANSION_PERIOD = 4.0

def lerp(a, b, f): return a * (1 - f) + b * f

def lerp_palette(p1, p2, f):
    return [tuple(p1[i][j] * (1-f) + p2[i][j] * f for j in range(3)) for i in range(len(p1))]

def get_gradient_color(t, seed, boost, palette, modifiers=None):
    n = len(palette)
    cycle = abs(t * 0.4 + seed * 3.0) % n
    idx = int(cycle)
    f = cycle - idx
    c1, c2 = palette[idx % n], palette[(idx + 1) % n]
    color = [((c1[i]*(1-f) + c2[i]*f) * boost) for i in range(3)]
    
    if modifiers:
        if 'brightness' in modifiers:
            for i in range(3):
                color[i] *= modifiers['brightness']
        if 'desaturation' in modifiers:
            gray = sum(color) / 3.0
            for i in range(3):
                color[i] = lerp(color[i], gray, modifiers['desaturation'])
        if 'tint' in modifiers:
            tint = modifiers['tint']
            for i in range(3):
                color[i] = lerp(color[i], tint[i], modifiers.get('tint_strength', 0.5))
        if 'dimming' in modifiers:
            for i in range(3):
                color[i] *= (1.0 - modifiers['dimming'])
    
    return tuple(color)

# ============================================================
# FOND PARAMÉTRABLE
# ============================================================

class ConfigurableBackground:
    """Fond paramétrable : uni ou dégradé 2-3 couleurs"""
    
    def __init__(self, colors):
        self.colors = colors
        self.num_colors = len(colors)
    
    def draw(self, light_level=1.0, sleep_factor=0.0):
        """Dessine le fond (uni ou dégradé)"""
        brightness = light_level * max(0.3, 1.0 - sleep_factor * 0.6)
        
        modulated_colors = [
            tuple(c * brightness for c in color)
            for color in self.colors
        ]
        
        if self.num_colors == 1:
            glClearColor(modulated_colors[0][0], modulated_colors[0][1], 
                        modulated_colors[0][2], 1.0)
        else:
            glDisable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(-1, 1, -1, 1, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            glBegin(GL_QUADS)
            
            if self.num_colors == 2:
                glColor3f(*modulated_colors[0])
                glVertex2f(-1, -1)
                glVertex2f(1, -1)
                
                glColor3f(*modulated_colors[1])
                glVertex2f(1, 1)
                glVertex2f(-1, 1)
            
            elif self.num_colors == 3:
                glColor3f(*modulated_colors[0])
                glVertex2f(-1, -1)
                glVertex2f(1, -1)
                
                glColor3f(*modulated_colors[1])
                glVertex2f(1, 0)
                glVertex2f(-1, 0)
            
            glEnd()
            
            if self.num_colors == 3:
                glBegin(GL_QUADS)
                glColor3f(*modulated_colors[1])
                glVertex2f(-1, 0)
                glVertex2f(1, 0)
                
                glColor3f(*modulated_colors[2])
                glVertex2f(1, 1)
                glVertex2f(-1, 1)
                glEnd()
            
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)

# ============================================================
# AURA GALACTIQUE (rêve)
# ============================================================

class GalacticAura:
    """Halo violet galactique autour de l'avatar endormi"""
    
    def __init__(self):
        self.particles = []
        for _ in range(60):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(1.8, 2.5)
            speed = random.uniform(0.01, 0.03)
            brightness = random.uniform(0.3, 0.8)
            size = random.uniform(2.0, 5.0)
            self.particles.append({
                'angle': angle,
                'radius': radius,
                'speed': speed,
                'brightness': brightness,
                'size': size,
                'phase': random.uniform(0, 6.28)
            })
    
    def draw(self, t, sleep_factor, dream_active, dream_intensity):
        """Dessine l'aura galactique violette"""
        if sleep_factor < 0.3:
            return
        
        glPushMatrix()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        base_color = (0.6, 0.3, 1.2)
        
        for p in self.particles:
            p['angle'] += p['speed'] * (1.0 + dream_active * dream_intensity * 2.0)
            
            x = math.cos(p['angle']) * p['radius']
            y = math.sin(p['angle']) * p['radius']
            z = -3.0
            
            pulse = math.sin(t * 2.0 + p['phase']) * 0.3 + 0.7
            
            if dream_active:
                pulse *= (1.0 + dream_intensity * 0.5)
            
            size = p['size'] * sleep_factor * pulse
            brightness = p['brightness'] * sleep_factor * pulse
            
            glPointSize(size)
            glBegin(GL_POINTS)
            glColor4f(
                base_color[0] * brightness,
                base_color[1] * brightness,
                base_color[2] * brightness,
                0.6 * sleep_factor
            )
            glVertex3f(x, y, z)
            glEnd()
        
        glDisable(GL_BLEND)
        glPopMatrix()

# ============================================================
# AURA CAUCHEMAR (ombres rampantes)
# ============================================================

class NightmareAura:
    """Ombres rampantes et figures menaçantes autour de l'avatar"""
    
    def __init__(self):
        self.shadows = []
        for _ in range(80):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(1.5, 3.0)
            speed = random.uniform(-0.05, -0.02)
            size = random.uniform(3.0, 8.0)
            threat_phase = random.uniform(0, 6.28)
            self.shadows.append({
                'angle': angle,
                'radius': radius,
                'speed': speed,
                'size': size,
                'threat_phase': threat_phase,
                'approach_speed': random.uniform(0.02, 0.08)
            })
    
    def draw(self, t, sleep_factor, nightmare_active, nightmare_intensity):
        """Dessine les ombres menaçantes du cauchemar"""
        if sleep_factor < 0.3 or not nightmare_active:
            return
        
        glPushMatrix()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        threat_color = (0.8, 0.0, 0.0)
        
        for s in self.shadows:
            s['angle'] += s['speed'] * (1.0 + nightmare_intensity)
            
            s['radius'] -= s['approach_speed'] * nightmare_intensity * 0.01
            if s['radius'] < 1.2:
                s['radius'] = 3.0
            
            x = math.cos(s['angle']) * s['radius']
            y = math.sin(s['angle']) * s['radius']
            z = -3.0
            
            threat_pulse = abs(math.sin(t * 5.0 + s['threat_phase'])) ** 0.3
            threat_pulse *= (1.0 + math.sin(t * 13.0 + s['threat_phase'] * 2) * 0.4)
            threat_pulse *= nightmare_intensity
            
            size = s['size'] * sleep_factor * threat_pulse * (0.8 + random.random() * 0.4)
            
            glPointSize(size)
            glBegin(GL_POINTS)
            glColor4f(
                threat_color[0] * threat_pulse,
                threat_color[1],
                threat_color[2],
                0.5 * sleep_factor * threat_pulse
            )
            glVertex3f(x, y, z)
            glEnd()
        
        glDisable(GL_BLEND)
        glPopMatrix()

class EmotionalMask:
    INPUT_TO_EMOTION = {
        "joy": ["smile", "laughter", "excitement"],
        "sadness": ["sadness", "tears", "melancholy"],
        "anger": ["anger", "rage"],
        "fear": ["fear", "terror"]
    }
    
    @staticmethod
    def compute_displayed_emotion(internal_emotion, internal_intensity, emotion_inputs):
        max_input = 0.0
        max_input_type = None
        
        for emotion_type in ["joy", "sadness", "anger", "fear"]:
            if emotion_inputs.get(emotion_type, 0.0) > max_input:
                max_input = emotion_inputs[emotion_type]
                max_input_type = emotion_type
        
        MASKING_THRESHOLD = 0.2
        
        if max_input < MASKING_THRESHOLD:
            return internal_emotion, internal_intensity, False, 0.0
        
        possible_emotions = EmotionalMask.INPUT_TO_EMOTION.get(max_input_type, [internal_emotion])
        
        if max_input > 0.8:
            displayed = possible_emotions[-1] if len(possible_emotions) > 2 else possible_emotions[-1]
        elif max_input > 0.5:
            displayed = possible_emotions[1] if len(possible_emotions) > 1 else possible_emotions[0]
        else:
            displayed = possible_emotions[0]
        
        emotion_distance = EmotionalMask._compute_emotion_distance(internal_emotion, displayed)
        effort = emotion_distance * max_input * internal_intensity
        
        return displayed, max_input, True, effort
    
    @staticmethod
    def _compute_emotion_distance(emotion1, emotion2):
        if emotion1 == emotion2:
            return 0.0
        
        opposites = [
            ("despair", "euphoria"), ("despair", "excitement"), ("despair", "laughter"),
            ("depression", "euphoria"), ("sadness", "euphoria"), ("tears", "laughter"),
            ("terror", "euphoria"), ("fear", "excitement"), ("anxiety", "felicity"),
            ("anger", "sadness"), ("rage", "melancholy")
        ]
        
        for e1, e2 in opposites:
            if (emotion1 == e1 and emotion2 == e2) or (emotion1 == e2 and emotion2 == e1):
                return 1.0
        
        close_pairs = [
            ("smile", "laughter"), ("laughter", "excitement"),
            ("sadness", "melancholy"), ("sadness", "tears"), ("tears", "depression"),
            ("fear", "terror"), ("fear", "anxiety"), ("anxiety", "despair"),
            ("anger", "rage")
        ]
        
        for e1, e2 in close_pairs:
            if (emotion1 == e1 and emotion2 == e2) or (emotion1 == e2 and emotion2 == e1):
                return 0.3
        
        return 0.6

class EmotionalScenario:
    SCENARIOS = {
        "sleep": {
            "triggers": lambda s: s.get("_sleep_active", False),
            "suppresses": [],
            "intensity": lambda s: s.get("_sleep_transition", 1.0),
            "palette_membrane": PAL_SLEEP_MEM,
            "palette_core": PAL_SLEEP_CORE,
            "priority": 105,
            "visuals": {
                "membrane": {
                    "slow_motion": 0.3,
                    "compression": 0.85,
                    "desaturation": 0.3,
                    "minimal_breath": 0.1
                },
                "core": {
                    "dimming": 0.4,
                    "slow_rotation": 0.2,
                    "compression": 0.9
                }
            }
        },
        "pain": {
            "triggers": lambda s: s["stress"] > 0.7,
            "suppresses": ["fel", "fear", "breath", "rotation"],
            "intensity": lambda s: s["stress"],
            "palette_membrane": PAL_PAIN_MEM,
            "palette_core": PAL_PAIN_CORE,
            "priority": 100,
            "existing_behavior": True
        },
        "despair": {
            "triggers": lambda s: s["energy"] < 0.15 and s["fear"] > 0.5 and all(s.get(k, 0.0) < 0.3 for k in ["joy_input", "sadness_input", "anger_input"]),
            "suppresses": ["fel", "rotation", "breath", "stress"],
            "intensity": lambda s: s["fear"] * (1.0 - s["energy"]),
            "palette_membrane": PAL_DESPAIR_MEM,
            "palette_core": PAL_DESPAIR_CORE,
            "priority": 97,
            "visuals": {
                "membrane": {"collapse": 0.7, "freeze_motion": True, "particle_death": 0.8, "opacity_loss": 0.7},
                "core": {"void_center": True, "rotation_stop": True, "dimming": 0.95}
            }
        },
        "terror": {
            "triggers": lambda s: s.get("fear_input", 0.0) > 0.8 or (s["fear"] > 0.85 and s["stress"] > 0.4),
            "suppresses": ["fel", "rotation", "breath"],
            "intensity": lambda s: max(s.get("fear_input", 0.0), s["fear"]),
            "palette_membrane": PAL_TERROR_MEM,
            "palette_core": PAL_TERROR_CORE,
            "priority": 96,
            "visuals": {
                "membrane": {"freeze_motion": True, "constant_jitter": True, "jitter_amplitude": 0.15, "jitter_frequency": 35.0, "collapse": 0.5},
                "core": {"rotation_stop": True, "extreme_dimming": 0.3}
            }
        },
        "rage": {
            "triggers": lambda s: s.get("anger_input", 0.0) > 0.7 or (s.get("anger_input", 0.0) > 0.5 and s["energy"] > 0.7),
            "suppresses": ["fel", "fear"],
            "intensity": lambda s: s.get("anger_input", 0.0),
            "palette_membrane": PAL_RAGE_MEM,
            "palette_core": PAL_RAGE_CORE,
            "priority": 95,
            "visuals": {
                "membrane": {"constant_jitter": True, "jitter_amplitude": 0.12, "jitter_frequency": 40.0, "particle_nervousness": True, "expansion_boost": 0.3},
                "core": {"erratic_rotation": True, "color_flicker": True, "pulse_frequency": 12.0}
            }
        },
        "euphoria": {
            "triggers": lambda s: s["fel"] > 0.8 and s["energy"] > 0.9 and s.get("joy_input", 0.0) < 0.3,
            "suppresses": ["fear", "stress"],
            "intensity": lambda s: min(s["fel"], s["energy"]),
            "palette_membrane": PAL_EUPHORIA_MEM,
            "palette_core": PAL_EUPHORIA_CORE,
            "priority": 92,
            "visuals": {
                "membrane": {"expansion_boost": 0.5, "particle_burst": True, "rotation_multiplier": 3.0, "saturation": 3.0},
                "core": {"pulse_frequency": 10.0, "brightness_boost": 2.5, "rotation_multiplier": 3.5}
            }
        },
        "excitement": {
            "triggers": lambda s: s.get("joy_input", 0.0) > 0.8 or (s["fel"] > 0.75 and s["energy"] > 0.85 and s["speed"] > 1.2),
            "suppresses": ["fear", "stress"],
            "intensity": lambda s: max(s.get("joy_input", 0.0), s["fel"] * s["energy"]),
            "palette_membrane": PAL_EXCITEMENT_MEM,
            "palette_core": PAL_EXCITEMENT_CORE,
            "priority": 91,
            "visuals": {
                "membrane": {"particle_burst": True, "constant_jitter": True, "jitter_amplitude": 0.08, "jitter_frequency": 30.0, "rotation_multiplier": 2.8},
                "core": {"pulse_frequency": 15.0, "brightness_boost": 2.2, "erratic_rotation": True}
            }
        },
        "tears": {
            "triggers": lambda s: (s["energy"] < 0.25 and s["stress"] > 0.3 and s["stress"] < 0.6 and s.get("sadness_input", 0.0) < 0.3) or s.get("sadness_input", 0.0) > 0.7,
            "suppresses": ["fel"],
            "intensity": lambda s: max(s["stress"] * (1.0 - s["energy"]), s.get("sadness_input", 0.0)),
            "palette_membrane": PAL_TEARS_MEM,
            "palette_core": PAL_TEARS_CORE,
            "priority": 88,
            "visuals": {
                "membrane": {"vertical_streaks": True, "particle_trails": True, "gravity_pull": 0.5, "irregular_tremor": True},
                "core": {"sob_rhythm": 2.5, "compression": 0.85, "dimming": 0.6}
            }
        },
        "fear": {
            "triggers": lambda s: (s["fear"] > 0.6 and s["fear"] < 0.85 and s["stress"] < 0.7 and s.get("fear_input", 0.0) < 0.3) or (s.get("fear_input", 0.0) > 0.3 and s.get("fear_input", 0.0) <= 0.8),
            "suppresses": ["fel", "expansion"],
            "intensity": lambda s: max(s["fear"], s.get("fear_input", 0.0)),
            "palette_membrane": PAL_FEAR_MEM,
            "palette_core": PAL_FEAR_CORE,
            "priority": 85,
            "visuals": {
                "membrane": {"collapse": 0.4, "constant_jitter": True, "jitter_amplitude": 0.05, "jitter_frequency": 20.0},
                "core": {"dimming": 0.7}
            }
        },
        "anger": {
            "triggers": lambda s: s.get("anger_input", 0.0) > 0.2 and s.get("anger_input", 0.0) <= 0.7,
            "suppresses": ["fel"],
            "intensity": lambda s: s.get("anger_input", 0.0),
            "palette_membrane": PAL_ANGER_MEM,
            "palette_core": PAL_ANGER_CORE,
            "priority": 82,
            "visuals": {
                "membrane": {"constant_jitter": True, "jitter_amplitude": 0.07, "jitter_frequency": 28.0, "particle_nervousness": True},
                "core": {"erratic_rotation": True, "color_flicker": True}
            }
        },
        "laughter": {
            "triggers": lambda s: (s["fel"] > 0.7 and s["energy"] > 0.7 and s["stress"] < 0.2 and s.get("joy_input", 0.0) < 0.3) or (s.get("joy_input", 0.0) > 0.5 and s.get("joy_input", 0.0) <= 0.8),
            "suppresses": ["fear", "stress"],
            "intensity": lambda s: max(s["fel"] * s["energy"], s.get("joy_input", 0.0)),
            "palette_membrane": PAL_LAUGHTER_MEM,
            "palette_core": PAL_LAUGHTER_CORE,
            "priority": 78,
            "visuals": {
                "membrane": {"rhythmic_shake": True, "shake_frequency": 7.0, "shake_amplitude": 0.1, "particle_bounce": True},
                "core": {"pulse_rhythm": 7.0, "brightness_variation": 1.8}
            }
        },
        "felicity": {
            "triggers": lambda s: s["fel"] > 0.5 and s["fear"] < 0.3 and s["stress"] < 0.5 and s.get("joy_input", 0.0) < 0.3,
            "suppresses": ["fear"],
            "intensity": lambda s: s["fel"],
            "palette_membrane": PAL_FELICITY_MEM,
            "palette_core": PAL_FELICITY_CORE,
            "priority": 75,
            "existing_behavior": True
        },
        "smile": {
            "triggers": lambda s: (s["fel"] > 0.3 and s["fel"] < 0.8 and s["energy"] > 0.6 and s["stress"] < 0.3 and s.get("joy_input", 0.0) < 0.3) or (s.get("joy_input", 0.0) > 0.2 and s.get("joy_input", 0.0) <= 0.5),
            "suppresses": ["fear"],
            "intensity": lambda s: max(s["fel"] * s["energy"], s.get("joy_input", 0.0)),
            "palette_membrane": PAL_SMILE_MEM,
            "palette_core": PAL_SMILE_CORE,
            "priority": 70,
            "visuals": {
                "membrane": {"asymmetry": 0.15, "lateral_expansion": 0.2, "gentle_wave": True, "warmth": 1.5},
                "core": {"gentle_glow": 1.6, "soft_pulse": 2.0}
            }
        },
        "anxiety": {
            "triggers": lambda s: s["fear"] > 0.4 and s["fear"] < 0.6 and (s["pressure"] > 1.2 or s["noise"] > 0.6 or s["cpu"] > 0.7) and s.get("fear_input", 0.0) < 0.3,
            "suppresses": [],
            "intensity": lambda s: s["fear"] * max(s["pressure"] - 1.0, s["noise"], s["cpu"]),
            "palette_membrane": PAL_ANXIETY_MEM,
            "palette_core": PAL_ANXIETY_CORE,
            "priority": 65,
            "visuals": {
                "membrane": {"constant_jitter": True, "jitter_amplitude": 0.04, "jitter_frequency": 18.0, "particle_nervousness": True, "breath_irregularity": 0.7},
                "core": {"color_flicker": True, "erratic_rotation": True}
            }
        },
        "sadness": {
            "triggers": lambda s: (s["energy"] < 0.3 and s["fear"] < 0.4 and s["stress"] < 0.3 and s["fel"] < 0.2 and s.get("sadness_input", 0.0) < 0.3) or (s.get("sadness_input", 0.0) > 0.2 and s.get("sadness_input", 0.0) <= 0.7),
            "suppresses": ["fel"],
            "intensity": lambda s: max(1.0 - s["energy"], s.get("sadness_input", 0.0)),
            "palette_membrane": PAL_SADNESS_MEM,
            "palette_core": PAL_SADNESS_CORE,
            "priority": 60,
            "visuals": {
                "membrane": {"gravity_pull": 0.4, "slow_motion": 0.6, "desaturation": 0.4, "compression": 0.8},
                "core": {"dimming": 0.5, "slow_rotation": 0.5}
            }
        },
        "melancholy": {
            "triggers": lambda s: s["energy"] < 0.5 and s["fear"] < 0.3 and s["hum"] > 0.6 and s["fel"] < 0.3 and s.get("sadness_input", 0.0) < 0.3,
            "suppresses": [],
            "intensity": lambda s: (1.0 - s["energy"]) * s["hum"],
            "palette_membrane": PAL_MELANCHOLY_MEM,
            "palette_core": PAL_MELANCHOLY_CORE,
            "priority": 55,
            "visuals": {
                "membrane": {"slow_waves": True, "wave_period": 10.0, "mist_effect": True, "softness": 2.0},
                "core": {"gentle_drift": True, "soft_glow": 0.7}
            }
        },
        "depression": {
            "triggers": lambda s: s["energy"] < 0.2 and s["fear"] < 0.4 and s["stress"] < 0.2 and s["fel"] < 0.1 and s["light"] < 0.3 and s.get("sadness_input", 0.0) < 0.3,
            "suppresses": ["fel", "breath"],
            "intensity": lambda s: (1.0 - s["energy"]) * (1.0 - s["light"]),
            "palette_membrane": PAL_DEPRESSION_MEM,
            "palette_core": PAL_DEPRESSION_CORE,
            "priority": 50,
            "visuals": {
                "membrane": {"near_stillness": True, "opacity_loss": 0.6, "minimal_breath": 0.15, "heaviness": 0.95},
                "core": {"rotation_factor": 0.08, "gray_wash": 0.85, "extreme_dimming": 0.85}
            }
        }
    }

class EmotionalArbiter:
    @staticmethod
    def compute_fear(targets):
        tech_stress = (targets["cpu"] * 0.4) + (targets["ram"] * 0.3)
        existential_void = (1.0 - targets["energy"]) * 0.6
        env_stress = (max(0, targets["pressure"] - 1.1) * 0.4) + \
                     (max(0, targets["hum"] - 0.7) * 0.3) + \
                     (targets["noise"] * 0.2)
        
        raw_fear = np.clip(tech_stress + existential_void + env_stress, 0.0, 1.0)
        calculated_fear = np.clip(raw_fear - (targets["fel"] * 0.8), 0.0, 1.0)
        
        if targets.get("fear_input", 0.0) > 0.1:
            return max(calculated_fear, targets["fear_input"])
        
        return calculated_fear
    
    @staticmethod
    def detect_internal_emotion(state):
        sorted_emotions = sorted(
            EmotionalScenario.SCENARIOS.items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )
        
        for emotion_name, scenario in sorted_emotions:
            if scenario["triggers"](state):
                intensity = scenario["intensity"](state)
                if intensity > 0.05:
                    return emotion_name, intensity, scenario
        
        return "neutral", 0.0, None
    
    @staticmethod
    def should_suppress(dominant_emotion, param):
        if dominant_emotion == "neutral":
            return False
        scenario = EmotionalScenario.SCENARIOS.get(dominant_emotion)
        if scenario:
            return param in scenario.get("suppresses", [])
        return False

class LifeCore:
    def __init__(self):
        self.res = 26
        self.rot = [random.uniform(0, 360) for _ in range(3)]
        self.drift = [random.uniform(0.8, 1.2) for _ in range(3)]
        
        self.rem_offset_x = 0.0
        self.rem_offset_y = 0.0
        self.rem_next_jump_time = 0.0

    def draw(self, t, scale, palette, state, emotion_scenario, emotion_intensity, transition, 
             sleep_factor, dream_active, dream_intensity, nightmare_active, nightmare_intensity):
        glPushMatrix()
        
        speed = state["speed"]
        temp = state["temp"]
        stress = state["stress"]
        en = state["energy"]
        fear = state["fear"]
        fel = state["fel"]
        light = state["light"]
        noise = state["noise"]
        cpu = state["cpu"]
        ram = state["ram"]
        
        visuals = {}
        if emotion_scenario and not emotion_scenario.get("existing_behavior"):
            visuals = emotion_scenario.get("visuals", {}).get("core", {})
        
        eff_speed = speed * 1.4 * 0.7 * (0.4 + temp * 0.6) * en
        
        if sleep_factor > 0:
            eff_speed *= (1.0 - sleep_factor * 0.8)
        
        if visuals.get("rotation_stop"):
            eff_speed *= (1.0 - emotion_intensity * transition)
        elif visuals.get("rotation_multiplier"):
            eff_speed *= lerp(1.0, visuals["rotation_multiplier"], emotion_intensity * transition)
        elif visuals.get("rotation_factor"):
            eff_speed *= lerp(1.0, visuals["rotation_factor"], emotion_intensity * transition)
        elif visuals.get("slow_rotation"):
            eff_speed *= lerp(1.0, visuals["slow_rotation"], emotion_intensity * transition)
        elif visuals.get("erratic_rotation"):
            eff_speed *= (1.0 + math.sin(t * 15.0) * 0.5 * emotion_intensity * transition)
        
        eff_speed *= (1.0 - fear * 0.5 + fel * 0.2)
        eff_speed *= (1.0 + noise * 0.3)
        
        if stress < 0.8: 
            for i in range(3): 
                self.rot[i] += (self.drift[i] * eff_speed)
        
        pain_jitter = (random.random() - 0.5) * (stress * 20.0)
        cpu_jitter = (random.random() - 0.5) * (cpu * 8.0)
        noise_jitter = (random.random() - 0.5) * (noise * 5.0)
        
        glRotatef(self.rot[0] + pain_jitter + cpu_jitter + noise_jitter, 1, 0, 0)
        glRotatef(self.rot[1], 0, 1, 0)
        glRotatef(self.rot[2], 0, 0, 1)
        
        # RÊVE (REM)
        if dream_active and sleep_factor > 0.8:
            if t >= self.rem_next_jump_time:
                self.rem_offset_x = random.uniform(-0.15, 0.15) * dream_intensity
                self.rem_offset_y = random.uniform(-0.08, 0.08) * dream_intensity
                self.rem_next_jump_time = t + random.uniform(0.1, 0.4)
            
            glTranslatef(self.rem_offset_x, self.rem_offset_y, 0)
        
        # CAUCHEMAR (secousses erratiques)
        elif nightmare_active and sleep_factor > 0.8:
            if random.random() < 0.15 * nightmare_intensity:
                self.rem_offset_x = random.uniform(-0.25, 0.25) * nightmare_intensity
                self.rem_offset_y = random.uniform(-0.25, 0.25) * nightmare_intensity
            
            self.rem_offset_x *= 0.85
            self.rem_offset_y *= 0.85
            
            glTranslatef(self.rem_offset_x, self.rem_offset_y, 0)
        else:
            self.rem_offset_x = 0.0
            self.rem_offset_y = 0.0
        
        final_scale = scale * (1.0 - (stress * STRESS_RETRACTION_PCT) - (fear * 0.2) + (fel * 0.25))
        
        if sleep_factor > 0:
            final_scale *= (1.0 - sleep_factor * 0.15)
        
        if nightmare_active and sleep_factor > 0.8:
            final_scale *= (1.0 - nightmare_intensity * 0.2)
        
        if visuals.get("compression"):
            final_scale *= lerp(1.0, visuals["compression"], emotion_intensity * transition)
        
        pulse_add = 0.0
        if visuals.get("pulse_frequency"):
            pulse_add = math.sin(t * visuals["pulse_frequency"]) * 0.12 * emotion_intensity * transition
        elif visuals.get("soft_pulse"):
            pulse_add = math.sin(t * visuals["soft_pulse"]) * 0.05 * emotion_intensity * transition
        elif visuals.get("pulse_rhythm"):
            pulse_add = abs(math.sin(t * visuals["pulse_rhythm"])) * 0.1 * emotion_intensity * transition
        elif visuals.get("sob_rhythm"):
            sob = abs(math.sin(t * visuals["sob_rhythm"])) ** 4
            pulse_add = sob * 0.15 * emotion_intensity * transition
        
        final_scale += pulse_add
        
        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)
        
        for i in range(self.res):
            lat0, lat1 = np.pi * (-0.5 + i/self.res), np.pi * (-0.5 + (i+1)/self.res)
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(self.res + 1):
                lng = 2 * np.pi * j / self.res
                for lat in [lat0, lat1]:
                    x, y, z = np.cos(lng)*np.cos(lat), np.sin(lng)*np.cos(lat), np.sin(lat)
                    r = (0.55 * final_scale) + np.sin(x*2.5 + t)*0.08 + np.cos(y*2.0 + t*0.8)*0.06
                    
                    color_mods = {}
                    
                    base_brightness = max(0.3, 0.8 * light)
                    
                    if sleep_factor > 0:
                        base_brightness = max(0.3, base_brightness * (1.0 - sleep_factor * 0.4))
                    
                    if visuals.get("brightness_boost"):
                        base_brightness *= lerp(1.0, visuals["brightness_boost"], emotion_intensity * transition)
                    elif visuals.get("gentle_glow"):
                        base_brightness *= lerp(1.0, visuals["gentle_glow"], emotion_intensity * transition)
                    elif visuals.get("soft_glow"):
                        base_brightness *= lerp(1.0, visuals["soft_glow"], emotion_intensity * transition)
                    color_mods['brightness'] = base_brightness
                    
                    if visuals.get("dimming"):
                        color_mods['dimming'] = visuals["dimming"] * emotion_intensity * transition
                    elif visuals.get("extreme_dimming"):
                        color_mods['dimming'] = visuals["extreme_dimming"] * emotion_intensity * transition
                    
                    if visuals.get("gray_wash"):
                        color_mods['desaturation'] = visuals["gray_wash"] * emotion_intensity * transition
                    
                    if sleep_factor > 0:
                        existing_desat = color_mods.get('desaturation', 0.0)
                        color_mods['desaturation'] = min(0.5, existing_desat + sleep_factor * 0.3)
                    
                    if visuals.get("color_flicker"):
                        flicker = math.sin(t * 25.0 + x * 12) * 0.4 * emotion_intensity * transition
                        color_mods['brightness'] *= (1.0 + flicker)
                    
                    if visuals.get("brightness_variation"):
                        var = abs(math.sin(t * 7.0)) * visuals["brightness_variation"]
                        color_mods['brightness'] *= lerp(1.0, var, emotion_intensity * transition)
                    
                    if dream_active and sleep_factor > 0.8:
                        dream_pulse = abs(math.sin(t * 12.0 + x * 8 + y * 6)) * 0.3 * dream_intensity
                        color_mods['brightness'] *= (1.0 + dream_pulse)
                    
                    c = get_gradient_color(t, x+y, 1.0, palette, color_mods)
                    r_c, g_c, b_c = [lerp(c[idx], 6.0, stress**2) for idx in range(3)]
                    
                    if visuals.get("void_center"):
                        void_factor = (1.0 - abs(z)) * emotion_intensity * transition
                        r_c, g_c, b_c = [col * (1.0 - void_factor * 0.95) for col in [r_c, g_c, b_c]]
                    
                    glColor4f(r_c, g_c, b_c, 0.25 + stress*0.4)
                    glVertex3f(x*r, y*r, z*r)
            glEnd()
        
        glDepthMask(GL_TRUE)
        glPopMatrix()

class GellyMaskSystem:
    def __init__(self):
        phi = math.pi * (3. - math.sqrt(5.))
        self.pts = []
        for i in range(PARTICLES_DENSITY):
            y = 1 - (i / float(PARTICLES_DENSITY - 1)) * 2
            r_val = math.sqrt(1 - y * y)
            theta = phi * i
            self.pts.append({
                'pos': np.array([math.cos(theta)*r_val, y, math.sin(theta)*r_val]), 
                'seed': random.random()
            })
        
        self.core = LifeCore()
        self.arbiter = EmotionalArbiter()
        self.mask_system = EmotionalMask()
        self.background = ConfigurableBackground(BACKGROUND_COLORS)
        self.galactic_aura = GalacticAura()
        self.nightmare_aura = NightmareAura()
        
        self.default_state = {
            "pressure": 1.0, "temp": 0.5, "hum": 0.5, "stress": 0.0, 
            "speed": 1.0, "morph_m": 0.0, "energy": 1.0, "fear": 0.0, "fel": 0.0,
            "light": 1.0, "noise": 0.0, "cpu": 0.0, "ram": 0.0,
            "joy_input": 0.0, "sadness_input": 0.0, "anger_input": 0.0, "fear_input": 0.0
        }
        self.current = self.default_state.copy()
        self.targets = self.default_state.copy()
        self.prev_energy = 1.0
        self.is_charging = False
        self.m_rot = [random.uniform(0, 360) for _ in range(3)]
        
        self.internal_emotion = "neutral"
        self.displayed_emotion = "neutral"
        self.internal_intensity = 0.0
        self.displayed_intensity = 0.0
        self.is_masking = False
        self.masking_effort = 0.0
        self.emotion_transition = 0.0
        self.emotion_scenario = None
        
        self.sleep_active = False
        self.sleep_transition = 0.0
        self.sleep_transition_speed = 0.0
        
        self.dream_active = False
        self.dream_intensity = 0.5
        
        self.nightmare_active = False
        self.nightmare_intensity = 0.5

    def toggle_sleep(self, t):
        self.sleep_active = not self.sleep_active
        
        if self.sleep_active:
            self.sleep_transition_speed = 1.0 / (SLEEP_TRANSITION_DURATION * FPS)
            print(f"[{t:.2f}] 😴 ENDORMISSEMENT...")
        else:
            self.sleep_transition_speed = -1.0 / (WAKE_TRANSITION_DURATION * FPS)
            print(f"[{t:.2f}] 🌅 RÉVEIL...")
            
            if self.dream_active:
                self.dream_active = False
                print(f"[{t:.2f}] 💭 Rêve interrompu (réveil)")
            
            if self.nightmare_active:
                self.nightmare_active = False
                print(f"[{t:.2f}] 😱 Cauchemar interrompu (réveil)")
    
    def toggle_dream(self, t):
        if not self.sleep_active:
            print(f"[{t:.2f}] ⚠️  Impossible de rêver : l'entité n'est pas endormie")
            return
        
        if self.sleep_transition < 0.8:
            print(f"[{t:.2f}] ⚠️  Sommeil pas assez profond pour rêver (transition: {self.sleep_transition:.2f})")
            return
        
        self.dream_active = not self.dream_active
        
        if self.dream_active:
            self.nightmare_active = False
            print(f"[{t:.2f}] 💭 MODE RÊVE (REM) activé - Intensité: {self.dream_intensity:.2f}")
        else:
            print(f"[{t:.2f}] 💤 Mode rêve désactivé - Sommeil calme")
    
    def toggle_nightmare(self, t):
        if not self.sleep_active:
            print(f"[{t:.2f}] ⚠️  Impossible de faire un cauchemar : l'entité n'est pas endormie")
            return
        
        if self.sleep_transition < 0.8:
            print(f"[{t:.2f}] ⚠️  Sommeil pas assez profond pour un cauchemar (transition: {self.sleep_transition:.2f})")
            return
        
        self.nightmare_active = not self.nightmare_active
        
        if self.nightmare_active:
            self.dream_active = False
            print(f"[{t:.2f}] 😱 CAUCHEMAR activé - Intensité: {self.nightmare_intensity:.2f}")
            print(f"[{t:.2f}] ⚠️  COMMUNICATION ANXIOGÈNE DÉTECTÉE")
        else:
            print(f"[{t:.2f}] 🌙 Cauchemar dissipé - Sommeil calme")
    
    def adjust_dream_intensity(self, delta):
        self.dream_intensity = np.clip(self.dream_intensity + delta, 0.1, 1.0)
        print(f"Intensité du rêve: {self.dream_intensity:.2f}")
    
    def adjust_nightmare_intensity(self, delta):
        self.nightmare_intensity = np.clip(self.nightmare_intensity + delta, 0.1, 1.0)
        print(f"Intensité du cauchemar: {self.nightmare_intensity:.2f}")

    def update_logic(self, t):
        if self.sleep_active and self.sleep_transition < 1.0:
            self.sleep_transition = min(1.0, self.sleep_transition + self.sleep_transition_speed)
            if self.sleep_transition >= 1.0:
                print(f"[{t:.2f}] 😴 ENDORMI")
        elif not self.sleep_active and self.sleep_transition > 0.0:
            self.sleep_transition = max(0.0, self.sleep_transition + self.sleep_transition_speed)
            if self.sleep_transition <= 0.0:
                print(f"[{t:.2f}] 🌅 ÉVEILLÉ")
        
        if self.sleep_transition > 0:
            self.current["_sleep_active"] = True
            self.current["_sleep_transition"] = self.sleep_transition
        else:
            self.current["_sleep_active"] = False
            self.current["_sleep_transition"] = 0.0
        
        if self.targets["fel"] > 0: 
            self.targets["fel"] = max(0.0, self.targets["fel"] - FEL_DECAY_STEP)
        
        self.targets["fear"] = self.arbiter.compute_fear(self.targets)
        
        self.is_charging = (self.targets["energy"] > self.prev_energy)
        self.prev_energy = self.targets["energy"]

        for key in ["pressure", "temp", "hum", "speed", "energy", "fel", "fear", "light", "noise", "cpu", "ram",
                    "joy_input", "sadness_input", "anger_input", "fear_input"]:
            self.current[key] = lerp(self.current[key], self.targets[key], 1.0/TRANSITION_SENSORS)
        
        self.current["stress"] = lerp(self.current["stress"], self.targets["stress"], 1.0/TRANSITION_STRESS)
        if self.targets["stress"] > 0: 
            self.targets["stress"] = max(0, self.targets["stress"] - 0.35)
        
        self.current["morph_m"] = lerp(self.current["morph_m"], self.targets["morph_m"], 0.05)
        
        internal_emo, internal_int, internal_scenario = self.arbiter.detect_internal_emotion(self.current)
        
        emotion_inputs = {
            "joy": self.current["joy_input"],
            "sadness": self.current["sadness_input"],
            "anger": self.current["anger_input"],
            "fear": self.current["fear_input"]
        }
        
        displayed_emo, displayed_int, is_masking, effort = self.mask_system.compute_displayed_emotion(
            internal_emo, internal_int, emotion_inputs
        )
        
        prev_displayed = self.displayed_emotion
        self.internal_emotion = internal_emo
        self.internal_intensity = internal_int
        self.displayed_emotion = displayed_emo
        self.displayed_intensity = displayed_int
        self.is_masking = is_masking
        self.masking_effort = effort
        
        if displayed_emo != prev_displayed:
            self.emotion_transition = 0.0
            if is_masking:
                print(f"[{t:.2f}] >> 🎭 MASKING: {internal_emo} → SHOWS {displayed_emo} | Effort: {effort:.2f}")
            else:
                print(f"[{t:.2f}] >> ✨ AUTHENTIC: {displayed_emo} ({displayed_int:.2f})")
        else:
            self.emotion_transition = min(1.0, self.emotion_transition + 0.05)
        
        displayed_scenario = EmotionalScenario.SCENARIOS.get(displayed_emo, None)
        
        if displayed_scenario:
            target_mem = displayed_scenario.get("palette_membrane", MEMBRANE_A)
            self.emotion_scenario = displayed_scenario
        else:
            target_mem = MEMBRANE_A
        
        if internal_scenario:
            target_core = internal_scenario.get("palette_core", CORE_A)
        else:
            target_core = CORE_A
        
        base_mem = lerp_palette(MEMBRANE_A, MEMBRANE_B, self.current["morph_m"])
        base_core = lerp_palette(CORE_A, CORE_B, self.current["morph_m"])
        
        # FIX TEMPÉRATURE : 0.0=Glacier, 1.0=Lava
        f_temp = (self.current["temp"] - 0.5) * 2.0
        
        if abs(f_temp) > 0.3:
            if f_temp > 0:  # CHAUD → LAVA
                target_mem = lerp_palette(target_mem, MEMBRANE_LAVA, min(1.0, abs(f_temp) * 1.5))
                target_core = lerp_palette(target_core, CORE_LAVA, min(1.0, abs(f_temp) * 1.5))
            else:  # FROID → GLACIER
                target_mem = lerp_palette(target_mem, MEMBRANE_GLACIER, min(1.0, abs(f_temp) * 1.5))
                target_core = lerp_palette(target_core, CORE_GLACIER, min(1.0, abs(f_temp) * 1.5))
        
        blend_mem = self.displayed_intensity * self.emotion_transition
        blend_core = self.internal_intensity * self.emotion_transition
        
        self.current_mem_pal = lerp_palette(base_mem, target_mem, blend_mem)
        self.current_core_pal = lerp_palette(base_core, target_core, blend_core)

    def draw(self, t):
        self.update_logic(t)
        en, stress, fear, fel = self.current["energy"], self.current["stress"], self.current["fear"], self.current["fel"]
        light, noise, cpu, ram = self.current["light"], self.current["noise"], self.current["cpu"], self.current["ram"]
        
        mem_visuals = {}
        if self.emotion_scenario and not self.emotion_scenario.get("existing_behavior"):
            mem_visuals = self.emotion_scenario.get("visuals", {}).get("membrane", {})
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.background.draw(light, self.sleep_transition)
        
        # AURA : Galactique (rêve) ou Cauchemar
        if self.sleep_transition > 0.3:
            if self.nightmare_active:
                self.nightmare_aura.draw(t, self.sleep_transition, self.nightmare_active, self.nightmare_intensity)
            else:
                self.galactic_aura.draw(t, self.sleep_transition, self.dream_active, self.dream_intensity)
        
        breath_frozen = (stress > 0.8) or \
                        (self.arbiter.should_suppress(self.displayed_emotion, "breath"))
        
        if breath_frozen:
            breath_factor = 1.0
        else:
            base_breath = abs(math.sin(t * math.pi / EXPANSION_PERIOD))
            
            if self.sleep_transition > 0:
                sleep_breath_period = EXPANSION_PERIOD * (1.0 + self.sleep_transition * 3.0)
                base_breath = abs(math.sin(t * math.pi / sleep_breath_period)) * (1.0 - self.sleep_transition * 0.7)
            
            if mem_visuals.get("minimal_breath"):
                breath_factor = lerp(base_breath, mem_visuals["minimal_breath"], 
                                    self.displayed_intensity * self.emotion_transition)
            elif mem_visuals.get("breath_irregularity"):
                irregular = abs(math.sin(t * 4.0)) * mem_visuals["breath_irregularity"]
                breath_factor = base_breath * (1.0 + irregular * self.displayed_intensity * self.emotion_transition)
            else:
                breath_factor = base_breath
        
        m_scale = (SCALE_MIN + (SCALE_MAX - SCALE_MIN) * (breath_factor * (0.5 + 0.5 * en)))
        m_scale *= (1.0 - (stress * STRESS_RETRACTION_PCT)) 
        m_scale *= (1.0 - fear * 0.15 + fel * 0.22)
        m_scale *= lerp(1.5, 0.7, self.current["pressure"] / 1.8)
        
        if mem_visuals.get("expansion_boost"):
            m_scale *= lerp(1.0, 1.0 + mem_visuals["expansion_boost"], 
                           self.displayed_intensity * self.emotion_transition)
        elif mem_visuals.get("collapse"):
            m_scale *= lerp(1.0, 1.0 - mem_visuals["collapse"], 
                           self.displayed_intensity * self.emotion_transition)
        
        if mem_visuals.get("compression"):
            m_scale *= lerp(1.0, mem_visuals["compression"], 
                           self.displayed_intensity * self.emotion_transition)
        elif mem_visuals.get("heaviness"):
            m_scale *= lerp(1.0, 1.0 - mem_visuals["heaviness"] * 0.12, 
                           self.displayed_intensity * self.emotion_transition)
        
        glPushMatrix()
        
        jitter_glob = np.random.normal(0, 0.04 * cpu + 0.03 * fear + 0.025 * noise, 3)
        glTranslatef(
            math.sin(t*0.4)*0.12 + jitter_glob[0],
            math.cos(t*0.3)*0.12 + jitter_glob[1],
            jitter_glob[2]
        )
        
        if self.is_masking and self.masking_effort > 0.3:
            effort_tremor = self.masking_effort * 0.05
            glTranslatef(
                (random.random() - 0.5) * effort_tremor,
                (random.random() - 0.5) * effort_tremor,
                (random.random() - 0.5) * effort_tremor
            )
        
        if stress > 0.5:
            v = stress * 0.08
            glTranslatef(random.uniform(-v,v), random.uniform(-v,v), random.uniform(-v,v))
        
        if mem_visuals.get("rhythmic_shake"):
            shake_amp = mem_visuals.get("shake_amplitude", 0.1)
            shake_freq = mem_visuals.get("shake_frequency", 7.0)
            shake = math.sin(t * shake_freq) * shake_amp * self.displayed_intensity * self.emotion_transition
            glTranslatef(shake, shake * 0.5, 0)
        
        self.core.draw(t, 1.0, self.current_core_pal, self.current, 
                      self.emotion_scenario, self.internal_intensity, self.emotion_transition,
                      self.sleep_transition, self.dream_active, self.dream_intensity,
                      self.nightmare_active, self.nightmare_intensity)
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glPushMatrix()
        
        rotation_frozen = (stress > 0.8) or \
                         (self.arbiter.should_suppress(self.displayed_emotion, "rotation"))
        
        m_speed = self.current["speed"] * 1.5 * en * (1.0 - fear * 0.35 + fel * 0.25)
        m_speed *= (1.0 + noise * 0.5)
        
        if self.sleep_transition > 0:
            m_speed *= (1.0 - self.sleep_transition * 0.9)
        
        if mem_visuals.get("freeze_motion") or mem_visuals.get("near_stillness"):
            m_speed *= (1.0 - self.displayed_intensity * self.emotion_transition * 0.98)
        elif mem_visuals.get("rotation_multiplier"):
            m_speed *= lerp(1.0, mem_visuals["rotation_multiplier"], 
                           self.displayed_intensity * self.emotion_transition)
        elif mem_visuals.get("slow_motion"):
            m_speed *= lerp(1.0, mem_visuals["slow_motion"], 
                           self.displayed_intensity * self.emotion_transition)
        
        if not rotation_frozen:
            for i in range(3): 
                self.m_rot[i] += m_speed
        
        glRotatef(self.m_rot[0], 0, 1, 1)
        glRotatef(self.m_rot[1], 1, 0, 0)
        glRotatef(self.m_rot[2], 0, 0, 1)
        
        scan_y = math.sin(t * 3.5)
        burst_active = 1.0 if en > 0.99 else 0.0
        
        base_softness = lerp(0.01, 0.25, self.current["hum"])
        if mem_visuals.get("softness"):
            base_softness *= lerp(1.0, mem_visuals["softness"], 
                                 self.displayed_intensity * self.emotion_transition)
        softness = base_softness
        
        sparkle_freq = lerp(5.0, 50.0, self.current["hum"]) * en
        
        if self.sleep_transition > 0:
            sparkle_freq *= (1.0 - self.sleep_transition * 0.8)

        for size, step, boost, alpha_base in [(20.0, 4, 0.8, 0.07), (2.5, 1, 1.3, 0.35)]:
            effective_size = size * (1.0 + ram * 1.8 + stress * 0.8 + fel * 0.6)
            
            if mem_visuals.get("particle_death"):
                death_factor = mem_visuals["particle_death"] * self.displayed_intensity * self.emotion_transition
                if step == 4:
                    effective_size *= (1.0 - death_factor)
            
            if mem_visuals.get("mist_effect") and step == 4:
                effective_size *= lerp(1.0, 2.5, self.displayed_intensity * self.emotion_transition)
            
            effective_size *= (1.0 + noise * 0.4)
            
            glPointSize(effective_size)
            glBegin(GL_POINTS)
            
            for p in self.pts[::step]:
                pos = p['pos'].copy()
                pos[1] *= 0.82
                
                shiver_wave = math.pow(max(0, math.sin(((pos[1] + 1.0)/2.0) * 10 - t * 8)), 4.0)
                shiver_off = (p['seed'] - 0.5) * shiver_wave * (fel * 0.7)
                
                charge_signal = 1.0 if (abs(pos[1] - scan_y) < 0.07 and self.is_charging) else 0.0
                burst_off = pos * (math.sin(t * 15 + p['seed'] * 10) * 0.3) * burst_active
                
                pain_chaos = (p['seed'] - 0.5) * stress * 6.0
                fear_chaos = (p['seed']-0.5)*fear*0.6
                
                noise_chaos = np.array([
                    (random.random() - 0.5) * noise * 0.5,
                    (random.random() - 0.5) * noise * 0.5,
                    (random.random() - 0.5) * noise * 0.5
                ])
                
                dream_chaos = np.array([0.0, 0.0, 0.0])
                if self.dream_active and self.sleep_transition > 0.8:
                    if p['seed'] < 0.4:
                        dream_chaos = np.array([
                            math.sin(t * 18.0 + p['seed'] * 25) * 0.08 * self.dream_intensity,
                            math.cos(t * 22.0 + p['seed'] * 30) * 0.05 * self.dream_intensity,
                            0.0
                        ])
                
                nightmare_chaos = np.array([0.0, 0.0, 0.0])
                if self.nightmare_active and self.sleep_transition > 0.8:
                    nightmare_chaos = np.array([
                        (random.random() - 0.5) * 0.25 * self.nightmare_intensity,
                        (random.random() - 0.5) * 0.25 * self.nightmare_intensity,
                        (random.random() - 0.5) * 0.15 * self.nightmare_intensity
                    ])
                
                gravity_off = np.array([0.0, 0.0, 0.0])
                if mem_visuals.get("gravity_pull"):
                    pull = mem_visuals["gravity_pull"] * self.displayed_intensity * self.emotion_transition
                    gravity_off[1] = -abs(pos[1]) * pull
                
                streak_off = np.array([0.0, 0.0, 0.0])
                if mem_visuals.get("vertical_streaks"):
                    if pos[1] < 0:
                        streak_phase = (t * 0.6 + p['seed'] * 5.0) % 2.0
                        if streak_phase < 1.0:
                            streak_off[1] = -streak_phase * 0.4 * self.displayed_intensity * self.emotion_transition
                
                jitter_off = np.array([0.0, 0.0, 0.0])
                if mem_visuals.get("constant_jitter"):
                    jitter_amp = mem_visuals.get("jitter_amplitude", 0.04)
                    jitter_freq = mem_visuals.get("jitter_frequency", 18.0)
                    jitter_off = np.array([
                        math.sin(t * jitter_freq + p['seed'] * 10) * jitter_amp,
                        math.cos(t * jitter_freq * 1.4 + p['seed'] * 7) * jitter_amp,
                        math.sin(t * jitter_freq * 0.9 + p['seed'] * 13) * jitter_amp
                    ]) * self.displayed_intensity * self.emotion_transition
                
                f_pos = pos * (m_scale * (1.18 if step > 1 else 1.0) + softness * math.sin(pos[1]*9+t*4.5)) + \
                        shiver_off + burst_off + pain_chaos + fear_chaos + noise_chaos + dream_chaos + \
                        nightmare_chaos + gravity_off + streak_off + jitter_off
                
                color_mods = {}
                
                if mem_visuals.get("warmth"):
                    color_mods['brightness'] = mem_visuals["warmth"] * light
                else:
                    color_mods['brightness'] = light
                
                if self.sleep_transition > 0:
                    color_mods['brightness'] = max(0.3, color_mods['brightness'] * (1.0 - self.sleep_transition * 0.5))
                
                if mem_visuals.get("desaturation"):
                    color_mods['desaturation'] = mem_visuals["desaturation"] * self.displayed_intensity * self.emotion_transition
                
                if self.sleep_transition > 0:
                    existing_desat = color_mods.get('desaturation', 0.0)
                    color_mods['desaturation'] = min(0.5, existing_desat + self.sleep_transition * 0.3)
                
                c = get_gradient_color(t, p['seed'], boost, self.current_mem_pal, color_mods)
                r, g, b = [lerp(c[idx], 6.0, stress**2) for idx in range(3)]
                
                r = lerp(r, PAL_FELICITY_MEM[0][0], max(charge_signal, burst_active * 0.6))
                g = lerp(g, PAL_FELICITY_MEM[0][1], max(charge_signal, burst_active * 0.6))
                b = lerp(b, PAL_FELICITY_MEM[0][2], max(charge_signal, burst_active * 0.6))
                
                # Teinte rouge sang pour cauchemar
                if self.nightmare_active and self.sleep_transition > 0.8:
                    blood_tint = self.nightmare_intensity * 0.6
                    r = lerp(r, 1.2, blood_tint)
                    g = lerp(g, 0.0, blood_tint)
                    b = lerp(b, 0.0, blood_tint)
                
                if mem_visuals.get("particle_burst"):
                    burst_factor = abs(math.sin(t * 9 + p['seed'] * 13)) * self.displayed_intensity * self.emotion_transition
                    r, g, b = [col * (1.0 + burst_factor) for col in [r, g, b]]
                
                if mem_visuals.get("saturation"):
                    sat = mem_visuals["saturation"] * self.displayed_intensity * self.emotion_transition
                    r, g, b = [col * sat for col in [r, g, b]]
                
                sparkle = 0.0
                if step == 1 and sparkle_freq > 0:
                    sparkle = math.pow(abs(math.sin(t * sparkle_freq + p['seed'] * 32.0)), 20.0)
                
                base_alpha = (alpha_base * (1.0 + ram + stress + fel)) + charge_signal * 0.7 + burst_active * 0.4 + sparkle * 0.8
                
                if self.sleep_transition > 0:
                    base_alpha = max(0.15, base_alpha * (1.0 - self.sleep_transition * 0.4))
                
                if mem_visuals.get("opacity_loss"):
                    loss = mem_visuals["opacity_loss"] * self.displayed_intensity * self.emotion_transition
                    base_alpha *= (1.0 - loss)
                
                if mem_visuals.get("particle_death"):
                    death = mem_visuals["particle_death"] * self.displayed_intensity * self.emotion_transition
                    if p['seed'] < death:
                        base_alpha *= (1.0 - death)
                
                if mem_visuals.get("mist_effect") and step == 4:
                    base_alpha *= 0.35
                
                glColor4f(r, g, b, base_alpha)
                glVertex3f(*f_pos)
            
            glEnd()
        
        glPopMatrix()
        glPopMatrix()

def print_glossary():
    print("""
    ════════════════════════════════════════════════════════════════════
    SYNTHETIC AWARENESS SYSTEM - Sleep/Dream/Nightmare
    ════════════════════════════════════════════════════════════════════
    
    FOND PARAMÉTRABLE (en haut du code):
    BACKGROUND_COLORS = [(r, g, b), ...]  # 1=uni, 2-3=dégradé
    
    CONTRÔLES PHYSIOLOGIQUES:
    [P] Pressure   | [T] Temp (0.0=🧊 0.5=Neutre 1.0=🌋)
    [H] Humidity   | [V] Velocity | [E] Energy | [F] Felicity
    [L] Light      | [W] Noise    | [C] CPU    | [R] RAM
    
    CONTRÔLES ÉMOTIONNELS (Masque Social):
    [J] Joy Input      | [U] Sadness Input
    [G] Anger Input    | [I] Fear Input
    
    ÉTATS DE CONSCIENCE:
    [Z] Toggle Sommeil/Éveil
    [O] Toggle Rêve (REM) - si endormi profond
    [N] Toggle Cauchemar - si endormi profond
    [↑][↓] Ajuste intensité rêve/cauchemar actif
    
    RÊVE vs CAUCHEMAR (exclusifs):
    💭 RÊVE      : Aura violette galactique + mouvements doux
    😱 CAUCHEMAR : Ombres rouges rampantes + secousses + teinte sang
    
    DÉCLENCHEURS:
    [S] Stress Spike | [M] Morph Palette | [D] Default/Reset
    
    DEBUG:
    [B] Affiche état émotionnel complet
    
    ════════════════════════════════════════════════════════════════════
    """)

def main():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("SAS - LIFE 2025")
    glEnable(GL_POINT_SMOOTH)
    gluPerspective(45, WIDTH/HEIGHT, 0.1, 50.0)
    glTranslatef(0.0, 0.0, -4.5)
    
    sas = GellyMaskSystem()
    print_glossary()
    
    clock = pygame.time.Clock()
    start_time = time.time()
    active_mode = "energy"
    
    while True:
        t = time.time() - start_time
        
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            
            if event.type == KEYDOWN:
                # SÉLECTION MODE
                if event.key == K_p:
                    active_mode = "pressure"
                    print(f"[{t:.2f}] >> FOCUS: PRESSURE")
                elif event.key == K_t:
                    active_mode = "temp"
                    print(f"[{t:.2f}] >> FOCUS: TEMP")
                elif event.key == K_h:
                    active_mode = "hum"
                    print(f"[{t:.2f}] >> FOCUS: HUMIDITY")
                elif event.key == K_v:
                    active_mode = "speed"
                    print(f"[{t:.2f}] >> FOCUS: SPEED")
                elif event.key == K_e:
                    active_mode = "energy"
                    print(f"[{t:.2f}] >> FOCUS: ENERGY")
                elif event.key == K_f:
                    active_mode = "fel"
                    print(f"[{t:.2f}] >> FOCUS: FELICITY")
                elif event.key == K_l:
                    active_mode = "light"
                    print(f"[{t:.2f}] >> FOCUS: LIGHT")
                elif event.key == K_w:
                    active_mode = "noise"
                    print(f"[{t:.2f}] >> FOCUS: NOISE")
                elif event.key == K_c:
                    active_mode = "cpu"
                    print(f"[{t:.2f}] >> FOCUS: CPU")
                elif event.key == K_r:
                    active_mode = "ram"
                    print(f"[{t:.2f}] >> FOCUS: RAM")
                elif event.key == K_j:
                    active_mode = "joy_input"
                    print(f"[{t:.2f}] >> FOCUS: 😊 JOY")
                elif event.key == K_u:
                    active_mode = "sadness_input"
                    print(f"[{t:.2f}] >> FOCUS: 😢 SADNESS")
                elif event.key == K_g:
                    active_mode = "anger_input"
                    print(f"[{t:.2f}] >> FOCUS: 😠 ANGER")
                elif event.key == K_i:
                    active_mode = "fear_input"
                    print(f"[{t:.2f}] >> FOCUS: 😱 FEAR")
                
                # ÉTATS CONSCIENCE
                elif event.key == K_z:
                    sas.toggle_sleep(t)
                elif event.key == K_o:
                    sas.toggle_dream(t)
                elif event.key == K_n:
                    sas.toggle_nightmare(t)
                
                # AJUSTEMENTS
                elif event.key in [K_UP, K_DOWN]:
                    if sas.nightmare_active:
                        delta = 0.1 if event.key == K_UP else -0.1
                        sas.adjust_nightmare_intensity(delta)
                    elif sas.dream_active:
                        delta = 0.1 if event.key == K_UP else -0.1
                        sas.adjust_dream_intensity(delta)
                    else:
                        k = active_mode
                        step = 0.05 if k in ["fel", "light"] else 0.1
                        
                        if event.key == K_UP:
                            sas.targets[k] = min(1.0 if k != "pressure" else 1.8, sas.targets[k] + step)
                        else:
                            sas.targets[k] = max(0.0, sas.targets[k] - step)
                        
                        print(f"[{t:.2f}] {k.upper()}: {sas.targets[k]:.2f}")
                
                # DÉCLENCHEURS
                elif event.key == K_s:
                    sas.targets["stress"] = 1.0
                    print(f"[{t:.2f}] >> ⚡ ACUTE PAIN")
                elif event.key == K_m:
                    sas.targets["morph_m"] = 1.0 if sas.targets["morph_m"] == 0.0 else 0.0
                    print(f"[{t:.2f}] >> MORPH: {'A→B' if sas.targets['morph_m'] == 1.0 else 'B→A'}")
                
                # DEFAULT (D)
                elif event.key == K_d:
                    for key in sas.default_state:
                        sas.targets[key] = sas.default_state[key]
                    sas.sleep_active = False
                    sas.dream_active = False
                    sas.nightmare_active = False
                    sas.sleep_transition = 0.0
                    print(f"[{t:.2f}] >> DEFAULT - Reset complet")
                
                # DEBUG (Bl)
                elif event.key == K_b:
                    print(f"\n═══ DEBUG [{t:.2f}s] ═══")
                    print(f"{'SLEEP':<12}: {'ACTIF' if sas.sleep_active else 'INACTIF'} (transition: {sas.sleep_transition:.2f})")
                    if sas.dream_active:
                        print(f"{'DREAM':<12}: ACTIF (intensité: {sas.dream_intensity:.2f})")
                    if sas.nightmare_active:
                        print(f"{'NIGHTMARE':<12}: ACTIF (intensité: {sas.nightmare_intensity:.2f})")
                    print(f"\n{'INTERNAL':<12}: {sas.internal_emotion.upper()} ({sas.internal_intensity:.2f})")
                    print(f"{'DISPLAYED':<12}: {sas.displayed_emotion.upper()} ({sas.displayed_intensity:.2f})")
                    print(f"{'MASKING':<12}: {'YES 🎭' if sas.is_masking else 'NO ✨'}")
                    if sas.is_masking:
                        print(f"{'EFFORT':<12}: {sas.masking_effort:.2f}")
                    print(f"\n{'TEMP':<12}: {sas.current['temp']:.2f} ({'GLACIER' if sas.current['temp'] < 0.4 else 'LAVA' if sas.current['temp'] > 0.6 else 'NEUTRE'})")
                    print("═" * 40 + "\n")
        
        glPushMatrix()
        glRotatef(15, 1, 0.5, 0)
        sas.draw(t)
        glPopMatrix()
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
