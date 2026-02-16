import sys
import os
import signal
import threading
import time
import json
import logging
from collections import deque
import numpy as np
import zenoh
import pygame
import math

# --- Configuration des logs ---
logger = logging.getLogger("StreamsVisualizer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
logger.addHandler(handler)

# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    @staticmethod
    def load(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Validation basique
            required_keys = ['element_types', 'groups', 'oscilloscope']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
            
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise

# ============================================================================
# Color Pulse Engine
# ============================================================================

class ColorPulseEngine:
    def __init__(self, config):
        self.config = config
        self.palettes = {}
        for type_name, type_config in config['element_types'].items():
            self.palettes[type_name] = {
                'dark': self._hex_to_rgb(type_config['color_palette']['dark']),
                'medium': self._hex_to_rgb(type_config['color_palette']['medium']),
                'bright': self._hex_to_rgb(type_config['color_palette']['bright'])
            }
    
    def get_color_from_frequency(self, frequency, type_name):
        type_config = self.config['element_types'][type_name]
        palette = self.palettes[type_name]
        low, high = type_config['frequency_thresholds']['low'], type_config['frequency_thresholds']['high']
        
        if frequency < low:
            t = frequency / low if low > 0 else 0
            return self._interpolate_rgb(palette['dark'], palette['medium'], t)
        elif frequency < high:
            t = (frequency - low) / (high - low) if (high - low) > 0 else 0
            return self._interpolate_rgb(palette['medium'], palette['bright'], t)
        return palette['bright']

    def get_halo_intensity(self, frequency, type_name):
        """Calcule l'intensité du halo (0-1) basée sur la fréquence"""
        type_config = self.config['element_types'][type_name]
        low, high = type_config['frequency_thresholds']['low'], type_config['frequency_thresholds']['high']
        
        if frequency <= low:
            return 0.2  # Halo faible
        elif frequency >= high:
            return 1.0  # Halo maximum
        else:
            # Interpolation linéaire entre low et high
            return 0.2 + 0.8 * ((frequency - low) / (high - low))

    @staticmethod
    def _interpolate_rgb(c1, c2, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))

    @staticmethod
    def _hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ============================================================================
# Stream Line (Modèle de données)
# ============================================================================

class StreamLine:
    def __init__(self, topic, type_name, color_engine, window_duration=5.0, max_impulses=1000):
        self.topic = topic
        self.type = type_name
        self.color_engine = color_engine
        self.window_duration = window_duration  # Durée de la fenêtre temporelle
        self.max_impulses = max_impulses
        self.impulse_history = deque(maxlen=max_impulses)  # (timestamp, payload)
        self.last_message_time = None
        self.current_frequency = 0.0
        self.frequency_history = deque(maxlen=10)
        self.last_payload = {}
        self.message_count = 0
        self._lock = threading.Lock()
        
    def on_message_received(self, payload):
        with self._lock:
            now = time.time()
            self.impulse_history.append((now, payload.copy()))  # Stocker le payload complet
            self.message_count += 1
            self.last_payload = payload.copy()
            
            # Calcul de la fréquence instantanée
            if self.last_message_time:
                dt = now - self.last_message_time
                if dt > 0.001:
                    instant_freq = 1.0 / dt
                    if instant_freq < 1000:  # Max 1000 Hz
                        self.frequency_history.append(instant_freq)
                        self.current_frequency = np.mean(self.frequency_history) if self.frequency_history else 0.0
            
            self.last_message_time = now
    
    def get_recent_impulses(self, current_time):
        """Retourne les impulsions dans la fenêtre temporelle"""
        with self._lock:
            cutoff = current_time - self.window_duration
            recent = [(ts, payload) for ts, payload in self.impulse_history if ts >= cutoff]
            return recent

# ============================================================================
# Main Visualizer (Pygame Edition)
# ============================================================================

class StreamsVisualizer:
    def __init__(self, config_filepath):
        pygame.init()
        
        # Configuration
        self.config = ConfigLoader.load(config_filepath)
        self.color_engine = ColorPulseEngine(self.config)
        self.window_duration = self.config.get('oscilloscope', {}).get('window_duration', 5.0)
        self.target_fps = self.config.get('oscilloscope', {}).get('fps', 30)
        
        # Signal handlers pour fermeture propre
        self._setup_signal_handlers()
        
        # Zenoh Init
        self.zenoh_session = self._setup_zenoh()
        
        # Streams & Groups
        self.streams = {}
        self._setup_subscribers()
        
        # Pygame Setup
        self.screen = pygame.display.set_mode((1400, 900), pygame.RESIZABLE)  # Plus large pour les payloads
        pygame.display.set_caption("LIFE 2025 - Écho du Vide")
        self.font_main = pygame.font.SysFont("Monospace", 18, bold=True)
        self.font_small = pygame.font.SysFont("Monospace", 12)
        self.font_payload = pygame.font.SysFont("Monospace", 10)  # Police plus petite pour les payloads
        self.clock = pygame.time.Clock()
        self.running = True
        self.last_stats_time = time.time()
        self.frame_count = 0
        
        logger.info(f"StreamsVisualizer initialized with {self.target_fps} FPS target")

    def _setup_signal_handlers(self):
        def signal_handler(sig, frame):
            logger.info("Received interrupt signal, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_zenoh(self):
        conf = zenoh.Config()
        zc = self.config.get('zenoh', {})
        
        if 'mode' in zc:
            conf.insert_json5("mode", json.dumps(zc['mode']))
        
        if 'connect' in zc and 'endpoints' in zc['connect']:
            endpoints = zc['connect']['endpoints']
            if isinstance(endpoints, list):
                conf.insert_json5("connect/endpoints", json.dumps(endpoints))
        
        try:
            session = zenoh.open(conf)
            logger.info("Zenoh session opened successfully")
            return session
        except Exception as e:
            logger.error(f"Failed to open Zenoh session: {e}")
            raise

    def _setup_subscribers(self):
        for group in self.config['groups']:
            for el in group['elements']:
                topic = el['topic']
                self.streams[topic] = StreamLine(
                    topic, 
                    el['type'], 
                    self.color_engine, 
                    self.window_duration
                )
                try:
                    self.zenoh_session.declare_subscriber(topic, self._zenoh_callback)
                    logger.info(f"Subscribed to topic: {topic}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to {topic}: {e}")

    def _zenoh_callback(self, sample):
        topic = str(sample.key_expr)
        try:
            payload = json.loads(sample.payload.to_string())
            if topic in self.streams:
                self.streams[topic].on_message_received(payload)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for topic {topic}: {e}")
        except Exception as e:
            logger.error(f"Zenoh callback error for {topic}: {e}")

    def draw(self):
        self.screen.fill((5, 5, 5))
        y_offset = 60
        margin = 20
        win_w, win_h = self.screen.get_size()
        
        # Largeur de l'oscilloscope (avec espace pour le payload)
        osc_width = int(win_w * 0.6)  # 60% pour l'oscilloscope
        payload_width = win_w - osc_width - margin * 2  # Le reste pour le payload
        
        # Calcul FPS
        self.frame_count += 1
        now = time.time()
        if now - self.last_stats_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_stats_time = now
        
        # Header
        header_text = f"🧠 STREAMS VISUALIZER - FENÊTRE: {self.window_duration}s | FPS: {getattr(self, 'fps', 0)}/{self.target_fps}"
        header_surf = self.font_main.render(header_text, True, (0, 255, 255))
        self.screen.blit(header_surf, (win_w//2 - header_surf.get_width()//2, 15))

        for group in self.config['groups']:
            # Titre Groupe
            group_label = self.font_main.render(f"[Groupe: {group['name']}]", True, (0, 191, 255))
            self.screen.blit(group_label, (margin, y_offset))
            y_offset += 30
            
            for el in group['elements']:
                stream = self.streams[el['topic']]
                if y_offset + 120 <= win_h:
                    self._draw_stream(stream, margin, y_offset, osc_width, 100, payload_width)
                    y_offset += 120

    def _draw_stream(self, stream, x, y, w, h, payload_width):
        now = time.time()
        
        # Couleur de base basée sur la fréquence
        base_color = stream.color_engine.get_color_from_frequency(stream.current_frequency, stream.type)
        halo_intensity = stream.color_engine.get_halo_intensity(stream.current_frequency, stream.type)
        
        # ===== DESSIN DE L'OSCILLOSCOPE =====
        # Dessiner le halo (effet de lueur)
        for i in range(5, 0, -1):
            alpha = int(15 * halo_intensity * (i / 5))
            if alpha > 0:
                halo_surf = pygame.Surface((w, h), pygame.SRCALPHA)
                halo_color = (*base_color, alpha)
                pygame.draw.rect(halo_surf, halo_color, (0, 0, w, h), 3)
                self.screen.blit(halo_surf, (x, y))
        
        # Fond sombre
        pygame.draw.rect(self.screen, (10, 10, 10), (x, y, w, h))
        pygame.draw.rect(self.screen, base_color, (x, y, w, h), 1)
        
        # Ligne horizontale centrale (le "temps")
        mid_y = y + h // 2
        line_color = self._interpolate_color(base_color, (200, 200, 200), 0.7)
        pygame.draw.line(self.screen, line_color, (x, mid_y), (x + w, mid_y), 2)
        
        # Point représentant l'instant courant (extrême droite)
        pygame.draw.circle(self.screen, (255, 255, 255), (x + w, mid_y), 4)
        pygame.draw.circle(self.screen, base_color, (x + w, mid_y), 2)
        
        # Récupérer les impulsions récentes
        recent_impulses = stream.get_recent_impulses(now)
        
        # Hauteur fixe des pics (90% de la moitié de l'oscilloscope)
        peak_h = int(h * 0.45)  # 90% de la moitié (h/2 * 0.9)
        
        # Dessiner les pics (crêtes) - tous de la même hauteur
        for timestamp, payload in recent_impulses:
            # Calculer la position x basée sur le timestamp
            age = now - timestamp
            progress = 1.0 - (age / stream.window_duration)  # 0 = vieux, 1 = récent
            px = x + int(w * progress)
            
            # Couleur du pic (plus intense si récent)
            color_intensity = 0.5 + 0.5 * (1.0 - age / stream.window_duration)
            pic_color = self._interpolate_color(base_color, (255, 255, 255), color_intensity)
            
            # Dessiner le pic (ligne verticale) - hauteur fixe
            # Effet de trail pour les pics récents
            if progress > 0.8:  # Derniers 20% de la fenêtre
                for trail in range(3):
                    trail_x = px - trail * 2
                    if trail_x >= x:
                        trail_alpha = 0.3 * (1 - trail/3)
                        trail_color = self._interpolate_color(pic_color, base_color, trail_alpha)
                        pygame.draw.line(self.screen, trail_color, 
                                       (trail_x, mid_y), (trail_x, mid_y - peak_h), 2)
            
            # Pic principal
            pygame.draw.line(self.screen, pic_color, 
                           (px, mid_y), (px, mid_y - peak_h), 3)
            
            # Point au sommet
            if progress > 0.9:  # Point plus brillant pour les pics très récents
                pygame.draw.circle(self.screen, (255, 255, 255), (px, mid_y - peak_h), 3)
            else:
                pygame.draw.circle(self.screen, pic_color, (px, mid_y - peak_h), 2)
        
        # Informations textuelles sur l'oscilloscope
        topic_color = (0, 191, 255) if stream.type == "nerf" else (255, 165, 0)
        txt_topic = self.font_small.render(f"{stream.topic}", True, topic_color)
        self.screen.blit(txt_topic, (x + 5, y + 5))
        
        # Fréquence et compteur
        freq_text = f"{stream.current_frequency:.1f} Hz | {stream.message_count} msgs"
        txt_freq = self.font_small.render(freq_text, True, base_color)
        self.screen.blit(txt_freq, (x + 5, y + h - 40))
        
        # Indicateur d'intensité du halo
        halo_indicator = "█" * int(halo_intensity * 10)
        txt_halo = self.font_small.render(halo_indicator, True, base_color)
        self.screen.blit(txt_halo, (x + 5, y + h - 20))
        
        # ===== ZONE D'AFFICHAGE DU PAYLOAD =====
        payload_x = x + w + 20  # Espace après l'oscilloscope
        payload_y = y
        payload_h = h
        
        # Fond pour le payload
        pygame.draw.rect(self.screen, (15, 15, 15), (payload_x, payload_y, payload_width, payload_h))
        pygame.draw.rect(self.screen, base_color, (payload_x, payload_y, payload_width, payload_h), 1)
        
        # Titre de la zone payload
        payload_title = self.font_small.render("📦 DERNIER PAYLOAD", True, base_color)
        self.screen.blit(payload_title, (payload_x + 5, payload_y + 5))
        
        # Afficher le dernier payload
        if stream.last_payload:
            # Formater le payload de façon lisible
            payload_str = json.dumps(stream.last_payload, indent=None)
            
            # Découper le payload en lignes pour l'affichage
            max_chars = int(payload_width / 7)  # Approx caractères par ligne
            words = payload_str.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += (" " + word if current_line else word)
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Si le payload est trop long, tronquer
            if len(lines) > 4:
                lines = lines[:4]
                lines[-1] = lines[-1] + "..."
            
            # Afficher chaque ligne
            line_y = payload_y + 25
            for line in lines:
                payload_surf = self.font_payload.render(line, True, (200, 200, 200))
                self.screen.blit(payload_surf, (payload_x + 5, line_y))
                line_y += 15
            
            # Horodatage du dernier message
            if stream.last_message_time:
                time_str = time.strftime("%H:%M:%S", time.localtime(stream.last_message_time))
                time_surf = self.font_payload.render(f"🕐 {time_str}", True, (150, 150, 150))
                self.screen.blit(time_surf, (payload_x + 5, payload_y + payload_h - 18))
        else:
            # Aucun message reçu
            no_data = self.font_payload.render("⏳ En attente de données...", True, (100, 100, 100))
            self.screen.blit(no_data, (payload_x + 5, payload_y + 40))

    def _interpolate_color(self, c1, c2, t):
        """Interpole entre deux couleurs RGB"""
        return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))

    def run(self):
        logger.info(f"Starting visualization loop at {self.target_fps} FPS")
        frame_time = 1.0 / self.target_fps
        next_frame_time = time.time()
        
        try:
            while self.running:
                # Gestion des événements
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                
                # Rendu
                self.draw()
                pygame.display.flip()
                
                # Contrôle du FPS
                current_time = time.time()
                if current_time < next_frame_time:
                    time.sleep(max(0, next_frame_time - current_time))
                next_frame_time += frame_time
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up resources...")
        if self.zenoh_session:
            try:
                self.zenoh_session.close()
                logger.info("Zenoh session closed")
            except Exception as e:
                logger.error(f"Error closing Zenoh session: {e}")
        pygame.quit()
        logger.info("Pygame closed")

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "streams_config.json"
    
    try:
        logger.info(f"Starting StreamsVisualizer with config: {config_path}")
        viz = StreamsVisualizer(config_path)
        viz.run()
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        sys.exit(1)