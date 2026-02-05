import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from enum import Enum

class DistributionMode(Enum):
    FIBONACCI = 0
    CONCENTRIC = 1
    CONCENTRIC_LINES = 2 # Sera géré comme CONCENTRIC avec juste les lignes
    QUAD = 3
    TRIANGLE = 4
    HEXAGONAL = 5
    SPIRAL = 6

class CoreMode(Enum):
    SMOOTH = 0
    QUAD = 1
    TRIANGLE = 2
    HEXAGONAL = 3

class LifeEntity:
    def __init__(self):
        self.memb_mode = DistributionMode.FIBONACCI
        self.core_mode = CoreMode.SMOOTH
        self.time = 0.0
        self.fps = 25
        
        # --- PARAMÈTRES DE DENSITÉ ET FORME ---
        self.nb_particles = 1500  # Nombre de points pour la membrane
        self.res_u, self.res_v = 30, 30 # Résolution de la grille pour les maillages
        self.patatoid_noise_amp_memb = 0.15 # Amplitude du bruit pour la membrane
        self.patatoid_noise_amp_core = 0.1 # Amplitude du bruit pour le noyau
        
        # --- CHROMATIQUE ---
        self.memb_color = np.array([0.3, 0.8, 1.0]) # Bleu électrique lumineux
        self.core_c1 = np.array([0.0, 0.4, 1.0]) # Bleu
        self.core_c2 = np.array([0.8, 0.0, 1.0]) # Mauve Néon
        
        # --- TRANSITION DE COULEUR ---
        self.color_transition_frames = 25  # Nombre de frames pour la transition (1 seconde à 25 fps)
        self.color_transition_progress = 0  # Progression de la transition (0 à transition_frames)
        self.is_transitioning = False
        self.core_c1_target = None  # Couleur cible pour C1
        self.core_c2_target = None  # Couleur cible pour C2
        self.core_c1_start = None   # Couleur de départ pour C1
        self.core_c2_start = None   # Couleur de départ pour C2
        self.memb_color_target = None  # Couleur cible pour la membrane
        self.memb_color_start = None   # Couleur de départ pour la membrane
        
        # --- SYSTÈME RESPIRATOIRE ---
        self.cycle_duration = 4.0
        self.expansion_ratio = 0.4
        self.retraction_ratio = 0.6
        self.volume_boost = 0.4
        self.lut_res = 1000
        self.breath_lut = self._precompute_breath_curve()
        
        # --- ROTATIONS ---
        self.rot_core = [0.0, 0.0, 0.0]
        self.rot_speed_core = [0.8, 1.2, 0.5]
        self.rot_memb = [0.0, 0.0, 0.0]
        self.rot_speed_memb = [-0.3, 0.4, -0.2]

    def _precompute_breath_curve(self):
        lut = np.zeros(self.lut_res)
        for i in range(self.lut_res):
            phase = i / self.lut_res
            if phase < self.expansion_ratio:
                t = phase / self.expansion_ratio
                val = (np.exp(3 * t) - 1) / (np.exp(3) - 1)
            else:
                t = (phase - self.expansion_ratio) / self.retraction_ratio
                val = 1 - (np.log(1 + 9 * t) / np.log(10))
            lut[i] = 1.0 + (val * self.volume_boost)
        return lut

    def get_breath_scale(self):
        idx = int(((self.time % self.cycle_duration) / self.cycle_duration) * self.lut_res)
        return self.breath_lut[idx % self.lut_res]
    
    def start_color_transition(self, new_c1, new_c2, new_memb_color=None, transition_frames=None):
        """Démarre une transition de couleur vers de nouvelles couleurs cibles
        
        Args:
            new_c1: Nouvelle couleur pour le noyau C1
            new_c2: Nouvelle couleur pour le noyau C2
            new_memb_color: Nouvelle couleur pour la membrane (optionnel)
            transition_frames: Nombre de frames pour la transition (optionnel)
        """
        if transition_frames is not None:
            self.color_transition_frames = transition_frames
        
        self.core_c1_start = self.core_c1.copy()
        self.core_c2_start = self.core_c2.copy()
        self.core_c1_target = np.array(new_c1)
        self.core_c2_target = np.array(new_c2)
        
        # Transition de la membrane si spécifiée
        if new_memb_color is not None:
            self.memb_color_start = self.memb_color.copy()
            self.memb_color_target = np.array(new_memb_color)
        else:
            self.memb_color_start = None
            self.memb_color_target = None
        
        self.color_transition_progress = 0
        self.is_transitioning = True
    
    def update_color_transition(self):
        """Met à jour la transition de couleur si elle est active"""
        if self.is_transitioning:
            self.color_transition_progress += 1
            
            # Calculer le facteur d'interpolation (easing smoothstep pour une transition fluide)
            t = self.color_transition_progress / self.color_transition_frames
            t = t * t * (3 - 2 * t)  # Smoothstep
            
            # Interpoler les couleurs du noyau
            self.core_c1 = self.core_c1_start * (1 - t) + self.core_c1_target * t
            self.core_c2 = self.core_c2_start * (1 - t) + self.core_c2_target * t
            
            # Interpoler la couleur de la membrane si spécifiée
            if self.memb_color_target is not None:
                self.memb_color = self.memb_color_start * (1 - t) + self.memb_color_target * t
            
            # Terminer la transition
            if self.color_transition_progress >= self.color_transition_frames:
                self.core_c1 = self.core_c1_target.copy()
                self.core_c2 = self.core_c2_target.copy()
                if self.memb_color_target is not None:
                    self.memb_color = self.memb_color_target.copy()
                self.is_transitioning = False

    def _get_patatoid_v(self, v_unit, noise_amp, base_scale):
        # Applique le bruit sinusoïdal pour la forme organique
        noise = np.sin(v_unit[0]*4 + self.time*0.5) * np.cos(v_unit[1]*4 + self.time*0.5)
        return v_unit * (base_scale + noise_amp * noise)

    def _create_hexagon_vertices(self, center, size):
        """Crée les 6 sommets d'un hexagone plat"""
        vertices = []
        for i in range(6):
            angle = np.pi / 3 * i  # 60 degrés
            x = center[0] + size * np.cos(angle)
            y = center[1] + size * np.sin(angle)
            vertices.append([x, y])
        return vertices

    def draw_membrane(self, scale):
        glPushMatrix()
        glRotatef(self.rot_memb[0], 1, 0, 0); glRotatef(self.rot_memb[1], 0, 1, 0); glRotatef(self.rot_memb[2], 0, 0, 1)
        
        # --- CALCUL DES POINTS FIBONACCI (BASE POUR TOUS LES MODES) ---
        fib_points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(self.nb_particles):
            y = 1 - (i / float(self.nb_particles - 1)) * 2 
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            v_unit = np.array([x, y, z])
            v_unit = v_unit / np.linalg.norm(v_unit)  # Normalisation pour une vraie sphère
            fib_points.append(self._get_patatoid_v(v_unit, self.patatoid_noise_amp_memb, 1.5) * scale)

        # --- RENDU DES POINTS ET LIAISONS ---
        glPointSize(2.0)  # Réduction de la taille des particules
        
        if self.memb_mode == DistributionMode.FIBONACCI:
            # Mode FIBONACCI : juste les points avec effet orb
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            
            glBegin(GL_POINTS)
            for idx, p in enumerate(fib_points):
                depth_lum = 0.5 + (p[2] + 2.5) / 5.0
                pulse = np.sin(self.time * 2.0 + idx * 0.1) * 0.2 + 1.0
                dist_from_center = np.linalg.norm(p) / (1.5 * scale)
                radial_lum = 0.7 + 0.3 * (1.0 - dist_from_center)
                final_lum = depth_lum * pulse * radial_lum
                final_lum = np.clip(final_lum, 0.5, 1.3)
                alpha = 0.85 + 0.15 * (pulse - 1.0) / 0.2
                
                glColor4f(
                    self.memb_color[0] * final_lum, 
                    self.memb_color[1] * final_lum, 
                    self.memb_color[2] * final_lum, 
                    alpha
                )
                glVertex3fv(p)
            glEnd()
            glDisable(GL_POINT_SMOOTH)
            
        elif self.memb_mode == DistributionMode.CONCENTRIC or self.memb_mode == DistributionMode.CONCENTRIC_LINES:
            # Mode CONCENTRIC : cercles concentriques de latitude
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            
            num_circles = 20  # Nombre de cercles de latitude
            points_per_circle = 40  # Nombre de points par cercle
            
            # Stocker les points pour les dessiner et éventuellement les relier
            all_circle_points = []
            
            for i in range(num_circles):
                lat = np.pi * (i / (num_circles - 1) - 0.5)  # De -π/2 à π/2
                circle_points = []
                
                for j in range(points_per_circle):
                    lon = 2 * np.pi * j / points_per_circle
                    v_unit = np.array([np.cos(lon)*np.cos(lat), np.sin(lon)*np.cos(lat), np.sin(lat)])
                    p = self._get_patatoid_v(v_unit, self.patatoid_noise_amp_memb, 1.5) * scale
                    circle_points.append(p)
                
                all_circle_points.append(circle_points)
            
            # Dessiner les points
            glBegin(GL_POINTS)
            for circle_points in all_circle_points:
                for idx, p in enumerate(circle_points):
                    depth_lum = 0.5 + (p[2] + 2.5) / 5.0
                    pulse = np.sin(self.time * 2.0 + idx * 0.1) * 0.2 + 1.0
                    dist_from_center = np.linalg.norm(p) / (1.5 * scale)
                    radial_lum = 0.7 + 0.3 * (1.0 - dist_from_center)
                    final_lum = depth_lum * pulse * radial_lum
                    final_lum = np.clip(final_lum, 0.5, 1.3)
                    alpha = 0.85 + 0.15 * (pulse - 1.0) / 0.2
                    
                    glColor4f(
                        self.memb_color[0] * final_lum, 
                        self.memb_color[1] * final_lum, 
                        self.memb_color[2] * final_lum, 
                        alpha
                    )
                    glVertex3fv(p)
            glEnd()
            glDisable(GL_POINT_SMOOTH)
            
            # Si mode CONCENTRIC_LINES, ajouter les liaisons
            if self.memb_mode == DistributionMode.CONCENTRIC_LINES:
                glLineWidth(1.0)
                
                # Lignes horizontales (le long des cercles de latitude)
                for circle_points in all_circle_points:
                    glBegin(GL_LINE_LOOP)
                    for p in circle_points:
                        lum_line = max(0.05, (p[2]+2.5)/5) * 0.25
                        glColor4f(self.memb_color[0]*lum_line, self.memb_color[1]*lum_line, 
                                 self.memb_color[2]*lum_line, 0.3)
                        glVertex3fv(p)
                    glEnd()
                
                # Lignes verticales (méridiens reliant les cercles)
                glBegin(GL_LINES)
                for j in range(points_per_circle):
                    for i in range(len(all_circle_points) - 1):
                        p1 = all_circle_points[i][j]
                        p2 = all_circle_points[i+1][j]
                        
                        lum_line = max(0.05, (p1[2]+2.5)/5) * 0.25
                        glColor4f(self.memb_color[0]*lum_line, self.memb_color[1]*lum_line, 
                                 self.memb_color[2]*lum_line, 0.3)
                        glVertex3fv(p1)
                        glVertex3fv(p2)
                glEnd()
            
        elif self.memb_mode == DistributionMode.HEXAGONAL:
            # Mode HEXAGONAL : vraie grille hexagonale en nid d'abeille
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            
            hex_size = 0.15
            hex_height = hex_size * np.sqrt(3)
            
            # Stocker les centres des hexagones et leurs points 3D
            hexagons = []
            
            for i in range(self.res_u):
                for j in range(self.res_v):
                    # Position 2D de base pour le centre de l'hexagone
                    offset_x = hex_size * 1.5 * j
                    offset_y = hex_height * i
                    if j % 2 == 1:
                        offset_y += hex_height / 2
                    
                    # Normaliser pour mapper sur sphère
                    u = offset_x / (hex_size * 1.5 * self.res_v) * 2 * np.pi
                    v = offset_y / (hex_height * self.res_u) * np.pi - np.pi/2
                    
                    # Projection sphérique
                    v_unit = np.array([np.cos(u)*np.cos(v), np.sin(u)*np.cos(v), np.sin(v)])
                    center_3d = self._get_patatoid_v(v_unit, self.patatoid_noise_amp_memb, 1.5) * scale
                    
                    # Créer les 6 sommets de l'hexagone
                    hex_verts_3d = []
                    for k in range(6):
                        angle = np.pi / 3 * k
                        du = hex_size * np.cos(angle) / (self.res_v * 1.5)
                        dv = hex_size * np.sin(angle) / (self.res_u * hex_height / np.pi)
                        
                        u_vert = u + du * 2 * np.pi
                        v_vert = v + dv * np.pi
                        v_vert = np.clip(v_vert, -np.pi/2 + 0.01, np.pi/2 - 0.01)
                        
                        v_unit_vert = np.array([np.cos(u_vert)*np.cos(v_vert), 
                                                np.sin(u_vert)*np.cos(v_vert), 
                                                np.sin(v_vert)])
                        vert_3d = self._get_patatoid_v(v_unit_vert, self.patatoid_noise_amp_memb, 1.5) * scale
                        hex_verts_3d.append(vert_3d)
                    
                    hexagons.append((center_3d, hex_verts_3d))
            
            # Dessiner les points aux sommets
            glBegin(GL_POINTS)
            for center_3d, hex_verts in hexagons:
                for vert in hex_verts:
                    depth_lum = 0.5 + (vert[2] + 2.5) / 5.0
                    pulse = np.sin(self.time * 2.0) * 0.2 + 1.0
                    dist_from_center = np.linalg.norm(vert) / (1.5 * scale)
                    radial_lum = 0.7 + 0.3 * (1.0 - dist_from_center)
                    final_lum = depth_lum * pulse * radial_lum
                    final_lum = np.clip(final_lum, 0.5, 1.3)
                    
                    glColor4f(
                        self.memb_color[0] * final_lum,
                        self.memb_color[1] * final_lum,
                        self.memb_color[2] * final_lum,
                        0.85
                    )
                    glVertex3fv(vert)
            glEnd()
            glDisable(GL_POINT_SMOOTH)
            
            # Dessiner les arêtes des hexagones
            glLineWidth(1.0)
            glBegin(GL_LINES)
            for center_3d, hex_verts in hexagons:
                lum_line = max(0.05, (center_3d[2]+2.5)/5) * 0.25
                glColor4f(self.memb_color[0]*lum_line, self.memb_color[1]*lum_line, 
                         self.memb_color[2]*lum_line, 0.3)
                
                for k in range(6):
                    glVertex3fv(hex_verts[k])
                    glVertex3fv(hex_verts[(k+1) % 6])
            glEnd()
            
        else:
            # Modes avec maillage standard (QUAD, TRIANGLE, SPIRAL)
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            
            glBegin(GL_POINTS)
            for i in range(self.res_u):
                for j in range(self.res_v):
                    lat = np.pi * (i / (self.res_u - 1) - 0.5)
                    lon = 2 * np.pi * j / self.res_v
                    v_unit = np.array([np.cos(lon)*np.cos(lat), np.sin(lon)*np.cos(lat), np.sin(lat)])
                    p = self._get_patatoid_v(v_unit, self.patatoid_noise_amp_memb, 1.5) * scale
                    
                    depth_lum = 0.5 + (p[2] + 2.5) / 5.0
                    idx = i * self.res_v + j
                    pulse = np.sin(self.time * 2.0 + idx * 0.05) * 0.2 + 1.0
                    dist_from_center = np.linalg.norm(p) / (1.5 * scale)
                    radial_lum = 0.7 + 0.3 * (1.0 - dist_from_center)
                    final_lum = depth_lum * pulse * radial_lum
                    final_lum = np.clip(final_lum, 0.5, 1.3)
                    alpha = 0.85 + 0.15 * (pulse - 1.0) / 0.2
                    
                    glColor4f(
                        self.memb_color[0] * final_lum, 
                        self.memb_color[1] * final_lum, 
                        self.memb_color[2] * final_lum, 
                        alpha
                    )
                    glVertex3fv(p)
            glEnd()
            glDisable(GL_POINT_SMOOTH)

            # Liaisons
            glLineWidth(1.0)
            if self.memb_mode == DistributionMode.SPIRAL: glBegin(GL_LINE_STRIP)
            else: glBegin(GL_LINES)
            
            for i in range(self.res_u):
                for j in range(self.res_v):
                    lat1 = np.pi * (i / (self.res_u - 1) - 0.5)
                    lon1 = 2 * np.pi * j / self.res_v
                    v_unit1 = np.array([np.cos(lon1)*np.cos(lat1), np.sin(lon1)*np.cos(lat1), np.sin(lat1)])
                    p1 = self._get_patatoid_v(v_unit1, self.patatoid_noise_amp_memb, 1.5) * scale
                    
                    lum_line = max(0.05, (p1[2]+2.5)/5) * 0.25
                    glColor4f(self.memb_color[0]*lum_line, self.memb_color[1]*lum_line, 
                             self.memb_color[2]*lum_line, 0.3)

                    if self.memb_mode == DistributionMode.SPIRAL:
                        glVertex3fv(p1)
                    else:
                        lat2 = lat1
                        lon2 = 2 * np.pi * ((j+1) % self.res_v) / self.res_v
                        v_unit2 = np.array([np.cos(lon2)*np.cos(lat2), np.sin(lon2)*np.cos(lat2), np.sin(lat2)])
                        p2 = self._get_patatoid_v(v_unit2, self.patatoid_noise_amp_memb, 1.5) * scale
                        glVertex3fv(p1); glVertex3fv(p2)
                        
                        if i < self.res_u - 1:
                            lat3 = np.pi * ((i+1) / (self.res_u - 1) - 0.5)
                            lon3 = 2 * np.pi * j / self.res_v
                            v_unit3 = np.array([np.cos(lon3)*np.cos(lat3), np.sin(lon3)*np.cos(lat3), np.sin(lat3)])
                            p3 = self._get_patatoid_v(v_unit3, self.patatoid_noise_amp_memb, 1.5) * scale
                            glVertex3fv(p1); glVertex3fv(p3)
                            
                            if self.memb_mode == DistributionMode.TRIANGLE:
                                lat4 = np.pi * ((i+1) / (self.res_u - 1) - 0.5)
                                lon4 = 2 * np.pi * ((j+1) % self.res_v) / self.res_v
                                v_unit4 = np.array([np.cos(lon4)*np.cos(lat4), np.sin(lon4)*np.cos(lat4), np.sin(lat4)])
                                p4 = self._get_patatoid_v(v_unit4, self.patatoid_noise_amp_memb, 1.5) * scale
                                glVertex3fv(p1); glVertex3fv(p4)
            glEnd()
        glPopMatrix()

    def draw_core(self, scale):
        glPushMatrix()
        glRotatef(self.rot_core[0], 1, 0, 0); glRotatef(self.rot_core[1], 0, 1, 0); glRotatef(self.rot_core[2], 0, 0, 1)
        
        res = 22 # Résolution du noyau
        
        if self.core_mode == CoreMode.HEXAGONAL:
            # Mode hexagonal : créer de vrais hexagones
            hex_size = 0.12
            hex_height = hex_size * np.sqrt(3)
            
            for i in range(res - 1):
                for j in range(res):
                    # Position 2D
                    offset_x = hex_size * 1.5 * j
                    offset_y = hex_height * i
                    if j % 2 == 1:
                        offset_y += hex_height / 2
                    
                    # Mapper sur la sphère
                    u = offset_x / (hex_size * 1.5 * res) * 2 * np.pi
                    v = offset_y / (hex_height * res) * np.pi - np.pi/2
                    
                    # Créer l'hexagone
                    hex_verts = []
                    for k in range(6):
                        angle = np.pi / 3 * k
                        du = hex_size * np.cos(angle) / (res * 1.5)
                        dv = hex_size * np.sin(angle) / (res * hex_height / np.pi)
                        
                        u_vert = u + du * 2 * np.pi
                        v_vert = v + dv * np.pi
                        v_vert = np.clip(v_vert, -np.pi/2 + 0.01, np.pi/2 - 0.01)
                        
                        v_unit = np.array([np.cos(u_vert)*np.cos(v_vert), 
                                          np.sin(u_vert)*np.cos(v_vert), 
                                          np.sin(v_vert)])
                        p = self._get_patatoid_v(v_unit, self.patatoid_noise_amp_core, 1.1) * scale
                        
                        # Couleur basée sur position centrale
                        v_unit_center = np.array([np.cos(u)*np.cos(v), np.sin(u)*np.cos(v), np.sin(v)])
                        t = (np.sin(v_unit_center[0]*2 + v_unit_center[1]*2 + self.time * 2.0) + 1.0) / 2.0
                        
                        hex_verts.append((p, t))
                    
                    # Dessiner l'hexagone comme polygone
                    glBegin(GL_POLYGON)
                    for p, t in hex_verts:
                        glColor3f(*(self.core_c1*(1-t) + self.core_c2*t))
                        glVertex3fv(p)
                    glEnd()
                    
                    # Arêtes de l'hexagone
                    glLineWidth(1.5)
                    glBegin(GL_LINE_LOOP)
                    glColor3f(0.1, 0.1, 0.15)
                    for p, t in hex_verts:
                        glVertex3fv(p)
                    glEnd()
        else:
            # Modes SMOOTH, QUAD, TRIANGLE
            for i in range(res - 1):
                for j in range(res):
                    coords = []
                    for di, dj in [(0,0), (1,0), (1,1), (0,1)]:
                        lat = np.pi * ((i+di) / (res - 1) - 0.5)
                        lon = 2 * np.pi * (j+dj) / res
                        
                        v_unit = np.array([np.cos(lon)*np.cos(lat), np.sin(lon)*np.cos(lat), np.sin(lat)])
                        p_base = self._get_patatoid_v(v_unit, self.patatoid_noise_amp_core, 1.1) * scale
                        t = (np.sin(v_unit[0]*2 + v_unit[1]*2 + self.time * 2.0) + 1.0) / 2.0
                        coords.append((p_base, t))

                    if self.core_mode == CoreMode.SMOOTH:
                        glBegin(GL_TRIANGLE_STRIP)
                        for p, t in [coords[0], coords[1], coords[3], coords[2]]:
                            glColor3f(*(self.core_c1*(1-t) + self.core_c2*t))
                            glVertex3fv(p)
                        glEnd()
                    elif self.core_mode == CoreMode.TRIANGLE:
                        glBegin(GL_TRIANGLES)
                        p,t = coords[0]; glColor3f(*(self.core_c1*(1-t)+self.core_c2*t)); glVertex3fv(p)
                        p,t = coords[1]; glColor3f(*(self.core_c1*(1-t)+self.core_c2*t)); glVertex3fv(p)
                        p,t = coords[2]; glColor3f(*(self.core_c1*(1-t)+self.core_c2*t)); glVertex3fv(p)
                        p,t = coords[0]; glColor3f(*(self.core_c1*(1-t)+self.core_c2*t)); glVertex3fv(p)
                        p,t = coords[2]; glColor3f(*(self.core_c1*(1-t)+self.core_c2*t)); glVertex3fv(p)
                        p,t = coords[3]; glColor3f(*(self.core_c1*(1-t)+self.core_c2*t)); glVertex3fv(p)
                        glEnd()
                        
                        glLineWidth(1.5)
                        glBegin(GL_LINES)
                        glColor3f(0.1, 0.1, 0.15)
                        glVertex3fv(coords[0][0]); glVertex3fv(coords[1][0])
                        glVertex3fv(coords[1][0]); glVertex3fv(coords[2][0])
                        glVertex3fv(coords[2][0]); glVertex3fv(coords[0][0])
                        glVertex3fv(coords[0][0]); glVertex3fv(coords[2][0])
                        glVertex3fv(coords[2][0]); glVertex3fv(coords[3][0])
                        glVertex3fv(coords[3][0]); glVertex3fv(coords[0][0])
                        glEnd()
                    else: # QUAD
                        glBegin(GL_QUADS)
                        for p, t in coords:
                            glColor3f(*(self.core_c1*(1-t) + self.core_c2*t))
                            glVertex3fv(p)
                        glEnd()
                        
                        glLineWidth(1.5)
                        glBegin(GL_LINE_LOOP)
                        glColor3f(0.1, 0.1, 0.15)
                        for p, t in coords:
                            glVertex3fv(p)
                        glEnd()
        glPopMatrix()

    def update(self):
        self.time += 1.0 / self.fps
        for i in range(3):
            self.rot_core[i] += self.rot_speed_core[i]
            self.rot_memb[i] += self.rot_speed_memb[i]
        
        # Mettre à jour la transition de couleur
        self.update_color_transition()

def main():
    pygame.init()
    pygame.display.set_mode((1024, 768), DOUBLEBUF | OPENGL)
    gluPerspective(45, 1.33, 0.1, 50.0); glTranslatef(0, 0, -8.0)
    glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    entity = LifeEntity(); clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m: entity.memb_mode = DistributionMode((entity.memb_mode.value+1)%len(DistributionMode))
                if event.key == pygame.K_n: entity.core_mode = CoreMode((entity.core_mode.value+1)%len(CoreMode))
                if event.key == pygame.K_s:
                    # Transition vers rouge néon et jaune orangé néon pour le noyau
                    # et or lumineux pour la membrane
                    red_neon = [1.0, 0.1, 0.1]           # Rouge néon
                    yellow_orange_neon = [1.0, 0.7, 0.0] # Jaune orangé néon
                    gold_luminous = [1.0, 0.84, 0.0]     # Or lumineux
                    entity.start_color_transition(red_neon, yellow_orange_neon, 
                                                 new_memb_color=gold_luminous, 
                                                 transition_frames=25)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        entity.update(); s = entity.get_breath_scale()
        entity.draw_core(s); entity.draw_membrane(s)
        pygame.display.flip(); clock.tick(entity.fps)

if __name__ == "__main__": main()
