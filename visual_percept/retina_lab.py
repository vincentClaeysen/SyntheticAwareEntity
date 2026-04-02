#!/usr/bin/env python3
"""
retina_lab.py — Rétine primitive + eFovea pour Shirka
=====================================================
Stade : jeune enfant (humeur ignorée)

Pipeline :
  1. Rétine périphérique basse résolution → carte de saillance brute
  2. Filtrage GEM (couleurs préférentielles + réactivité)
  3. Détection de multiples POIs avec pondération (saillance, mouvement, contraste, intersections)
  4. Sélection et classement des POIs
  5. eFovea : élargissement par depth map → bbox physique de l'objet
  6. Assainissement des ROIs (filtrage décrochage Z)
  7. Détection des plans (sol, plafond) et des lignes structurelles
  8. Visualisation : chemin attentionnel + mires + heatmap optionnelle

Touche ESPACE : analyse de la frame courante
Touche H      : toggle heatmap
Touche Q      : quitter
"""

import math
import threading
import queue
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

try:
    import depthai as dai
    DEPTHAI_DISPONIBLE = True
except ImportError:
    DEPTHAI_DISPONIBLE = False

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION DES LOGS
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retina_lab")

# ─────────────────────────────────────────────────────────────
#  GEM — Shirka_001
# ─────────────────────────────────────────────────────────────
GEM = {
    "identifiant" : "Shirka_001",
    "naissance"   : "2026-03-07T10:30:00Z",
    "genre"       : "omega",
    "tempo"       : {"base": 0.65, "variabilite": 0.3},
    "intensite"   : {"base": 0.7,  "seuil": 0.4},
    "grace"       : 0.5,
    "reactivite"  : 0.8,
    "emotive"     : 0.6,
    "courbe"      : "exponentielle",
    "couleur"     : {
        "principale" : "#3A6EA5",   # bleu moyen
        "secondaire" : "#E8C1A0",   # beige chaud
        "intensite"  : 0.7
    },
    "resonance"   : {
        "frequence"   : 432,
        "largeur"     : 0.3,
        "harmoniques" : [864, 1296]
    },
    "preferences" : {"direction": 0.3, "interaction": 0.7, "esthetique": 0.5},
}


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convertit #RRGGBB → (B, G, R) OpenCV."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)
    except (ValueError, IndexError) as e:
        logger.error(f"Conversion hex_to_bgr échouée pour {hex_color}: {e}")
        return (128, 128, 128)


# Couleurs préférentielles de Shirka en BGR
COULEUR_PRINCIPALE  = hex_to_bgr(GEM["couleur"]["principale"])
COULEUR_SECONDAIRE  = hex_to_bgr(GEM["couleur"]["secondaire"])
REACTIVITE          = GEM["reactivite"]
INTENSITE_SEUIL     = GEM["intensite"]["seuil"]


# ─────────────────────────────────────────────────────────────
#  PARAMÈTRES
# ─────────────────────────────────────────────────────────────
W, H         = 640, 400
FPS_ACQ      = 20
QUEUE_MAX    = 2
DIV_RETINE   = 4
AF_ZONE_FRAC = 0.25
LENSPOS_MIN  = 0
LENSPOS_MAX  = 255
AF_PERIODE_S = 2.0
EFOVEA_MARGE_FRAC = 0.15
FOCALE_PIXELS = 500.0

# Seuils de sécurité
PROFONDEUR_MIN_MM = 100
PROFONDEUR_MAX_MM = 15000
COHERENCE_RAYON_PX = 8
INTERSECTION_TOLERANCE_PX = 20
LIGNE_ANCRAGE_TOLERANCE_Z_MM = 150


# ─────────────────────────────────────────────────────────────
#  DÉTECTION PLATEFORME
# ─────────────────────────────────────────────────────────────
def _detecter_pi() -> bool:
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "Raspberry Pi" in f.read()
    except Exception:
        return False

IS_PI = _detecter_pi()

WLS_DISPONIBLE = False
try:
    _test = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    WLS_DISPONIBLE = True
except (AttributeError, cv2.error):
    pass

USE_WLS = WLS_DISPONIBLE and not IS_PI

def _detecter_oakd() -> bool:
    if not DEPTHAI_DISPONIBLE:
        return False
    try:
        return len(dai.Device.getAllAvailableDevices()) > 0
    except Exception as e:
        logger.warning(f"Détection OAK-D échouée: {e}")
        return False

USE_OAKD = _detecter_oakd()
logger.info(f"CAPTEUR: {'OAK-D Lite' if USE_OAKD else 'Webcam'}")
logger.info(f"GEM: {GEM['identifiant']} — genre:{GEM['genre']} — réactivité:{REACTIVITE}")


# ─────────────────────────────────────────────────────────────
#  UTILITAIRES
# ─────────────────────────────────────────────────────────────
def safe_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalise un tableau en [0,1] avec gestion des divisions par zéro."""
    if arr is None or arr.size == 0:
        return np.array([], dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def safe_median(values: np.ndarray) -> float:
    """Retourne la médiane d'un tableau, 0 si vide."""
    if values is None or values.size == 0:
        return 0.0
    return float(np.median(values))


# ─────────────────────────────────────────────────────────────
#  UTILITAIRES DEPTH
# ─────────────────────────────────────────────────────────────
def distance_mediane_centrale(depth_map: np.ndarray) -> float:
    """Distance médiane dans la zone centrale."""
    if depth_map is None or depth_map.size == 0:
        return 0.0
    
    try:
        h, w = depth_map.shape
        dy = max(1, int(h * AF_ZONE_FRAC / 2))
        dx = max(1, int(w * AF_ZONE_FRAC / 2))
        cy, cx = h // 2, w // 2
        zone = depth_map[cy-dy:cy+dy, cx-dx:cx+dx]
        valides = zone[(zone > PROFONDEUR_MIN_MM) & (zone < PROFONDEUR_MAX_MM)]
        return safe_median(valides)
    except Exception as e:
        logger.warning(f"distance_mediane_centrale: {e}")
        return 0.0


def distance_vers_lenspos(dist_mm: float) -> int:
    if dist_mm <= 0:
        return 120
    try:
        lp = int(1500 / dist_mm * 30)
        return int(np.clip(lp, LENSPOS_MIN, LENSPOS_MAX))
    except Exception:
        return 120


# ─────────────────────────────────────────────────────────────
#  CARTE DE SAILLANCE
# ─────────────────────────────────────────────────────────────
def carte_saillance_brute(frame_small: np.ndarray) -> np.ndarray:
    """Saillance brute sur image basse résolution."""
    if frame_small is None or frame_small.size == 0:
        return np.array([[]], dtype=np.float32)
    
    try:
        if len(frame_small.shape) == 3:
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_small.copy()
        
        gray_f = gray.astype(np.float32)
        lap = cv2.Laplacian(gray_f, cv2.CV_32F)
        grad = np.abs(lap)
        
        blur = cv2.GaussianBlur(gray_f, (15, 15), 0)
        ecart = np.abs(gray_f - blur)
        
        carte = 0.5 * grad + 0.5 * ecart
        return safe_normalize(carte)
    except Exception as e:
        logger.error(f"carte_saillance_brute: {e}")
        return np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)


def carte_saillance_couleur(frame_small: np.ndarray) -> np.ndarray:
    """Carte de saillance couleur filtrée par le GEM."""
    if frame_small is None or frame_small.size == 0 or len(frame_small.shape) != 3:
        return np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)
    
    try:
        h, w = frame_small.shape[:2]
        bonus = np.zeros((h, w), dtype=np.float32)
        frame_f = frame_small.astype(np.float32)
        
        for couleur_ref, poids in [
            (np.array(COULEUR_PRINCIPALE, dtype=np.float32), 1.0),
            (np.array(COULEUR_SECONDAIRE, dtype=np.float32), 0.6),
        ]:
            diff = frame_f - couleur_ref
            dist = np.sqrt(np.sum(diff**2, axis=2)) / (math.sqrt(3) * 255 + 1e-6)
            proximite = np.clip(1.0 - dist, 0, 1)
            
            tolerance = 0.3 + GEM["grace"] * 0.4
            proximite = np.where(proximite > (1 - tolerance), proximite, 0)
            bonus += proximite * poids
        
        intensite_gem = GEM["couleur"]["intensite"]
        bonus *= intensite_gem * REACTIVITE
        return np.clip(bonus, 0, 1).astype(np.float32)
    except Exception as e:
        logger.error(f"carte_saillance_couleur: {e}")
        return np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)


def carte_saillance_gem(frame_small: np.ndarray) -> np.ndarray:
    """Carte de saillance finale = brute + bonus couleur GEM."""
    if frame_small is None or frame_small.size == 0:
        return np.array([[]], dtype=np.float32)
    
    try:
        brute = carte_saillance_brute(frame_small)
        couleur = carte_saillance_couleur(frame_small)
        
        if brute.size == 0 or couleur.size == 0:
            return np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)
        
        carte = brute + couleur * REACTIVITE
        return safe_normalize(carte)
    except Exception as e:
        logger.error(f"carte_saillance_gem: {e}")
        return np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  CARTE MOUVEMENT
# ─────────────────────────────────────────────────────────────
def carte_mouvement(frame_curr: np.ndarray,
                    frame_prev: Optional[np.ndarray]) -> np.ndarray:
    """Carte de mouvement entre deux frames [0,1]."""
    if frame_prev is None or frame_curr is None:
        h, w = (frame_curr.shape[0] if frame_curr is not None else H,
                frame_curr.shape[1] if frame_curr is not None else W)
        return np.zeros((max(1, h // DIV_RETINE), max(1, w // DIV_RETINE)), dtype=np.float32)
    
    try:
        h, w = frame_curr.shape[:2]
        sw, sh = max(1, w // DIV_RETINE), max(1, h // DIV_RETINE)
        
        curr_small = cv2.resize(frame_curr, (sw, sh), interpolation=cv2.INTER_AREA)
        prev_small = cv2.resize(frame_prev, (sw, sh), interpolation=cv2.INTER_AREA)
        
        curr_g = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev_g = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        diff = np.abs(curr_g - prev_g)
        return safe_normalize(diff)
    except Exception as e:
        logger.error(f"carte_mouvement: {e}")
        return np.zeros((max(1, H // DIV_RETINE), max(1, W // DIV_RETINE)), dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  STATISTIQUES SCENE
# ─────────────────────────────────────────────────────────────
def stats_scene(frame_small: np.ndarray) -> Dict[str, float]:
    """Calcule teinte et luminosité moyennes de la scène."""
    if frame_small is None or frame_small.size == 0 or len(frame_small.shape) != 3:
        return {"lum_moy": 128.0, "lum_std": 50.0, "sat_moy": 100.0}
    
    try:
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV).astype(np.float32)
        lum_moy = float(np.mean(hsv[:,:,2]))
        lum_std = float(np.std(hsv[:,:,2]))
        sat_moy = float(np.mean(hsv[:,:,1]))
        return {"lum_moy": lum_moy, "lum_std": max(lum_std, 1.0), "sat_moy": sat_moy}
    except Exception as e:
        logger.error(f"stats_scene: {e}")
        return {"lum_moy": 128.0, "lum_std": 50.0, "sat_moy": 100.0}


def carte_contraste_scene(frame_small: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Carte de contraste local par rapport à la scène globale."""
    if frame_small is None or frame_small.size == 0 or len(frame_small.shape) != 3:
        return np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)
    
    try:
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV).astype(np.float32)
        lum = hsv[:,:,2]
        sat = hsv[:,:,1]
        
        lum_std = max(stats.get("lum_std", 1.0), 1.0)
        ecart_lum = np.abs(lum - stats.get("lum_moy", 128.0)) / lum_std
        ecart_lum = np.clip(ecart_lum / 3.0, 0, 1)
        
        ecart_sat = np.clip(sat / 255.0, 0, 1)
        
        carte = 0.7 * ecart_lum + 0.3 * ecart_sat
        return safe_normalize(carte)
    except Exception as e:
        logger.error(f"carte_contraste_scene: {e}")
        return np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)


def carte_intersections(frame_small: np.ndarray) -> np.ndarray:
    """Détecte les intersections de traits (coins Harris)."""
    if frame_small is None or frame_small.size == 0:
        return np.array([[]], dtype=np.float32)
    
    try:
        if len(frame_small.shape) == 3:
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame_small.astype(np.float32)
        
        corners = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.04)
        corners = np.clip(corners, 0, None)
        carte = safe_normalize(corners)
        return (carte * GEM["preferences"]["esthetique"]).astype(np.float32)
    except Exception as e:
        logger.error(f"carte_intersections: {e}")
        return np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  PROFONDEUR ET COHERENCE
# ─────────────────────────────────────────────────────────────
def z_ref_poi(cx: int, cy: int, depth_map: Optional[np.ndarray]) -> Optional[float]:
    """Retourne la profondeur au point (cx,cy), None si invalide."""
    if depth_map is None:
        return None
    try:
        h, w = depth_map.shape
        x = min(max(cx, 0), w - 1)
        y = min(max(cy, 0), h - 1)
        z = float(depth_map[y, x])
        return z if PROFONDEUR_MIN_MM < z < PROFONDEUR_MAX_MM else None
    except Exception as e:
        logger.warning(f"z_ref_poi: {e}")
        return None


def coherence_z_poi(cx: int, cy: int, depth_map: Optional[np.ndarray],
                    rayon: int = COHERENCE_RAYON_PX) -> float:
    """Score de cohérence Z autour d'un point (0=incohérent, 1=très cohérent)."""
    if depth_map is None:
        return 0.5
    
    try:
        h, w = depth_map.shape
        x1 = max(0, cx - rayon)
        y1 = max(0, cy - rayon)
        x2 = min(w, cx + rayon)
        y2 = min(h, cy + rayon)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        zone = depth_map[y1:y2, x1:x2].astype(np.float32)
        valide = zone[(zone > PROFONDEUR_MIN_MM) & (zone < PROFONDEUR_MAX_MM)]
        
        if valide.size < 4:
            return 0.0
        
        std_z = float(np.std(valide))
        moy_z = float(np.mean(valide))
        
        if moy_z < 1e-6:
            return 0.0
        
        cv = std_z / moy_z
        coherence = max(0.0, 1.0 - cv * 5)
        return min(1.0, float(coherence))
    except Exception as e:
        logger.warning(f"coherence_z_poi: {e}")
        return 0.5


def poids_z(z_ref: Optional[float], coherence: float,
            depth_map: Optional[np.ndarray]) -> float:
    """Poids prioritaire lié à la profondeur."""
    if z_ref is None or depth_map is None:
        return 0.3
    
    try:
        prox = max(0.0, 1.0 - (z_ref - PROFONDEUR_MIN_MM) / (PROFONDEUR_MAX_MM - PROFONDEUR_MIN_MM))
        return float(prox * coherence)
    except Exception:
        return 0.3


# ─────────────────────────────────────────────────────────────
#  eFOVEA
# ─────────────────────────────────────────────────────────────
def efovea_bbox(poi: Tuple[int, int], frame_h: int, frame_w: int,
                depth_map: Optional[np.ndarray]) -> Tuple[int, int, int, int]:
    """À partir du POI, calcule la bounding box de la zone d'attention."""
    cx, cy = poi
    
    # Valeurs par défaut (20% de la frame)
    default_bw = int(frame_w * 0.20)
    default_bh = int(frame_h * 0.20)
    default_bbox = (
        max(0, cx - default_bw // 2),
        max(0, cy - default_bh // 2),
        min(frame_w, cx + default_bw // 2),
        min(frame_h, cy + default_bh // 2)
    )
    
    if depth_map is None:
        return default_bbox
    
    try:
        h, w = depth_map.shape
        x = min(max(cx, 0), w - 1)
        y = min(max(cy, 0), h - 1)
        z_ref = float(depth_map[y, x])
        
        if z_ref < PROFONDEUR_MIN_MM or z_ref > PROFONDEUR_MAX_MM:
            return default_bbox
        
        z_tol = z_ref * 0.15
        z_min = max(PROFONDEUR_MIN_MM, z_ref - z_tol)
        z_max = min(PROFONDEUR_MAX_MM, z_ref + z_tol)
        
        masque_z = ((depth_map >= z_min) & (depth_map <= z_max) & (depth_map > 0)).astype(np.uint8) * 255
        
        if not np.any(masque_z):
            return default_bbox
        
        masque_fill = masque_z.copy()
        seed = (min(cx, masque_fill.shape[1]-1), min(cy, masque_fill.shape[0]-1))
        mask_flood = np.zeros((masque_fill.shape[0]+2, masque_fill.shape[1]+2), dtype=np.uint8)
        cv2.floodFill(masque_fill, mask_flood, seed, 128,
                      loDiff=0, upDiff=0,
                      flags=cv2.FLOODFILL_MASK_ONLY | (128 << 8))
        
        region = (mask_flood[1:-1, 1:-1] == 128)
        
        if not region.any():
            return default_bbox
        
        ys, xs = np.where(region)
        marge_x = max(5, int((xs.max() - xs.min()) * EFOVEA_MARGE_FRAC))
        marge_y = max(5, int((ys.max() - ys.min()) * EFOVEA_MARGE_FRAC))
        
        x1 = max(0, xs.min() - marge_x)
        y1 = max(0, ys.min() - marge_y)
        x2 = min(frame_w, xs.max() + marge_x)
        y2 = min(frame_h, ys.max() + marge_y)
        
        return (x1, y1, x2, y2)
    except Exception as e:
        logger.warning(f"efovea_bbox: {e}")
        return default_bbox


# ─────────────────────────────────────────────────────────────
#  ASSAINISSEMENT ROI
# ─────────────────────────────────────────────────────────────
def assainir_roi(roi_bgr: np.ndarray,
                 roi_depth: Optional[np.ndarray],
                 sigma_seuil: float = 2.0) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    """Supprime les pixels en décrochage Z dans une ROI clippée."""
    h, w = roi_bgr.shape[:2]
    
    if roi_depth is None:
        masque = np.ones((h, w), dtype=np.uint8) * 255
        return roi_bgr.copy(), masque, None, None
    
    try:
        if roi_depth.shape != (h, w):
            roi_depth = cv2.resize(
                roi_depth.astype(np.float32), (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint16)
        
        depth_f = roi_depth.astype(np.float32)
        valides = depth_f[(depth_f > PROFONDEUR_MIN_MM) & (depth_f < PROFONDEUR_MAX_MM)]
        
        if valides.size < 10:
            masque = np.ones((h, w), dtype=np.uint8) * 255
            return roi_bgr.copy(), masque, None, None
        
        z_ref = float(np.median(valides))
        sigma_z = float(np.std(valides))
        
        seuil_z = max(50.0, sigma_seuil * sigma_z)
        
        masque_bool = (
            (depth_f > PROFONDEUR_MIN_MM) &
            (depth_f < PROFONDEUR_MAX_MM) &
            (np.abs(depth_f - z_ref) <= seuil_z)
        )
        masque = (masque_bool * 255).astype(np.uint8)
        
        roi_assainie = roi_bgr.copy()
        roi_assainie[~masque_bool] = 0
        
        return roi_assainie, masque, z_ref, sigma_z
    except Exception as e:
        logger.warning(f"assainir_roi: {e}")
        masque = np.ones((h, w), dtype=np.uint8) * 255
        return roi_bgr.copy(), masque, None, None


# ─────────────────────────────────────────────────────────────
#  DÉTECTION DES PLANS ET STRUCTURES
# ─────────────────────────────────────────────────────────────
def detecter_plans_principaux(depth_map: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """Détecte les plans principaux de la scène (sol, plafond)."""
    if depth_map is None:
        return {"sol": None, "plafond": None}
    
    try:
        valides = (depth_map > PROFONDEUR_MIN_MM) & (depth_map < PROFONDEUR_MAX_MM)
        
        if not valides.any():
            return {"sol": None, "plafond": None}
        
        profondeurs = depth_map[valides]
        hist, bins = np.histogram(profondeurs, bins=50)
        
        pics = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > hist.max() * 0.3:
                pics.append((bins[i] + bins[i+1]) / 2)
        
        pics.sort()
        
        sol = pics[0] if pics else None
        plafond = pics[-1] if pics else None
        
        return {"sol": sol, "plafond": plafond}
    except Exception as e:
        logger.warning(f"detecter_plans_principaux: {e}")
        return {"sol": None, "plafond": None}


def detecter_lignes_verticales(depth_map: Optional[np.ndarray],
                                sol: Optional[float],
                                plafond: Optional[float],
                                tolerance_z: float = LIGNE_ANCRAGE_TOLERANCE_Z_MM) -> List[Dict]:
    """Détecte les lignes verticales ancrées au sol ou au plafond."""
    if depth_map is None or (sol is None and plafond is None):
        return []
    
    try:
        h, w = depth_map.shape
        grad_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        
        seuil_grad = np.percentile(np.abs(grad_x), 90) if grad_x.size > 0 else 100
        
        lignes = []
        
        for x in range(0, w, 5):
            colonne = depth_map[:, x]
            valides = (colonne > PROFONDEUR_MIN_MM) & (colonne < PROFONDEUR_MAX_MM)
            if not valides.any():
                continue
            
            diff = np.diff(colonne.astype(np.float32))
            seuil_diff = np.percentile(np.abs(diff[valides[:-1]]), 95) if valides[:-1].any() else 100
            
            for y in range(1, h-1):
                if abs(diff[y]) > seuil_diff:
                    z_point = colonne[y]
                    
                    if sol is not None and abs(z_point - sol) < tolerance_z:
                        lignes.append({
                            "type": "sol",
                            "position": (x, y),
                            "z": z_point,
                            "hauteur": y
                        })
                    elif plafond is not None and abs(z_point - plafond) < tolerance_z:
                        lignes.append({
                            "type": "plafond",
                            "position": (x, y),
                            "z": z_point,
                            "hauteur": h - y
                        })
        
        return lignes
    except Exception as e:
        logger.warning(f"detecter_lignes_verticales: {e}")
        return []


def detecter_lignes_horizontales(depth_map: Optional[np.ndarray],
                                  sol: Optional[float],
                                  plafond: Optional[float],
                                  tolerance_z: float = LIGNE_ANCRAGE_TOLERANCE_Z_MM) -> List[Dict]:
    """Détecte les lignes horizontales proches du sol ou du plafond."""
    if depth_map is None or (sol is None and plafond is None):
        return []
    
    try:
        h, w = depth_map.shape
        lignes = []
        
        for y in range(0, h, 5):
            ligne = depth_map[y, :]
            valides = (ligne > PROFONDEUR_MIN_MM) & (ligne < PROFONDEUR_MAX_MM)
            if not valides.any():
                continue
            
            z_ligne = np.median(ligne[valides])
            
            if sol is not None and abs(z_ligne - sol) < tolerance_z:
                segments = []
                debut = 0
                for x in range(w):
                    if valides[x] and (x == 0 or not valides[x-1]):
                        debut = x
                    if valides[x] and (x == w-1 or not valides[x+1]):
                        if x - debut > 30:
                            segments.append((debut, x))
                
                for x1, x2 in segments:
                    lignes.append({
                        "type": "sol",
                        "position": (x1, y),
                        "largeur": x2 - x1,
                        "z": z_ligne
                    })
            
            elif plafond is not None and abs(z_ligne - plafond) < tolerance_z:
                segments = []
                debut = 0
                for x in range(w):
                    if valides[x] and (x == 0 or not valides[x-1]):
                        debut = x
                    if valides[x] and (x == w-1 or not valides[x+1]):
                        if x - debut > 30:
                            segments.append((debut, x))
                
                for x1, x2 in segments:
                    lignes.append({
                        "type": "plafond",
                        "position": (x1, y),
                        "largeur": x2 - x1,
                        "z": z_ligne
                    })
        
        return lignes
    except Exception as e:
        logger.warning(f"detecter_lignes_horizontales: {e}")
        return []


def detecter_intersections_3d(depth_map: Optional[np.ndarray],
                               sol: Optional[float],
                               plafond: Optional[float],
                               lignes_verticales: List[Dict],
                               lignes_horizontales: List[Dict]) -> List[Dict]:
    """Détecte les intersections 3D dans l'espace."""
    intersections = []
    
    try:
        for v in lignes_verticales:
            for h_l in lignes_horizontales:
                if v["type"] == h_l["type"]:
                    dist_x = abs(v["position"][0] - h_l["position"][0])
                    if dist_x < INTERSECTION_TOLERANCE_PX:
                        intersections.append({
                            "type": f"coin_{v['type']}",
                            "position": (h_l["position"][0], v["position"][1]),
                            "z": v["z"],
                            "confiance": 0.8
                        })
    except Exception as e:
        logger.warning(f"detecter_intersections_3d: {e}")
    
    return intersections


# ─────────────────────────────────────────────────────────────
#  DÉTECTION MULTI-POIs
# ─────────────────────────────────────────────────────────────
def detecter_pois(frame: np.ndarray,
                  frame_prev: Optional[np.ndarray],
                  depth_map: Optional[np.ndarray],
                  n_max: int = 6) -> List[Dict]:
    """Détecte jusqu'à n_max POIs avec poids multi-critères."""
    if frame is None or frame.size == 0:
        return []
    
    try:
        h, w = frame.shape[:2]
        sw, sh = max(1, w // DIV_RETINE), max(1, h // DIV_RETINE)
        sx, sy = w / sw, h / sh
        
        frame_small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        scene_stats = stats_scene(frame_small)
        
        # Cartes composantes
        carte_sal = carte_saillance_gem(frame_small)
        carte_mov = carte_mouvement(frame, frame_prev)
        carte_ctr = carte_contraste_scene(frame_small, scene_stats)
        carte_inter = carte_intersections(frame_small)
        
        # Vérifier les dimensions
        if carte_sal.size == 0 or carte_mov.size == 0:
            return []
        
        # Carte proximité Z
        carte_prox = np.zeros((sh, sw), dtype=np.float32)
        if depth_map is not None:
            try:
                depth_small = cv2.resize(
                    depth_map.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA
                )
                valide = (depth_small > PROFONDEUR_MIN_MM) & (depth_small < PROFONDEUR_MAX_MM)
                if valide.any():
                    d_min = depth_small[valide].min()
                    d_max = depth_small[valide].max()
                    if d_max > d_min + 1e-6:
                        carte_prox[valide] = 1.0 - (depth_small[valide] - d_min) / (d_max - d_min)
            except Exception as e:
                logger.warning(f"carte_prox: {e}")
        
        # Fusion
        carte_poids = (
            carte_sal * 1.0 +
            carte_mov * REACTIVITE +
            carte_ctr * 0.6 +
            carte_inter * 0.4 +
            carte_prox * 0.5
        )
        carte_poids = safe_normalize(carte_poids)
        
        # Z comme multiplicateur prioritaire
        if depth_map is not None:
            try:
                depth_small_n = cv2.resize(
                    depth_map.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA
                )
                valide = (depth_small_n > PROFONDEUR_MIN_MM) & (depth_small_n < PROFONDEUR_MAX_MM)
                z_mult = np.ones((sh, sw), dtype=np.float32) * 0.5
                if valide.any():
                    d_min = depth_small_n[valide].min()
                    d_max = depth_small_n[valide].max()
                    if d_max > d_min + 1e-6:
                        z_mult[valide] = 0.5 + 0.5 * (1.0 - (depth_small_n[valide] - d_min) / (d_max - d_min))
                carte_poids = carte_poids * z_mult
                carte_poids = safe_normalize(carte_poids)
            except Exception as e:
                logger.warning(f"z_mult: {e}")
        
        carte_lissee = cv2.GaussianBlur(carte_poids, (7, 7), 0)
        
        pois = []
        
        # Centre de l'image — candidat systématique
        cx_centre, cy_centre = w // 2, h // 2
        z_centre = z_ref_poi(cx_centre, cy_centre, depth_map)
        coh_centre = coherence_z_poi(cx_centre, cy_centre, depth_map)
        
        if coh_centre > 0.4:
            pz_centre = poids_z(z_centre, coh_centre, depth_map)
            cx_s = int(min(max(cx_centre / sx, 0), sw - 1))
            cy_s = int(min(max(cy_centre / sy, 0), sh - 1))
            score_centre = float(carte_lissee[cy_s, cx_s])
            bbox_centre = efovea_bbox((cx_centre, cy_centre), h, w, depth_map)
            pois.append({
                "poi": (cx_centre, cy_centre),
                "poids": round(score_centre * (0.5 + 0.5 * pz_centre), 3),
                "saillance": 0.0,
                "mouvement": 0.0,
                "contraste": 0.0,
                "intersection": 0.0,
                "proximite": round(pz_centre, 3),
                "poids_z": round(pz_centre, 3),
                "coherence_z": round(coh_centre, 3),
                "z_ref": z_centre,
                "source": "centre",
                "bbox": bbox_centre
            })
        
        # Extraction des maxima locaux
        rayon_sup = max(1, min(sw, sh) // (n_max * 2))
        seuil = INTENSITE_SEUIL * max(0.1, 1.0 - REACTIVITE * 0.3)
        carte_travail = carte_lissee.copy()
        
        for _ in range(n_max):
            max_val = float(carte_travail.max())
            if max_val < seuil:
                break
            
            _, _, _, max_loc = cv2.minMaxLoc(carte_travail)
            cx_s, cy_s = max_loc
            cx = int(min(max(cx_s * sx, 0), w - 1))
            cy = int(min(max(cy_s * sy, 0), h - 1))
            
            z_r = z_ref_poi(cx, cy, depth_map)
            coh_z = coherence_z_poi(cx, cy, depth_map)
            pz = poids_z(z_r, coh_z, depth_map)
            
            sal_val = float(carte_sal[cy_s, cx_s]) if carte_sal.size > 0 else 0.0
            mov_val = float(carte_mov[cy_s, cx_s]) if carte_mov.size > 0 else 0.0
            ctr_val = float(carte_ctr[cy_s, cx_s]) if carte_ctr.size > 0 else 0.0
            int_val = float(carte_inter[cy_s, cx_s]) if carte_inter.size > 0 else 0.0
            prox_val = float(carte_prox[cy_s, cx_s]) if carte_prox.size > 0 else 0.0
            
            poids_final = max_val * (0.5 + 0.5 * pz)
            
            bbox = efovea_bbox((cx, cy), h, w, depth_map)
            pois.append({
                "poi": (cx, cy),
                "poids": round(poids_final, 3),
                "saillance": round(sal_val, 3),
                "mouvement": round(mov_val, 3),
                "contraste": round(ctr_val, 3),
                "intersection": round(int_val, 3),
                "proximite": round(prox_val, 3),
                "poids_z": round(pz, 3),
                "coherence_z": round(coh_z, 3),
                "z_ref": z_r,
                "source": "saillance",
                "bbox": bbox
            })
            
            # Supprimer la zone autour du maximum
            y_start = max(0, cy_s - rayon_sup)
            y_end = min(sh, cy_s + rayon_sup + 1)
            x_start = max(0, cx_s - rayon_sup)
            x_end = min(sw, cx_s + rayon_sup + 1)
            carte_travail[y_start:y_end, x_start:x_end] = 0
        
        # Tri final par poids décroissant
        pois.sort(key=lambda p: p["poids"], reverse=True)
        return pois[:n_max]
        
    except Exception as e:
        logger.error(f"detecter_pois: {e}")
        return []


# ─────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────
COULEUR_POI = (255, 220, 0)
COULEUR_CHEMIN = (255, 220, 0)
COULEUR_MIRE = (0, 0, 220)
COULEUR_BBOX = (0, 220, 255)
COULEUR_HEATMAP_ALPHA = 0.4
TAILLE_MIRE_MAX = 28
TAILLE_MIRE_MIN = 8


def dessiner_mire_poi(canvas: np.ndarray, cx: int, cy: int,
                       taille: int, couleur: Tuple[int, int, int], epaisseur: int = 1):
    try:
        cv2.line(canvas, (cx-taille, cy), (cx+taille, cy), couleur, epaisseur, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy-taille), (cx, cy+taille), couleur, epaisseur, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), max(3, taille//4), couleur, epaisseur, cv2.LINE_AA)
    except Exception as e:
        logger.warning(f"dessiner_mire_poi: {e}")


def dessiner_mire_centrale(canvas: np.ndarray, cx: int, cy: int, taille: int = 20):
    try:
        cv2.line(canvas, (cx - taille, cy), (cx + taille, cy), COULEUR_MIRE, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - taille), (cx, cy + taille), COULEUR_MIRE, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 6, COULEUR_MIRE, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 2, COULEUR_MIRE, -1)
    except Exception as e:
        logger.warning(f"dessiner_mire_centrale: {e}")


def dessiner_efovea(canvas: np.ndarray, poi: Tuple[int, int], bbox: Tuple[int, int, int, int],
                    saillance_val: float, z_ref: Optional[float] = None):
    try:
        cx, cy = poi
        x1, y1, x2, y2 = bbox
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COULEUR_BBOX, 2, cv2.LINE_AA)
        
        coin = 12
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(canvas, (px, py), (px + dx*coin, py), COULEUR_BBOX, 2)
            cv2.line(canvas, (px, py), (px, py + dy*coin), COULEUR_BBOX, 2)
        
        dessiner_mire_centrale(canvas, cx, cy)
        
        label = f"POI s:{saillance_val:.2f}"
        if z_ref and z_ref > 0:
            label += f" z:{z_ref:.0f}mm"
        cv2.putText(canvas, label, (x1, max(y1-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COULEUR_BBOX, 1)
    except Exception as e:
        logger.warning(f"dessiner_efovea: {e}")


def dessiner_chemin_attentionnel(canvas: np.ndarray, pois: List[Dict]):
    if not pois:
        return
    
    try:
        poids_max = pois[0]["poids"]
        poids_min = pois[-1]["poids"] if len(pois) > 1 else poids_max
        
        def taille_mire(poids):
            if poids_max == poids_min:
                return TAILLE_MIRE_MAX
            ratio = (poids - poids_min) / max(poids_max - poids_min, 1e-6)
            return int(TAILLE_MIRE_MIN + ratio * (TAILLE_MIRE_MAX - TAILLE_MIRE_MIN))
        
        # Trait en tirets entre POIs
        for i in range(len(pois) - 1):
            p1 = pois[i]["poi"]
            p2 = pois[i+1]["poi"]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1:
                continue
            n_tirets = max(2, int(dist / 18))
            long_tiret = 0.4
            for j in range(n_tirets):
                t0 = j / n_tirets
                t1 = t0 + long_tiret / n_tirets
                x0 = int(p1[0] + dx * t0)
                y0 = int(p1[1] + dy * t0)
                x1 = int(p1[0] + dx * t1)
                y1 = int(p1[1] + dy * t1)
                cv2.line(canvas, (x0, y0), (x1, y1), COULEUR_CHEMIN, 2, cv2.LINE_AA)
        
        # Mires POIs (ordre inverse pour que le principal soit au-dessus)
        for i, poi_data in enumerate(reversed(pois)):
            cx, cy = poi_data["poi"]
            poids = poi_data["poids"]
            taille = taille_mire(poids)
            
            if i == len(pois) - 1:
                dessiner_efovea(canvas, (cx, cy), poi_data["bbox"],
                                poi_data["saillance"], poi_data["z_ref"])
            else:
                dessiner_mire_poi(canvas, cx, cy, taille, COULEUR_POI, epaisseur=1)
                cv2.putText(canvas, f"{poids:.2f}",
                            (cx + taille + 3, cy - taille),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COULEUR_POI, 1)
    except Exception as e:
        logger.warning(f"dessiner_chemin_attentionnel: {e}")


def dessiner_lignes_structures(canvas: np.ndarray,
                                lignes_verticales: List[Dict],
                                lignes_horizontales: List[Dict],
                                intersections: List[Dict]):
    try:
        for lv in lignes_verticales:
            x, y = lv["position"]
            couleur = (0, 255, 255) if lv["type"] == "sol" else (255, 255, 0)
            cv2.line(canvas, (x, y), (x, y - min(y, 100)), couleur, 1)
        
        for lh in lignes_horizontales:
            x, y = lh["position"]
            couleur = (0, 255, 255) if lh["type"] == "sol" else (255, 255, 0)
            cv2.line(canvas, (x, y), (x + lh["largeur"], y), couleur, 1)
        
        for inter in intersections:
            x, y = inter["position"]
            couleur = {
                "coin_sol": (0, 165, 255),
                "coin_plafond": (255, 165, 0)
            }.get(inter["type"], (200, 200, 200))
            
            taille = 8
            cv2.line(canvas, (x-taille, y-taille), (x+taille, y+taille), couleur, 2)
            cv2.line(canvas, (x+taille, y-taille), (x-taille, y+taille), couleur, 2)
    except Exception as e:
        logger.warning(f"dessiner_lignes_structures: {e}")


def afficher_heatmap(frame: np.ndarray, carte: np.ndarray) -> np.ndarray:
    try:
        carte_u8 = (carte * 255).astype(np.uint8)
        carte_big = cv2.resize(carte_u8, (frame.shape[1], frame.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(carte_big, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - COULEUR_HEATMAP_ALPHA,
                               heatmap, COULEUR_HEATMAP_ALPHA, 0)
    except Exception as e:
        logger.warning(f"afficher_heatmap: {e}")
        return frame.copy()


# ─────────────────────────────────────────────────────────────
#  PIPELINE OAK-D
# ─────────────────────────────────────────────────────────────
def creer_pipeline():
    pipeline = dai.Pipeline()
    
    try:
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setPreviewSize(W, H)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(FPS_ACQ)
        cam_rgb.initialControl.setManualFocus(120)
        
        mono_l = pipeline.create(dai.node.MonoCamera)
        mono_r = pipeline.create(dai.node.MonoCamera)
        mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono_l.setFps(FPS_ACQ)
        mono_r.setFps(FPS_ACQ)
        
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(W, H)
        stereo.setSubpixel(True)
        
        config = stereo.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 60
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.temporalFilter.alpha = 0.4
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.thresholdFilter.minRange = PROFONDEUR_MIN_MM
        config.postProcessing.thresholdFilter.maxRange = PROFONDEUR_MAX_MM
        stereo.initialConfig.set(config)
        
        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)
        
        ctrl_in = pipeline.create(dai.node.XLinkIn)
        ctrl_in.setStreamName("control")
        ctrl_in.out.link(cam_rgb.inputControl)
        
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_depth.setStreamName("depth")
        cam_rgb.preview.link(xout_rgb.input)
        stereo.depth.link(xout_depth.input)
        
        return pipeline
    except Exception as e:
        logger.error(f"creer_pipeline: {e}")
        raise


# ─────────────────────────────────────────────────────────────
#  THREADS ACQUISITION
# ─────────────────────────────────────────────────────────────
def thread_acquisition_oak(device, frame_queue, stop_event):
    try:
        q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=1, blocking=False)
        q_ctrl = device.getInputQueue("control")
        
        dernier_af = datetime.now()
        lenspos_actuel = 120
        lenspos_cible = 120
        EMA_ALPHA = 0.25 if IS_PI else 0.35
        depth_ema = None
        
        while not stop_event.is_set():
            try:
                in_rgb = q_rgb.tryGet()
                in_depth = q_depth.tryGet()
                if in_rgb is None or in_depth is None:
                    continue
                
                frame_bgr = in_rgb.getCvFrame()
                depth_raw = in_depth.getFrame()
                ts = datetime.now()
                
                lenspos_reel = in_rgb.getLensPosition()
                if lenspos_reel != -1 and abs(lenspos_reel - lenspos_cible) > 2:
                    continue
                
                if depth_ema is None:
                    depth_ema = depth_raw.astype(np.float32)
                else:
                    mask = depth_raw > 0
                    depth_ema[mask] = (EMA_ALPHA * depth_raw[mask].astype(np.float32)
                                       + (1 - EMA_ALPHA) * depth_ema[mask])
                depth_final = np.clip(depth_ema, 0, 65535).astype(np.uint16)
                
                if (ts - dernier_af).total_seconds() >= AF_PERIODE_S:
                    dist_mm = distance_mediane_centrale(depth_final)
                    nouveau_lp = distance_vers_lenspos(dist_mm)
                    if abs(nouveau_lp - lenspos_actuel) > 5:
                        ctrl = dai.CameraControl()
                        ctrl.setManualFocus(nouveau_lp)
                        q_ctrl.send(ctrl)
                        lenspos_actuel = nouveau_lp
                        lenspos_cible = nouveau_lp
                        depth_ema = None
                    dernier_af = ts
                
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put((frame_bgr, depth_final, ts))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"thread_acquisition_oak boucle: {e}")
                
    except Exception as e:
        logger.error(f"thread_acquisition_oak: {e}")


def thread_acquisition_webcam(frame_queue, stop_event):
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, FPS_ACQ)
        
        if not cap.isOpened():
            logger.error("Webcam inaccessible")
            stop_event.set()
            return
        
        logger.info(f"Webcam ouverte — {W}x{H} @ {FPS_ACQ}fps")
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            
            fh, fw = frame.shape[:2]
            if fh >= 400:
                debut_y = (fh - 400) // 2
                frame = frame[debut_y:debut_y+400, :]
            else:
                frame = cv2.resize(frame, (W, H))
            
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put((frame, None, datetime.now()))
        
        cap.release()
    except Exception as e:
        logger.error(f"thread_acquisition_webcam: {e}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=QUEUE_MAX)
    stop_event = threading.Event()
    
    try:
        if USE_OAKD:
            pipeline = creer_pipeline()
            device = dai.Device(pipeline)
            t_acq = threading.Thread(
                target=thread_acquisition_oak,
                args=(device, frame_queue, stop_event),
                daemon=True
            )
        else:
            t_acq = threading.Thread(
                target=thread_acquisition_webcam,
                args=(frame_queue, stop_event),
                daemon=True
            )
        t_acq.start()
        
        mode_analyse = False
        derniere_frame = None
        avant_derniere = None
        derniere_depth = None
        resultats_analyse = None
        
        logger.info("=== Retina Lab — Shirka_001 ===")
        logger.info("ESPACE : analyser la frame courante")
        logger.info("H      : toggle heatmap de saillance")
        logger.info("Q      : quitter")
        
        show_heatmap = False
        
        while not stop_event.is_set():
            try:
                frame, depth, _ = frame_queue.get(timeout=0.05)
                avant_derniere = derniere_frame
                derniere_frame = frame
                derniere_depth = depth
            except queue.Empty:
                pass
            
            if mode_analyse and resultats_analyse is not None:
                affichage = resultats_analyse
            elif derniere_frame is not None:
                affichage = derniere_frame.copy()
                cv2.putText(affichage, "ESPACE:analyser  H:heatmap  Q:quitter",
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                continue
            
            cv2.imshow("Retina Lab — Shirka", affichage)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                stop_event.set()
                logger.info("Arrêt demandé par l'utilisateur")
            
            elif key == ord('h'):
                show_heatmap = not show_heatmap
                logger.info(f"Heatmap: {'ON' if show_heatmap else 'OFF'}")
            
            elif key == ord(' '):
                if not mode_analyse and derniere_frame is not None:
                    logger.info("Analyse rétine en cours...")
                    
                    try:
                        # Détection des POIs
                        pois = detecter_pois(
                            derniere_frame,
                            avant_derniere,
                            derniere_depth,
                            n_max=6
                        )
                        
                        # Détection des plans et structures
                        plans = detecter_plans_principaux(derniere_depth)
                        sol = plans["sol"]
                        plafond = plans["plafond"]
                        
                        lignes_verticales = []
                        lignes_horizontales = []
                        intersections = []
                        
                        if derniere_depth is not None:
                            lignes_verticales = detecter_lignes_verticales(
                                derniere_depth, sol, plafond
                            )
                            lignes_horizontales = detecter_lignes_horizontales(
                                derniere_depth, sol, plafond
                            )
                            intersections = detecter_intersections_3d(
                                derniere_depth, sol, plafond,
                                lignes_verticales, lignes_horizontales
                            )
                        
                        affichage = derniere_frame.copy()
                        
                        # Heatmap si activée
                        if show_heatmap:
                            frame_small = cv2.resize(
                                derniere_frame,
                                (max(1, W//DIV_RETINE), max(1, H//DIV_RETINE)),
                                interpolation=cv2.INTER_AREA
                            )
                            carte = carte_saillance_gem(frame_small)
                            affichage = afficher_heatmap(affichage, carte)
                        
                        if pois:
                            # Assainissement des ROIs
                            for p in pois:
                                x1, y1, x2, y2 = p["bbox"]
                                if x2 > x1 and y2 > y1:
                                    roi_bgr = derniere_frame[y1:y2, x1:x2]
                                    roi_depth = derniere_depth[y1:y2, x1:x2] if derniere_depth is not None else None
                                    roi_saine, masque, z_r, sigma_z = assainir_roi(roi_bgr, roi_depth)
                                    p["roi_assainie"] = roi_saine
                                    p["masque_z"] = masque
                                    p["sigma_z"] = round(sigma_z, 1) if sigma_z else None
                            
                            # Chemin attentionnel
                            dessiner_chemin_attentionnel(affichage, pois)
                            
                            # Lignes structurelles
                            dessiner_lignes_structures(affichage, lignes_verticales,
                                                       lignes_horizontales, intersections)
                            
                            logger.info(f"{len(pois)} POI(s) détecté(s)")
                            for i, p in enumerate(pois):
                                z_info = f" z:{p['z_ref']:.0f}mm" if p['z_ref'] else ""
                                logger.info(f"  #{i+1} [{p['source']}] {p['poi']}  poids:{p['poids']:.2f}{z_info}")
                            
                            logger.info(f"  Structures: {len(lignes_verticales)} lignes verticales, "
                                       f"{len(lignes_horizontales)} lignes horizontales, "
                                       f"{len(intersections)} intersections")
                        else:
                            logger.info("Aucun POI détecté (saillance insuffisante)")
                            cv2.putText(affichage, "Aucun POI",
                                        (W//2-40, H//2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,200), 2)
                        
                        # Fenêtre debug — ROI assainie du POI principal
                        if pois and pois[0].get("roi_assainie") is not None:
                            roi_vis = pois[0]["roi_assainie"]
                            if roi_vis.size > 0:
                                roi_vis = cv2.resize(roi_vis, (320, 200),
                                                     interpolation=cv2.INTER_NEAREST)
                                masque_vis = pois[0]["masque_z"]
                                roi_debug = roi_vis.copy()
                                if masque_vis is not None and masque_vis.size > 0:
                                    masque_vis_resized = cv2.resize(masque_vis, (320, 200),
                                                                     interpolation=cv2.INTER_NEAREST)
                                    roi_debug[masque_vis_resized == 0] = (0, 0, 80)
                                z_r = pois[0]["z_ref"]
                                sig_z = pois[0]["sigma_z"]
                                cv2.putText(roi_debug,
                                            f"ROI #1 assainie  z:{z_r:.0f}mm  σ:{sig_z}mm"
                                            if z_r else "ROI #1 (webcam — pas de Z)",
                                            (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.35, (0, 255, 180), 1)
                                cv2.imshow("ROI assainie — POI #1", roi_debug)
                        
                        # Fenêtre debug — carte de saillance
                        frame_small = cv2.resize(
                            derniere_frame,
                            (max(1, W//DIV_RETINE), max(1, H//DIV_RETINE)),
                            interpolation=cv2.INTER_AREA
                        )
                        carte_sal = carte_saillance_gem(frame_small)
                        carte_mov = carte_mouvement(derniere_frame, avant_derniere)
                        
                        if carte_sal.size > 0 and carte_mov.size > 0:
                            sal_u8 = cv2.resize((carte_sal*255).astype(np.uint8), (W//2, H))
                            mov_u8 = cv2.resize((carte_mov*255).astype(np.uint8), (W//2, H))
                            sal_col = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
                            mov_col = cv2.applyColorMap(mov_u8, cv2.COLORMAP_HOT)
                            debug_img = np.hstack([sal_col, mov_col])
                            cv2.putText(debug_img, "Saillance GEM",
                                        (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                            cv2.putText(debug_img, "Mouvement",
                                        (W//2+6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                            cv2.imshow("Debug — Saillance | Mouvement", debug_img)
                        
                        resultats_analyse = affichage
                        mode_analyse = True
                        
                    except Exception as e:
                        logger.error(f"Erreur pendant l'analyse: {e}")
                        mode_analyse = False
                
                else:
                    mode_analyse = False
                    resultats_analyse = None
                    for win in ["Debug — Saillance | Mouvement",
                                "ROI assainie — POI #1"]:
                        try:
                            cv2.destroyWindow(win)
                        except Exception:
                            pass
                    logger.info("Retour temps réel")
        
    except Exception as e:
        logger.error(f"Erreur dans le main: {e}")
    finally:
        if USE_OAKD:
            try:
                device.close()
            except:
                pass
        cv2.destroyAllWindows()
        logger.info("Arrêt.")