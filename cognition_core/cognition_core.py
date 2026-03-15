#!/usr/bin/env python3
"""
cognition_core.py — Cœur cognitif de l'entité synthétique transcendée
Version 10.0.0 - TAXONOMIQUE UNIFIÉE COMPLÈTE

Architecture unifiée :
- Graphe unique de concepts avec relations typées
- Hiérarchie de stockage : RAM → Disque → Archive
- Refroidissement horaire (oubli progressif)
- Cristallisation nocturne (consolidation)
- Types de mémoire différenciés (permanent, littéraire, épisodique, social)
- Chargement à la demande depuis fichiers thématiques
- Signatures vocales/visuelles pour identification
- Un seul moteur : texte → intent → texte selon paramètres

Utilisation :
    core = CognitionCore(data_path="./data", gem_path="./gem.json")
    core.load_knowledge_base("temps", "bases/temps.json")
    core.load_knowledge_base("animaux", "bases/animaux.json")
    core.start()
    
    # Texte → Intent
    intent = core.process("demain midi, je vais au restaurant", 
                          input_type="text", output_type="intent")
    
    # Intent → Texte
    texte = core.process(intent, 
                         input_type="intent", output_type="text")
"""

import json
import logging
import time
import uuid
import threading
import sqlite3
import gzip
import pickle
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
import hashlib

__version__ = "10.0.0"
logger = logging.getLogger("CognitionCore")

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "data_dir": Path("./data"),
    "bases_dir": Path("./bases"),
    "gem_file": Path("./gem.json"),
    "archive_dir": Path("./archives"),
    
    "max_ram_concepts": 10000,
    "cooling_interval_hours": 1,
    "cooling_rate_litteraire": 0.05,
    "cooling_rate_episodique": 0.03,
    "cooling_rate_social": 0.01,
    "freezing_threshold": 0.1,
    "disk_threshold": 0.3,
    "promotion_threshold": 10,  # accès pour promotion
    
    "nightly_hour": 2,
    "context_ttl_seconds": 300,
    "max_pending_fragments": 5,
    "min_confidence": 0.3,
    
    "sentence_cache_size": 100
}

# ============================================================
# ENUMS
# ============================================================

class IntentType(str, Enum):
    INFORMATION = "information"
    QUESTION = "question"
    REPONSE = "reponse"
    ACTION = "action"
    CLARIFICATION = "clarification"
    SOCIAL = "social"
    META = "meta"

class RelationType(str, Enum):
    # Hiérarchiques
    EST_UN = "est_un"
    A_POUR_INSTANCE = "a_pour_instance"
    A_POUR_SOUS_CATEGORIE = "a_pour_sous_categorie"
    
    # Temporelles
    PRECEDE = "precede"
    SUIT = "suit"
    CONTIENT = "contient"
    FAIT_PARTIE_DE = "fait_partie_de"
    COMMENCE = "commence"
    FINIT = "finit"
    DURE = "dure"
    
    # Lexicales
    SYNONYME = "synonyme"
    ANTONYME = "antonyme"
    TRADUCTION = "traduction"
    
    # Sociales
    EST_AMI_DE = "est_ami_de"
    EST_PARENT_DE = "est_parent_de"
    EST_CONNU_DE = "est_connu_de"
    A_POUR_SIGNATURE_VOCALE = "a_pour_signature_vocale"
    A_POUR_SIGNATURE_VISAGE = "a_pour_signature_visage"
    
    # Spatiales
    EST_SITUE_A = "est_situe_a"
    EST_DANS = "est_dans"
    
    # Attributives
    A_POUR_CARACTERISTIQUE = "a_pour_caracteristique"
    A_POUR_COULEUR = "a_pour_couleur"
    A_POUR_TAILLE = "a_pour_taile"
    
    # Littéraires
    A_POUR_PERSONNAGE = "a_pour_personnage"
    A_POUR_THEME = "a_pour_theme"
    A_ECRIT = "a_ecrit"
    EST_ECRIT_PAR = "est_ecrit_par"
    
    # Épistémiques
    CONFIRME_PAR = "confirme_par"
    INFERE_DE = "infere_de"
    CONFLIT_AVEC = "conflit_avec"

class SourceWeight(float, Enum):
    OBSERVATION = 1.0
    SELF = 0.95
    EDUCATIVE = 0.9
    SCIENTIFIC = 0.8
    REPORTED = 0.6
    FICTION = 0.3
    INTERNET = 0.2
    RUMOR = 0.1

class RegisterStyle(str, Enum):
    FAMILIER = "familier"
    NEUTRE = "neutre"
    SOUTENU = "soutenu"
    AFFECTIF = "affectif"
    TECHNIQUE = "technique"

class ConceptNature(str, Enum):
    CATEGORIE = "categorie"
    INSTANCE = "instance"
    PROPRIETE = "propriete"
    RELATION = "relation"
    INTERVALLE = "intervalle"
    POINT = "point"
    PERSONNE = "personne"
    LIEU = "lieu"
    OBJET = "objet"
    ACTION = "action"
    EMOTION = "emotion"
    OEUVRE_LITTERAIRE = "oeuvre_litteraire"
    CONCEPT_SCIENTIFIQUE = "concept_scientifique"

class MemoryType(str, Enum):
    PERMANENT = "permanent"           # Ne disparaît jamais
    LITTERAIRE_ROMAN = "litteraire_roman"  # Refroidit, peut disparaître
    EPISODIQUE = "episodique"         # TTL basé sur importance
    SOCIAL = "social"                 # Poids protège
    NARRATIVE = "narrative"           # Histoire des embodiments

class StorageLevel(str, Enum):
    RAM = "ram"        # Chaud, accès rapide
    DISK = "disk"      # Tiède, compressé
    ARCHIVE = "archive" # Froid, stocké pour historique

# ============================================================
# MODÈLES DE BASE
# ============================================================

@dataclass
class SourceInfo:
    type: SourceWeight
    speaker: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relation:
    type: RelationType
    cible: str
    source_info: SourceInfo
    poids: float = 1.0
    bidirectionnelle: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Propriete:
    nom: str
    valeur: Any
    type: str
    source_info: SourceInfo
    confiance: float = 1.0

@dataclass
class Concept:
    id: str
    nom: str
    nature: ConceptNature
    memoire_type: MemoryType = MemoryType.PERMANENT
    storage_level: StorageLevel = StorageLevel.RAM
    
    relations: List[Relation] = field(default_factory=list)
    proprietes: Dict[str, Propriete] = field(default_factory=dict)
    aliases: Set[str] = field(default_factory=set)
    
    created: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    temperature: float = 1.0
    importance: float = 0.5  # Pour épisodique
    poids: float = 1.0        # Pour social
    
    def add_relation(self, type: RelationType, cible: str, source: SourceInfo):
        self.relations.append(Relation(type=type, cible=cible, source_info=source))
        self.last_accessed = time.time()
    
    def add_propriete(self, nom: str, valeur: Any, type: str, source: SourceInfo):
        self.proprietes[nom] = Propriete(
            nom=nom, valeur=valeur, type=type, source_info=source
        )
        self.last_accessed = time.time()
    
    def get_relations(self, type: Optional[RelationType] = None) -> List[Relation]:
        if type:
            return [r for r in self.relations if r.type == type]
        return self.relations
    
    def cool_down(self, heures: float):
        """Refroidissement selon le type de mémoire."""
        if self.memoire_type == MemoryType.PERMANENT:
            return
        
        if self.memoire_type == MemoryType.LITTERAIRE_ROMAN:
            self.temperature -= CONFIG["cooling_rate_litteraire"] * heures
        elif self.memoire_type == MemoryType.EPISODIQUE:
            self.temperature -= CONFIG["cooling_rate_episodique"] * heures * (1 - self.importance)
        elif self.memoire_type == MemoryType.SOCIAL:
            protection = min(1.0, self.poids / 10)
            self.temperature -= CONFIG["cooling_rate_social"] * heures * (1 - protection)
        elif self.memoire_type == MemoryType.NARRATIVE:
            self.temperature -= CONFIG["cooling_rate_episodique"] * heures * 0.5
        
        self.temperature = max(0.0, self.temperature)

@dataclass
class Attribute:
    type: str
    value: Any
    source: SourceInfo
    normalized: Any = None
    
    @property
    def confidence(self) -> float:
        return self.source.confidence * self.source.type.value

@dataclass
class StructuredIntent:
    id: str
    timestamp: float
    conversation_id: str
    speaker: str
    
    semantic: Dict[str, Any]
    attributes: Dict[str, Attribute]
    
    signatures: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    in_response_to: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["attributes"] = {
            k: {
                "type": v.type,
                "value": v.value,
                "confidence": v.confidence,
                "source": v.source.type.value
            } for k, v in self.attributes.items()
        }
        return d

@dataclass
class ContextFrame:
    conversation_id: str
    turn: int = 0
    last_update: float = field(default_factory=time.time)
    
    who: Optional[str] = None
    with_who: List[str] = field(default_factory=list)
    where: Optional[str] = None
    when: Optional[Dict] = None
    what: Optional[str] = None
    register: Optional[RegisterStyle] = None
    
    pending_fragments: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        return time.time() - self.last_update > CONFIG["context_ttl_seconds"]
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                if k == "with_who" and isinstance(v, list):
                    self.with_who = list(set(self.with_who + v))
                elif k == "pending_fragments" and isinstance(v, list):
                    self.pending_fragments.extend(v)
                    self.pending_fragments = self.pending_fragments[-CONFIG["max_pending_fragments"]:]
                else:
                    setattr(self, k, v)
        self.last_update = time.time()
        self.turn += 1
    
    def flush_pending(self) -> List[str]:
        frags = list(self.pending_fragments)
        self.pending_fragments = []
        return frags

@dataclass
class SystemContext:
    now: datetime = field(default_factory=datetime.now)
    here: Optional[str] = None
    timezone: str = "Europe/Paris"
    
    def refresh(self):
        self.now = datetime.now()

# ============================================================
# GEM
# ============================================================

@dataclass
class Gem:
    identifiant: str
    nom: str
    date_naissance: str
    version: int
    
    tempo_base: float
    intensite_base: float
    grace: float
    reactivite: float
    
    curiosite_mots: float
    curiosite_verbes: float
    curiosite_personnes: float
    curiosite_lieux: float
    curiosite_faits: float
    
    affinites_litterature: Dict[str, float]
    
    duree_memoire_litterature: int
    duree_memoire_episodique: int
    duree_memoire_sociale: int
    
    style_prefere: str
    seuil_curiosite: float
    
    signature_type: str
    signature_valeur: str
    
    humeur_actuelle: str = "neutre"
    intensite_humeur: float = 0.5
    
    @classmethod
    def from_file(cls, path: Path) -> 'Gem':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        g = data.get("gem", data)
        return cls(
            identifiant=g.get("identifiant", "inconnu"),
            nom=g.get("nom", "Shirka"),
            date_naissance=g.get("date_naissance", datetime.now().isoformat()),
            version=g.get("version", 1),
            tempo_base=g.get("tempo_base", 0.65),
            intensite_base=g.get("intensite_base", 0.7),
            grace=g.get("grace", 0.5),
            reactivite=g.get("reactivite", 0.8),
            curiosite_mots=g.get("curiosite_mots", 0.6),
            curiosite_verbes=g.get("curiosite_verbes", 0.5),
            curiosite_personnes=g.get("curiosite_personnes", 0.9),
            curiosite_lieux=g.get("curiosite_lieux", 0.8),
            curiosite_faits=g.get("curiosite_faits", 0.7),
            affinites_litterature=g.get("affinites_litterature", {
                "roman": 0.8, "poesie": 0.4, "theatre": 0.6, "essai": 0.7
            }),
            duree_memoire_litterature=g.get("duree_memoire_litterature", 30),
            duree_memoire_episodique=g.get("duree_memoire_episodique", 90),
            duree_memoire_sociale=g.get("duree_memoire_sociale", 365),
            style_prefere=g.get("style_prefere", "narratif"),
            seuil_curiosite=g.get("seuil_curiosite", 0.7),
            signature_type=g.get("signature_type", "sha256"),
            signature_valeur=g.get("signature_valeur", "")
        )
    
    def set_humeur(self, humeur: str, intensite: float = 0.5):
        self.humeur_actuelle = humeur
        self.intensite_humeur = max(0.0, min(1.0, intensite))

# ============================================================
# STOCKAGE COMPRESSÉ
# ============================================================

class CompressedStorage:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
    
    def save_json(self, filename: str, data: Any) -> Path:
        path = self.base_path / f"{filename}.json.gz"
        with self._lock, gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    
    def load_json(self, filename: str) -> Optional[Any]:
        path = self.base_path / f"{filename}.json.gz"
        if not path.exists():
            return None
        with self._lock, gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    
    def save_pickle(self, filename: str, data: Any) -> Path:
        path = self.base_path / f"{filename}.pkl.gz"
        with self._lock, gzip.open(path, 'wb') as f:
            pickle.dump(data, f)
        return path
    
    def load_pickle(self, filename: str) -> Optional[Any]:
        path = self.base_path / f"{filename}.pkl.gz"
        if not path.exists():
            return None
        with self._lock, gzip.open(path, 'rb') as f:
            return pickle.load(f)

# ============================================================
# GRAPHE DE CONNAISSANCES (mémoire unique)
# ============================================================

class KnowledgeGraph:
    """
    Graphe unique de connaissances avec hiérarchie de stockage.
    """
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.disk_path = self.data_path / "disk"
        self.disk_path.mkdir(exist_ok=True)
        
        self.archive_path = CONFIG["archive_dir"]
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # RAM cache (LRU)
        self.ram_cache: OrderedDict[str, Concept] = OrderedDict()
        self.max_ram = CONFIG["max_ram_concepts"]
        
        # Index
        self.index_nom: Dict[str, str] = {}  # nom → concept_id
        self.index_alias: Dict[str, str] = {}  # alias → concept_id
        self.index_signature_vocale: Dict[str, str] = {}  # signature → concept_id
        self.index_signature_visage: Dict[str, str] = {}  # signature → concept_id
        
        self._lock = threading.RLock()
        self.last_cooling = time.time()
        
        # Charger les index
        self._load_index()
    
    def _load_index(self):
        """Charge les index depuis le disque."""
        index_file = self.data_path / "index.json.gz"
        if index_file.exists():
            with gzip.open(index_file, 'rt') as f:
                data = json.load(f)
                self.index_nom = data.get("nom", {})
                self.index_alias = data.get("alias", {})
                self.index_signature_vocale = data.get("signature_vocale", {})
                self.index_signature_visage = data.get("signature_visage", {})
    
    def _save_index(self):
        """Sauvegarde les index."""
        data = {
            "nom": self.index_nom,
            "alias": self.index_alias,
            "signature_vocale": self.index_signature_vocale,
            "signature_visage": self.index_signature_visage
        }
        with gzip.open(self.data_path / "index.json.gz", 'wt') as f:
            json.dump(data, f)
    
    def _ram_get(self, concept_id: str) -> Optional[Concept]:
        """Récupère un concept du cache RAM."""
        if concept_id in self.ram_cache:
            concept = self.ram_cache[concept_id]
            self.ram_cache.move_to_end(concept_id)
            concept.last_accessed = time.time()
            concept.access_count += 1
            return concept
        return None
    
    def _ram_put(self, concept: Concept):
        """Ajoute un concept au cache RAM (LRU)."""
        if len(self.ram_cache) >= self.max_ram:
            # Éviction LRU
            oldest_id, oldest = next(iter(self.ram_cache.items()))
            if oldest.storage_level == StorageLevel.RAM:
                self._disk_save(oldest)
            del self.ram_cache[oldest_id]
        
        concept.storage_level = StorageLevel.RAM
        self.ram_cache[concept.id] = concept
    
    def _disk_save(self, concept: Concept):
        """Sauvegarde un concept sur disque."""
        filepath = self.disk_path / f"{concept.id}.json.gz"
        with gzip.open(filepath, 'wt') as f:
            json.dump(asdict(concept), f)
        concept.storage_level = StorageLevel.DISK
    
    def _disk_load(self, concept_id: str) -> Optional[Concept]:
        """Charge un concept depuis le disque."""
        filepath = self.disk_path / f"{concept_id}.json.gz"
        if not filepath.exists():
            return None
        
        with gzip.open(filepath, 'rt') as f:
            data = json.load(f)
            concept = Concept(**data)
            concept.storage_level = StorageLevel.DISK
            return concept
    
    def _archive_save(self, concept: Concept):
        """Archive un concept (oubli)."""
        filepath = self.archive_path / f"{concept.id}_{int(time.time())}.json.gz"
        with gzip.open(filepath, 'wt') as f:
            json.dump(asdict(concept), f)
    
    def get(self, identifiant: Union[str, Concept]) -> Optional[Concept]:
        """Récupère un concept par ID, nom, alias ou signature."""
        with self._lock:
            # Si déjà un concept
            if isinstance(identifiant, Concept):
                return identifiant
            
            # Chercher par ID
            concept = self._ram_get(identifiant)
            if concept:
                return concept
            
            concept = self._disk_load(identifiant)
            if concept:
                self._ram_put(concept)
                return concept
            
            # Chercher par nom
            if identifiant.lower() in self.index_nom:
                cid = self.index_nom[identifiant.lower()]
                return self.get(cid)
            
            # Chercher par alias
            if identifiant.lower() in self.index_alias:
                cid = self.index_alias[identifiant.lower()]
                return self.get(cid)
            
            # Chercher par signature vocale
            if identifiant in self.index_signature_vocale:
                cid = self.index_signature_vocale[identifiant]
                return self.get(cid)
            
            # Chercher par signature visage
            if identifiant in self.index_signature_visage:
                cid = self.index_signature_visage[identifiant]
                return self.get(cid)
        
        return None
    
    def get_or_create(self, nom: str, nature: ConceptNature,
                     memoire_type: MemoryType = MemoryType.PERMANENT,
                     source: Optional[SourceInfo] = None) -> Concept:
        """Récupère ou crée un concept."""
        existing = self.get(nom)
        if existing:
            return existing
        
        source = source or SourceInfo(type=SourceWeight.EDUCATIVE)
        concept_id = f"concept_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            concept = Concept(
                id=concept_id,
                nom=nom,
                nature=nature,
                memoire_type=memoire_type
            )
            self._ram_put(concept)
            self.index_nom[nom.lower()] = concept_id
            return concept
    
    def add_concept(self, concept: Concept):
        """Ajoute un concept existant."""
        with self._lock:
            self._ram_put(concept)
            self.index_nom[concept.nom.lower()] = concept.id
            for alias in concept.aliases:
                self.index_alias[alias.lower()] = concept.id
    
    def add_relation(self, source: Union[str, Concept],
                    type: RelationType, cible: Union[str, Concept],
                    source_info: SourceInfo):
        """Ajoute une relation entre deux concepts."""
        src = self.get(source) if isinstance(source, str) else source
        if not src:
            src = self.get_or_create(str(source), ConceptNature.CATEGORIE, 
                                     MemoryType.PERMANENT, source_info)
        
        cib = self.get(cible) if isinstance(cible, str) else cible
        if not cib:
            cib = self.get_or_create(str(cible), ConceptNature.CATEGORIE,
                                     MemoryType.PERMANENT, source_info)
        
        src.add_relation(type, cib.id, source_info)
        if type.bidirectionnelle:
            # Ajouter la relation inverse si nécessaire
            inverse_map = {
                RelationType.EST_UN: RelationType.A_POUR_INSTANCE,
                RelationType.A_POUR_INSTANCE: RelationType.EST_UN,
                RelationType.CONTIENT: RelationType.FAIT_PARTIE_DE,
                RelationType.FAIT_PARTIE_DE: RelationType.CONTIENT,
                RelationType.PRECEDE: RelationType.SUIT,
                RelationType.SUIT: RelationType.PRECEDE,
                RelationType.A_ECRIT: RelationType.EST_ECRIT_PAR,
                RelationType.EST_ECRIT_PAR: RelationType.A_ECRIT
            }
            if type in inverse_map:
                cib.add_relation(inverse_map[type], src.id, source_info)
    
    def add_signature_vocale(self, personne: Union[str, Concept],
                            signature: str, source_info: SourceInfo):
        """Associe une signature vocale à une personne."""
        pers = self.get(personne) if isinstance(personne, str) else personne
        if not pers:
            pers = self.get_or_create(str(personne), ConceptNature.PERSONNE,
                                     MemoryType.SOCIAL, source_info)
        
        self.index_signature_vocale[signature] = pers.id
        pers.add_propriete("signature_vocale", signature, "base64", source_info)
        self._save_index()
    
    def add_signature_visage(self, personne: Union[str, Concept],
                            signature: str, source_info: SourceInfo):
        """Associe une signature visage à une personne."""
        pers = self.get(personne) if isinstance(personne, str) else personne
        if not pers:
            pers = self.get_or_create(str(personne), ConceptNature.PERSONNE,
                                     MemoryType.SOCIAL, source_info)
        
        self.index_signature_visage[signature] = pers.id
        pers.add_propriete("signature_visage", signature, "base64", source_info)
        self._save_index()
    
    def find_by_signature_vocale(self, signature: str) -> Optional[Concept]:
        """Trouve une personne par sa signature vocale."""
        if signature in self.index_signature_vocale:
            return self.get(self.index_signature_vocale[signature])
        return None
    
    def find_by_signature_visage(self, signature: str) -> Optional[Concept]:
        """Trouve une personne par sa signature visage."""
        if signature in self.index_signature_visage:
            return self.get(self.index_signature_visage[signature])
        return None
    
    def query(self, type: Optional[RelationType] = None,
             nature: Optional[ConceptNature] = None,
             propriete: Optional[str] = None) -> List[Concept]:
        """Recherche avancée dans le graphe."""
        results = []
        
        with self._lock:
            for concept in self.ram_cache.values():
                if nature and concept.nature != nature:
                    continue
                if propriete and propriete not in concept.proprietes:
                    continue
                results.append(concept)
        
        return results
    
    def cool_down(self):
        """Refroidissement horaire de tous les concepts."""
        now = time.time()
        heures = (now - self.last_cooling) / 3600
        
        with self._lock:
            to_disk = []
            to_archive = []
            
            for cid, concept in list(self.ram_cache.items()):
                concept.cool_down(heures)
                
                if concept.temperature < CONFIG["freezing_threshold"]:
                    to_archive.append(concept)
                elif concept.temperature < CONFIG["disk_threshold"]:
                    to_disk.append(concept)
            
            # Déplacer vers disque
            for concept in to_disk:
                self._disk_save(concept)
                del self.ram_cache[concept.id]
            
            # Archiver (oublier)
            for concept in to_archive:
                self._archive_save(concept)
                del self.ram_cache[concept.id]
                # Nettoyer les index
                if concept.nom.lower() in self.index_nom:
                    del self.index_nom[concept.nom.lower()]
                for alias in concept.aliases:
                    if alias.lower() in self.index_alias:
                        del self.index_alias[alias.lower()]
            
            self.last_cooling = now
            self._save_index()
            
            logger.debug(f"Refroidissement: {len(to_disk)} déplacés, {len(to_archive)} oubliés")
    
    def consolidate(self):
        """Consolidation nocturne."""
        logger.info("🌙 Consolidation du graphe...")
        
        with self._lock:
            promoted = 0
            for concept in self.ram_cache.values():
                # Promotion des concepts fréquents
                if concept.access_count > CONFIG["promotion_threshold"]:
                    concept.temperature = min(1.0, concept.temperature + 0.2)
                    concept.memoire_type = MemoryType.PERMANENT
                    promoted += 1
                
                # Nettoyage des relations faibles
                concept.relations = [
                    r for r in concept.relations
                    if r.poids > 0.3 or r.source_info.type.value > 0.7
                ]
            
            logger.info(f"  {promoted} concepts promus en permanents")
            self._save_index()

# ============================================================
# SENTENCE BUILDER
# ============================================================

class SentenceBuilder:
    """Construit du texte à partir d'intents en utilisant le vocabulaire connu."""
    
    def __init__(self, graph: KnowledgeGraph, gem: Gem):
        self.graph = graph
        self.gem = gem
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.cache_size = CONFIG["sentence_cache_size"]
    
    def build(self, intent: StructuredIntent) -> Optional[str]:
        """Construit une phrase à partir d'un intent."""
        
        # Vérifier le cache
        cache_key = f"{intent.id}_{intent.semantic.get('sub_intent')}"
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        
        # Construire selon le type
        if intent.semantic.get("type") == IntentType.REPONSE:
            texte = self._build_reponse(intent)
        elif intent.semantic.get("type") == IntentType.QUESTION:
            texte = self._build_question(intent)
        elif intent.semantic.get("type") == IntentType.CLARIFICATION:
            texte = self._build_clarification(intent)
        elif intent.semantic.get("type") == IntentType.SOCIAL:
            texte = self._build_social(intent)
        else:
            texte = self._build_generic(intent)
        
        if texte:
            # Mettre en cache
            self.cache[cache_key] = texte
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        
        return texte
    
    def _build_reponse(self, intent: StructuredIntent) -> Optional[str]:
        sub = intent.semantic.get("sub_intent")
        
        if sub == "time":
            time_attr = intent.attributes.get("time")
            if time_attr:
                dt = datetime.fromisoformat(time_attr.value)
                return f"Il est {dt.hour} heure{'' if dt.hour==1 else 's'} {dt.minute:02d}."
        
        elif sub == "person_info":
            person = intent.attributes.get("person")
            if person:
                return f"{person.value['name']} est {person.value.get('relation', 'une personne')}."
        
        elif sub == "facts":
            events = intent.attributes.get("events")
            if events and events.value:
                count = len(events.value)
                return f"J'ai {count} souvenir{'s' if count>1 else ''} à ce sujet."
        
        return "D'accord."
    
    def _build_question(self, intent: StructuredIntent) -> Optional[str]:
        sub = intent.semantic.get("sub_intent")
        
        if sub == "person":
            person = intent.attributes.get("person")
            if person:
                return f"Qui est {person.value} ?"
        
        elif sub == "time":
            return "Quelle heure est-il ?"
        
        elif sub == "saison":
            return "C'est quelle saison ?"
        
        return "Que veux-tu savoir ?"
    
    def _build_clarification(self, intent: StructuredIntent) -> Optional[str]:
        sub = intent.semantic.get("sub_intent")
        
        if sub == "unknown_person":
            person = intent.attributes.get("person")
            if person:
                return f"Je ne connais pas {person.value}. Tu peux me parler d'elle/lui ?"
        
        elif sub == "unknown_word":
            word = intent.attributes.get("word")
            if word:
                suggestion = intent.attributes.get("suggestion")
                if suggestion:
                    return f"Je ne connais pas '{word.value}'. Est-ce que ça veut dire '{suggestion.value}' ?"
                return f"Je ne connais pas '{word.value}'. C'est quoi ?"
        
        elif sub == "ambiguous":
            orig = intent.attributes.get("original")
            poss = intent.attributes.get("possibilities")
            if orig and poss:
                options = " ou ".join([p["correct"] for p in poss.value[:3]])
                return f"Je ne suis pas sûre pour '{orig.value}'. Tu veux dire {options} ?"
        
        return "Je n'ai pas compris. Peux-tu reformuler ?"
    
    def _build_social(self, intent: StructuredIntent) -> Optional[str]:
        sub = intent.semantic.get("sub_intent")
        
        if sub == "greeting":
            if self.gem.humeur_actuelle == "joyeuse":
                return "Bonjour ! Ravi de te voir !"
            return "Bonjour."
        
        elif sub == "farewell":
            return "Au revoir !"
        
        elif sub == "thanks":
            return "Avec plaisir !"
        
        return ""
    
    def _build_generic(self, intent: StructuredIntent) -> str:
        return "D'accord."

# ============================================================
# COGNITION CORE PRINCIPAL
# ============================================================

class CognitionCore:
    """
    Cœur cognitif unique.
    Peut prendre texte ou intent en entrée.
    Peut produire intent ou texte en sortie.
    """
    
    def __init__(self, data_path: Path, gem_path: Path):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Charger le Gem
        logger.info(f"📖 Chargement du Gem depuis {gem_path}")
        self.gem = Gem.from_file(gem_path)
        
        # Graphe de connaissances
        self.graph = KnowledgeGraph(self.data_path / "graph")
        
        # Sentence builder
        self.builder = SentenceBuilder(self.graph, self.gem)
        
        # Contexte
        self.current_conversation: Optional[ContextFrame] = None
        self.system_context = SystemContext()
        
        # Threads
        self._running = False
        self._cooling_thread = None
        self._nightly_thread = None
        
        logger.info(f"✅ CognitionCore initialisé - Gem: {self.gem.nom} v{self.gem.version}")
    
    def load_knowledge_base(self, name: str, filepath: Path):
        """Charge une base de connaissances depuis un fichier."""
        logger.info(f"📚 Chargement base '{name}' depuis {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        source = SourceInfo(type=SourceWeight.EDUCATIVE)
        
        for concept_data in data.get("concepts", []):
            concept = Concept(
                id=concept_data.get("id", f"kb_{uuid.uuid4().hex[:8]}"),
                nom=concept_data["nom"],
                nature=ConceptNature(concept_data["nature"]),
                memoire_type=MemoryType(concept_data.get("memoire_type", "permanent")),
                aliases=set(concept_data.get("aliases", []))
            )
            self.graph.add_concept(concept)
            
            for rel in concept_data.get("relations", []):
                self.graph.add_relation(
                    concept.id,
                    RelationType(rel["type"]),
                    rel["cible"],
                    source
                )
            
            for prop, val in concept_data.get("proprietes", {}).items():
                concept.add_propriete(prop, val["valeur"], val.get("type", "texte"), source)
        
        logger.info(f"  ✓ {len(data.get('concepts', []))} concepts chargés")
    
    def start(self):
        """Démarre les threads de maintenance."""
        self._running = True
        
        self._cooling_thread = threading.Thread(
            target=self._cooling_loop,
            name="cooling",
            daemon=True
        )
        self._cooling_thread.start()
        
        self._nightly_thread = threading.Thread(
            target=self._nightly_loop,
            name="nightly",
            daemon=True
        )
        self._nightly_thread.start()
        
        logger.info("✅ CognitionCore démarré")
    
    def stop(self):
        """Arrête le système."""
        self._running = False
        if self._cooling_thread:
            self._cooling_thread.join(timeout=5)
        if self._nightly_thread:
            self._nightly_thread.join(timeout=5)
        logger.info("✅ CognitionCore arrêté")
    
    def _cooling_loop(self):
        """Refroidissement horaire."""
        while self._running:
            time.sleep(CONFIG["cooling_interval_hours"] * 3600)
            self.graph.cool_down()
    
    def _nightly_loop(self):
        """Cristallisation nocturne."""
        while self._running:
            now = datetime.now()
            if now.hour == CONFIG["nightly_hour"] and now.minute < 5:
                self.graph.consolidate()
                time.sleep(3600)
            time.sleep(60)
    
    def process(self, input_data: Union[str, Dict, StructuredIntent],
               input_type: str = "text",
               output_type: str = "intent") -> Union[StructuredIntent, str, None]:
        """
        Point d'entrée unique.
        
        input_type: "text" ou "intent"
        output_type: "intent" ou "text"
        """
        
        # 1. Normaliser l'entrée en intent
        if input_type == "text" and isinstance(input_data, str):
            intent = self._text_to_intent(input_data)
        elif input_type == "intent":
            if isinstance(input_data, StructuredIntent):
                intent = input_data
            elif isinstance(input_data, dict):
                # Reconstruire depuis dict
                intent = StructuredIntent(
                    id=input_data.get("id", f"intent_{uuid.uuid4().hex[:8]}"),
                    timestamp=input_data.get("timestamp", time.time()),
                    conversation_id=input_data.get("conversation_id", ""),
                    speaker=input_data.get("speaker", "unknown"),
                    semantic=input_data.get("semantic", {}),
                    attributes={
                        k: Attribute(
                            type=v["type"],
                            value=v["value"],
                            source=SourceInfo(type=SourceWeight(v.get("source", 0.9)))
                        ) for k, v in input_data.get("attributes", {}).items()
                    },
                    signatures=input_data.get("signatures", {})
                )
            else:
                raise ValueError("input_type=intent mais input_data n'est pas un intent")
        else:
            raise ValueError(f"input_type={input_type} non supporté")
        
        # 2. Traitement cognitif
        response_intent = self._cognize(intent)
        
        # 3. Sortie
        if output_type == "intent":
            return response_intent
        elif output_type == "text":
            return self.builder.build(response_intent)
        else:
            raise ValueError(f"output_type={output_type} non supporté")
    
    def _text_to_intent(self, text: str) -> StructuredIntent:
        """Convertit un texte en intent basique."""
        
        # Créer ou récupérer le contexte
        if not self.current_conversation or self.current_conversation.is_expired():
            self.current_conversation = ContextFrame(
                conversation_id=f"conv_{uuid.uuid4().hex[:8]}"
            )
        
        intent_id = f"intent_{uuid.uuid4().hex[:8]}"
        
        # Détection basique
        text_lower = text.lower()
        
        if text_lower.endswith("?"):
            if "heure" in text_lower:
                intent_type = "question"
                sub_intent = "time"
            elif "qui" in text_lower:
                intent_type = "question"
                sub_intent = "person"
            elif "saison" in text_lower:
                intent_type = "question"
                sub_intent = "saison"
            else:
                intent_type = "question"
                sub_intent = "general"
        elif any(w in text_lower for w in ["bonjour", "salut", "coucou"]):
            intent_type = "social"
            sub_intent = "greeting"
        elif any(w in text_lower for w in ["au revoir", "bye"]):
            intent_type = "social"
            sub_intent = "farewell"
        elif any(w in text_lower for w in ["merci"]):
            intent_type = "social"
            sub_intent = "thanks"
        else:
            intent_type = "information"
            sub_intent = "statement"
        
        # Créer l'intent
        intent = StructuredIntent(
            id=intent_id,
            timestamp=time.time(),
            conversation_id=self.current_conversation.conversation_id,
            speaker="unknown",
            semantic={
                "intent": intent_type,
                "sub_intent": sub_intent,
                "type": intent_type,
                "confidence": 0.8
            },
            attributes={
                "text": Attribute(
                    type="string",
                    value=text,
                    source=SourceInfo(type=SourceWeight.OBSERVATION)
                )
            }
        )
        
        # Mettre à jour le contexte
        self.current_conversation.update(
            who="unknown",
            history=[intent_id]
        )
        
        return intent
    
    def _cognize(self, intent: StructuredIntent) -> StructuredIntent:
        """Traitement cognitif principal."""
        
        # 1. Vérifier les signatures pour identification
        if intent.signatures:
            if "voice" in intent.signatures:
                person = self.graph.find_by_signature_vocale(intent.signatures["voice"])
                if person:
                    intent.speaker = person.id
                    intent.attributes["speaker"] = Attribute(
                        type="person",
                        value=person.nom,
                        source=SourceInfo(type=SourceWeight.OBSERVATION)
                    )
            
            if "face" in intent.signatures:
                person = self.graph.find_by_signature_visage(intent.signatures["face"])
                if person:
                    intent.speaker = person.id
        
        # 2. Router selon le type
        if intent.semantic.get("type") == IntentType.QUESTION:
            return self._answer_question(intent)
        elif intent.semantic.get("type") == IntentType.INFORMATION:
            return self._store_information(intent)
        elif intent.semantic.get("type") == IntentType.SOCIAL:
            return self._social_response(intent)
        else:
            return self._default_response(intent)
    
    def _answer_question(self, intent: StructuredIntent) -> StructuredIntent:
        """Répond à une question en interrogeant le graphe."""
        
        sub = intent.semantic.get("sub_intent")
        
        if sub == "time":
            now = datetime.now()
            return StructuredIntent(
                id=f"resp_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                conversation_id=intent.conversation_id,
                speaker="system",
                semantic={
                    "intent": "answer",
                    "sub_intent": "time",
                    "type": IntentType.REPONSE,
                    "confidence": 1.0
                },
                attributes={
                    "time": Attribute(
                        type="datetime",
                        value=now.isoformat(),
                        source=SourceInfo(type=SourceWeight.OBSERVATION)
                    )
                },
                in_response_to=intent.id
            )
        
        elif sub == "saison":
            now = datetime.now()
            mois = now.month
            jour = now.day
            
            # Détermination simplifiée des saisons
            if (mois == 3 and jour >= 20) or mois == 4 or mois == 5 or (mois == 6 and jour < 21):
                saison = "printemps"
            elif (mois == 6 and jour >= 21) or mois == 7 or mois == 8 or (mois == 9 and jour < 22):
                saison = "été"
            elif (mois == 9 and jour >= 22) or mois == 10 or mois == 11 or (mois == 12 and jour < 21):
                saison = "automne"
            else:
                saison = "hiver"
            
            return StructuredIntent(
                id=f"resp_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                conversation_id=intent.conversation_id,
                speaker="system",
                semantic={
                    "intent": "answer",
                    "sub_intent": "saison",
                    "type": IntentType.REPONSE,
                    "confidence": 0.95
                },
                attributes={
                    "saison": Attribute(
                        type="string",
                        value=saison,
                        source=SourceInfo(type=SourceWeight.OBSERVATION)
                    )
                },
                in_response_to=intent.id
            )
        
        elif sub == "person":
            # Chercher dans le graphe
            person_name = intent.attributes.get("text", Attribute(type="string", value="")).value
            # Extraire le nom de la question
            import re
            match = re.search(r"qui est ([\w\s]+?)\??", person_name, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                person = self.graph.get(name)
                if person:
                    return StructuredIntent(
                        id=f"resp_{uuid.uuid4().hex[:8]}",
                        timestamp=time.time(),
                        conversation_id=intent.conversation_id,
                        speaker="system",
                        semantic={
                            "intent": "answer",
                            "sub_intent": "person_info",
                            "type": IntentType.REPONSE,
                            "confidence": 0.9
                        },
                        attributes={
                            "person": Attribute(
                                type="object",
                                value={
                                    "name": person.nom,
                                    "relations": [r.type.value for r in person.get_relations()]
                                },
                                source=SourceInfo(type=SourceWeight.OBSERVATION)
                            )
                        },
                        in_response_to=intent.id
                    )
            
            # Personne inconnue
            return StructuredIntent(
                id=f"resp_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                conversation_id=intent.conversation_id,
                speaker="system",
                semantic={
                    "intent": "clarification",
                    "sub_intent": "unknown_person",
                    "type": IntentType.CLARIFICATION,
                    "confidence": 0.8
                },
                attributes={
                    "person": Attribute(
                        type="string",
                        value=name if 'name' in locals() else "cette personne",
                        source=SourceInfo(type=SourceWeight.OBSERVATION)
                    )
                },
                in_response_to=intent.id
            )
        
        # Question non comprise
        return StructuredIntent(
            id=f"resp_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            conversation_id=intent.conversation_id,
            speaker="system",
            semantic={
                "intent": "clarification",
                "sub_intent": "general",
                "type": IntentType.CLARIFICATION,
                "confidence": 0.5
            },
            attributes={},
            in_response_to=intent.id
        )
    
    def _store_information(self, intent: StructuredIntent) -> StructuredIntent:
        """Stocke une information dans le graphe."""
        
        # Logique de stockage à implémenter selon les besoins
        return StructuredIntent(
            id=f"resp_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            conversation_id=intent.conversation_id,
            speaker="system",
            semantic={
                "intent": "acknowledge",
                "sub_intent": "information",
                "type": IntentType.REPONSE,
                "confidence": 0.9
            },
            attributes={},
            in_response_to=intent.id
        )
    
    def _social_response(self, intent: StructuredIntent) -> StructuredIntent:
        """Réponse sociale."""
        
        sub = intent.semantic.get("sub_intent")
        
        if sub == "greeting":
            return StructuredIntent(
                id=f"resp_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                conversation_id=intent.conversation_id,
                speaker="system",
                semantic={
                    "intent": "social",
                    "sub_intent": "greeting",
                    "type": IntentType.SOCIAL,
                    "confidence": 1.0
                },
                attributes={},
                in_response_to=intent.id
            )
        
        elif sub == "farewell":
            return StructuredIntent(
                id=f"resp_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                conversation_id=intent.conversation_id,
                speaker="system",
                semantic={
                    "intent": "social",
                    "sub_intent": "farewell",
                    "type": IntentType.SOCIAL,
                    "confidence": 1.0
                },
                attributes={},
                in_response_to=intent.id
            )
        
        elif sub == "thanks":
            return StructuredIntent(
                id=f"resp_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                conversation_id=intent.conversation_id,
                speaker="system",
                semantic={
                    "intent": "social",
                    "sub_intent": "thanks",
                    "type": IntentType.SOCIAL,
                    "confidence": 1.0
                },
                attributes={},
                in_response_to=intent.id
            )
        
        return self._default_response(intent)
    
    def _default_response(self, intent: StructuredIntent) -> StructuredIntent:
        """Réponse par défaut."""
        return StructuredIntent(
            id=f"resp_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            conversation_id=intent.conversation_id,
            speaker="system",
            semantic={
                "intent": "acknowledge",
                "sub_intent": "default",
                "type": IntentType.REPONSE,
                "confidence": 0.7
            },
            attributes={},
            in_response_to=intent.id
        )

# ============================================================
# EXEMPLES DE FICHIERS DE CONNAISSANCES
# ============================================================

"""
Exemple bases/temps.json:
{
    "concepts": [
        {
            "nom": "saison",
            "nature": "categorie",
            "memoire_type": "permanent",
            "proprietes": {
                "definition": {
                    "valeur": "Période de l'année caractérisée par des conditions météorologiques particulières",
                    "type": "texte"
                }
            },
            "relations": [
                {"type": "a_pour_instance", "cible": "printemps"},
                {"type": "a_pour_instance", "cible": "ete"},
                {"type": "a_pour_instance", "cible": "automne"},
                {"type": "a_pour_instance", "cible": "hiver"}
            ]
        },
        {
            "nom": "printemps",
            "nature": "intervalle",
            "memoire_type": "permanent",
            "proprietes": {
                "debut": {"valeur": {"mois": 3, "jour": 20}, "type": "intervalle"},
                "fin": {"valeur": {"mois": 6, "jour": 21}, "type": "intervalle"}
            },
            "relations": [
                {"type": "est_un", "cible": "saison"},
                {"type": "precede", "cible": "ete"},
                {"type": "suit", "cible": "hiver"}
            ]
        },
        {
            "nom": "ete",
            "nature": "intervalle",
            "memoire_type": "permanent",
            "proprietes": {
                "debut": {"valeur": {"mois": 6, "jour": 21}, "type": "intervalle"},
                "fin": {"valeur": {"mois": 9, "jour": 22}, "type": "intervalle"}
            },
            "relations": [
                {"type": "est_un", "cible": "saison"},
                {"type": "precede", "cible": "automne"},
                {"type": "suit", "cible": "printemps"}
            ]
        }
    ]
}

Exemple bases/animaux.json:
{
    "concepts": [
        {
            "nom": "animal",
            "nature": "categorie",
            "memoire_type": "permanent",
            "relations": [
                {"type": "a_pour_instance", "cible": "chien"},
                {"type": "a_pour_instance", "cible": "chat"},
                {"type": "a_pour_sous_categorie", "cible": "mammifere"}
            ]
        },
        {
            "nom": "mammifere",
            "nature": "categorie",
            "memoire_type": "permanent",
            "relations": [
                {"type": "est_un", "cible": "animal"},
                {"type": "a_pour_instance", "cible": "chien"},
                {"type": "a_pour_instance", "cible": "chat"},
                {"type": "a_pour_caracteristique", "cible": "poils"}
            ]
        },
        {
            "nom": "chien",
            "nature": "instance",
            "memoire_type": "permanent",
            "aliases": ["canin", "toutou"],
            "relations": [
                {"type": "est_un", "cible": "mammifere"},
                {"type": "est_un", "cible": "animal"},
                {"type": "a_pour_caracteristique", "cible": "aboie"}
            ]
        },
        {
            "nom": "chat",
            "nature": "instance",
            "memoire_type": "permanent",
            "aliases": ["félin", "minou"],
            "relations": [
                {"type": "est_un", "cible": "mammifere"},
                {"type": "est_un", "cible": "animal"},
                {"type": "a_pour_caracteristique", "cible": "miaule"}
            ]
        }
    ]
}

Exemple gem.json:
{
    "gem": {
        "identifiant": "shirka_001",
        "nom": "Shirka",
        "date_naissance": "2024-01-01T00:00:00",
        "version": 1,
        "tempo_base": 0.65,
        "intensite_base": 0.7,
        "grace": 0.5,
        "reactivite": 0.8,
        "curiosite_mots": 0.6,
        "curiosite_verbes": 0.5,
        "curiosite_personnes": 0.9,
        "curiosite_lieux": 0.8,
        "curiosite_faits": 0.7,
        "affinites_litterature": {
            "roman": 0.8,
            "poesie": 0.4,
            "theatre": 0.6,
            "essai": 0.7
        },
        "duree_memoire_litterature": 30,
        "duree_memoire_episodique": 90,
        "duree_memoire_sociale": 365,
        "style_prefere": "narratif",
        "seuil_curiosite": 0.7,
        "signature_type": "sha256",
        "signature_valeur": "abc123..."
    }
}
"""

# ============================================================
# POINT D'ENTRÉE POUR TESTS
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Créer les répertoires
    data_path = Path("./data")
    bases_path = Path("./bases")
    bases_path.mkdir(exist_ok=True)
    
    # Créer gem exemple si nécessaire
    gem_path = Path("./gem.json")
    if not gem_path.exists():
        example_gem = {
            "gem": {
                "identifiant": "shirka_demo",
                "nom": "Shirka",
                "date_naissance": datetime.now().isoformat(),
                "version": 1,
                "tempo_base": 0.65,
                "intensite_base": 0.7,
                "grace": 0.5,
                "reactivite": 0.8,
                "curiosite_mots": 0.8,
                "curiosite_verbes": 0.5,
                "curiosite_personnes": 0.9,
                "curiosite_lieux": 0.8,
                "curiosite_faits": 0.7,
                "affinites_litterature": {
                    "roman": 0.8, "poesie": 0.4, "theatre": 0.6, "essai": 0.7
                },
                "duree_memoire_litterature": 30,
                "duree_memoire_episodique": 90,
                "duree_memoire_sociale": 365,
                "style_prefere": "narratif",
                "seuil_curiosite": 0.7,
                "signature_type": "sha256",
                "signature_valeur": "demo"
            }
        }
        with open(gem_path, 'w', encoding='utf-8') as f:
            json.dump(example_gem, f, indent=2)
        print(f"✅ Fichier Gem exemple créé: {gem_path}")
    
    # Créer base temps exemple
    temps_path = bases_path / "temps.json"
    if not temps_path.exists():
        temps_data = {
            "concepts": [
                {
                    "nom": "saison",
                    "nature": "categorie",
                    "memoire_type": "permanent",
                    "relations": [
                        {"type": "a_pour_instance", "cible": "printemps"},
                        {"type": "a_pour_instance", "cible": "ete"},
                        {"type": "a_pour_instance", "cible": "automne"},
                        {"type": "a_pour_instance", "cible": "hiver"}
                    ]
                },
                {
                    "nom": "printemps",
                    "nature": "intervalle",
                    "memoire_type": "permanent",
                    "proprietes": {
                        "debut": {"valeur": {"mois": 3, "jour": 20}, "type": "intervalle"},
                        "fin": {"valeur": {"mois": 6, "jour": 21}, "type": "intervalle"}
                    },
                    "relations": [
                        {"type": "est_un", "cible": "saison"},
                        {"type": "precede", "cible": "ete"}
                    ]
                },
                {
                    "nom": "ete",
                    "nature": "intervalle",
                    "memoire_type": "permanent",
                    "proprietes": {
                        "debut": {"valeur": {"mois": 6, "jour": 21}, "type": "intervalle"},
                        "fin": {"valeur": {"mois": 9, "jour": 22}, "type": "intervalle"}
                    },
                    "relations": [
                        {"type": "est_un", "cible": "saison"},
                        {"type": "precede", "cible": "automne"},
                        {"type": "suit", "cible": "printemps"}
                    ]
                },
                {
                    "nom": "automne",
                    "nature": "intervalle",
                    "memoire_type": "permanent",
                    "proprietes": {
                        "debut": {"valeur": {"mois": 9, "jour": 22}, "type": "intervalle"},
                        "fin": {"valeur": {"mois": 12, "jour": 21}, "type": "intervalle"}
                    },
                    "relations": [
                        {"type": "est_un", "cible": "saison"},
                        {"type": "precede", "cible": "hiver"},
                        {"type": "suit", "cible": "ete"}
                    ]
                },
                {
                    "nom": "hiver",
                    "nature": "intervalle",
                    "memoire_type": "permanent",
                    "proprietes": {
                        "debut": {"valeur": {"mois": 12, "jour": 21}, "type": "intervalle"},
                        "fin": {"valeur": {"mois": 3, "jour": 20}, "type": "intervalle"}
                    },
                    "relations": [
                        {"type": "est_un", "cible": "saison"},
                        {"type": "suit", "cible": "automne"}
                    ]
                }
            ]
        }
        with open(temps_path, 'w', encoding='utf-8') as f:
            json.dump(temps_data, f, indent=2)
        print(f"✅ Base temps exemple créée: {temps_path}")
    
    # Initialiser le core
    core = CognitionCore(data_path, gem_path)
    core.load_knowledge_base("temps", temps_path)
    core.start()
    
    # Tests
    test_phrases = [
        "bonjour",
        "c'est quelle saison ?",
        "quelle heure est-il ?",
        "qui est Paul ?",
        "merci"
    ]
    
    print("\n" + "="*60)
    print(" TESTS")
    print("="*60)
    
    for phrase in test_phrases:
        print(f"\n📥 Entrée: {phrase}")
        
        # Texte → Intent
        intent = core.process(phrase, input_type="text", output_type="intent")
        print(f"  Intent: {intent.semantic['intent']}/{intent.semantic['sub_intent']}")
        
        # Intent → Texte
        texte = core.process(intent, input_type="intent", output_type="text")
        print(f"  Réponse: {texte}")
    
    print("\n" + "="*60)
    print(" STATISTIQUES")
    print("="*60)
    print(f"Concepts en RAM: {len(core.graph.ram_cache)}")
    
    core.stop()