#!/usr/bin/env python3
"""
cognition_core.py — Cœur cognitif de l'entité synthétique transcendée
Version 8.0.0 - FINALE

Architecture complète avec :
- Gem (âme immuable) chargé depuis fichier JSON
- Mémoires multiples (mots, verbes, erreurs, temporel, épisodique, social, narratif, romans, éducatif)
- Poids des sources (observation > self > éducatif > ...)
- Normalisation des verbes en concepts (COMMUNICATE, MOVE, etc.)
- Contexte conversationnel et variables systèmes
- Apprentissage des inconnus par questionnement
- Cristallisation nocturne des connaissances
- Sorties 100% structurées (intents)

Aucun hard coding - tout est dans des fichiers de configuration.
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
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod
import hashlib

__version__ = "8.0.0"
logger = logging.getLogger("CognitionCore")

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "model_dir": Path("./models"),
    "data_dir": Path("./data"),
    "gem_file": Path("./gem.json"),
    "concept_file": Path("./concepts.json"),
    "max_context_turns": 10,
    "context_ttl_seconds": 300,
    "cooling_rate_hourly": 0.05,
    "freezing_threshold": 0.1,
    "nightly_hour": 2,  # 2 AM
    "min_salience": 0.1,
    "max_pending_fragments": 5
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

class MemoryType(str, Enum):
    GEM = "gem"
    WORDS = "words"
    VERBS = "verbs"
    ERRORS = "errors"
    TEMPORAL = "temporal"
    EPISODIC = "episodic"
    SOCIAL = "social"
    NARRATIVE = "narrative"
    ROMAN = "roman"
    EDUCATIONAL = "educational"

class SourceWeight(float, Enum):
    OBSERVATION = 1.0
    SELF = 0.95
    EDUCATIVE = 0.9
    SCIENTIFIC = 0.8
    REPORTED = 0.6
    FICTION = 0.3
    INTERNET = 0.2
    RUMOR = 0.1

class Concept(str, Enum):
    COMMUNICATE = "COMMUNICATE"
    MOVE = "MOVE"
    PERCEIVE = "PERCEIVE"
    BE = "BE"
    ACTION = "ACTION"
    UNKNOWN = "UNKNOWN"

class RegisterStyle(str, Enum):
    FAMILIER = "familier"
    NEUTRE = "neutre"
    SOUTENU = "soutenu"
    AFFECTIF = "affectif"
    TECHNIQUE = "technique"

# ============================================================
# MODÈLES DE BASE
# ============================================================

@dataclass
class SourceInfo:
    """Information sur la source d'une donnée."""
    type: SourceWeight
    speaker: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0

@dataclass
class Attribute:
    """Attribut typé avec source et confiance."""
    type: str
    value: Any
    source: SourceInfo
    normalized: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence(self) -> float:
        return self.source.confidence * self.source.type.value

@dataclass
class TemporalInfo:
    """Information temporelle résolue."""
    raw: str
    iso_start: Optional[str] = None
    iso_end: Optional[str] = None
    duration: Optional[float] = None
    unit: Optional[str] = None
    confidence: float = 0.5
    source: str = "unknown"

@dataclass
class ContextFrame:
    """Contexte de conversation en cours."""
    conversation_id: str
    turn: int = 0
    last_update: float = field(default_factory=time.time)
    
    who: Optional[str] = None
    with_who: List[str] = field(default_factory=list)
    where: Optional[str] = None
    when: Optional[TemporalInfo] = None
    what: Optional[str] = None
    register: Optional[RegisterStyle] = None
    
    pending_fragments: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    durations: List[Dict] = field(default_factory=list)
    
    history: List[str] = field(default_factory=list)  # intent ids
    
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
    """Variables système (temps, lieu, etc.)."""
    now: datetime = field(default_factory=datetime.now)
    here: Optional[str] = None
    timezone: str = "Europe/Paris"
    
    def refresh(self):
        self.now = datetime.now()
    
    def resolve_temporal(self, expr: str) -> Optional[TemporalInfo]:
        """Résout une expression temporelle relative."""
        # À implémenter avec dateparser si disponible
        return None

@dataclass
class StructuredIntent:
    """
    Intent purement structurel - format unique dans tout le système.
    """
    id: str
    timestamp: float
    conversation_id: str
    speaker: str
    
    semantic: Dict[str, Any]  # intent, sub_intent, type, confidence
    attributes: Dict[str, Attribute]
    
    # Enrichissements (optionnels)
    analysis: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    memory_hint: Dict[str, Any] = field(default_factory=dict)
    
    # Relations
    in_response_to: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["attributes"] = {
            k: {
                "type": v.type,
                "value": v.value,
                "confidence": v.confidence,
                "source": v.source.type.value,
                "normalized": v.normalized
            } for k, v in self.attributes.items()
        }
        return d

# ============================================================
# INTERFACE MÉMOIRE
# ============================================================

class MemoryInterface(ABC):
    """Interface commune à toutes les mémoires."""
    
    @abstractmethod
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        """Point d'entrée unique pour le moteur d'inférence."""
        pass
    
    @abstractmethod
    def update(self, intent: StructuredIntent) -> bool:
        """Met à jour la mémoire à partir d'un intent."""
        pass
    
    @abstractmethod
    def consolidate(self) -> Dict:
        """Consolidation nocturne."""
        pass
    
    @abstractmethod
    def relevance(self, query_type: str, constraints: Dict) -> float:
        """Score de pertinence pour une requête."""
        pass

# ============================================================
# GEM - ÂME IMMUABLE
# ============================================================

@dataclass
class Gem:
    """Âme de l'entité - immuable, définit la personnalité."""
    
    # Identité
    identifiant: str
    nom: str
    date_naissance: str
    version: int
    
    # Personnalité
    tempo_base: float
    intensite_base: float
    grace: float
    reactivite: float
    
    # Curiosités
    curiosite_mots: float
    curiosite_verbes: float
    curiosite_personnes: float
    curiosite_lieux: float
    curiosite_faits: float
    
    # Affinités
    affinites_litterature: Dict[str, float]
    
    # Durées mémoire (jours)
    duree_memoire_litterature: int
    duree_memoire_episodique: int
    duree_memoire_sociale: int
    
    # Préférences
    style_prefere: str
    seuil_curiosite: float
    
    # Signature
    signature_type: str
    signature_valeur: str
    
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
# MÉMOIRE: VERBES
# ============================================================

@dataclass
class VerbEntry:
    infinitive: str
    concept: Concept
    synonyms: Set[str] = field(default_factory=set)
    embeddings: Optional[List[float]] = None
    source_weights: Dict[str, float] = field(default_factory=dict)
    frequency: int = 0

class VerbMemory(MemoryInterface):
    """Mémoire des verbes avec concepts."""
    
    def __init__(self, storage: CompressedStorage, gem: Gem):
        self.storage = storage
        self.gem = gem
        self.verbs: Dict[str, VerbEntry] = {}
        self._lock = threading.RLock()
        self._init_defaults()
        self._load()
    
    def _init_defaults(self):
        """Verbes de base avec concepts."""
        defaults = [
            ("parler", Concept.COMMUNICATE, ["discuter", "causer", "dialoguer"]),
            ("dire", Concept.COMMUNICATE, ["annoncer", "déclarer"]),
            ("aller", Concept.MOVE, ["se déplacer", "partir"]),
            ("venir", Concept.MOVE, ["arriver"]),
            ("voir", Concept.PERCEIVE, ["regarder", "observer"]),
            ("entendre", Concept.PERCEIVE, ["écouter"]),
            ("être", Concept.BE, ["rester", "demeurer"]),
            ("faire", Concept.ACTION, ["effectuer", "réaliser"]),
        ]
        for inf, concept, syns in defaults:
            self.verbs[inf] = VerbEntry(
                infinitive=inf,
                concept=concept,
                synonyms=set(syns)
            )
    
    def add(self, infinitive: str, concept: Concept, source: SourceInfo):
        inf = infinitive.lower()
        with self._lock:
            if inf in self.verbs:
                self.verbs[inf].frequency += 1
                self.verbs[inf].source_weights[source.type.name] = \
                    self.verbs[inf].source_weights.get(source.type.name, 0) + source.type.value
            else:
                self.verbs[inf] = VerbEntry(
                    infinitive=inf,
                    concept=concept,
                    source_weights={source.type.name: source.type.value},
                    frequency=1
                )
            self._save()
    
    def add_synonym(self, verb: str, synonym: str, source: SourceInfo):
        verb = verb.lower()
        syn = synonym.lower()
        with self._lock:
            if verb in self.verbs:
                self.verbs[verb].synonyms.add(syn)
            if syn in self.verbs:
                self.verbs[syn].synonyms.add(verb)
            self._save()
    
    def resolve(self, verb: str) -> Tuple[Concept, float, Optional[str]]:
        """
        Résout un verbe en concept.
        Retourne (concept, confiance, forme_normalisée)
        """
        v = verb.lower()
        
        # Recherche directe
        if v in self.verbs:
            entry = self.verbs[v]
            return entry.concept, 1.0, entry.infinitive
        
        # Recherche par synonyme
        for inf, entry in self.verbs.items():
            if v in entry.synonyms:
                return entry.concept, 0.9, inf
        
        return Concept.UNKNOWN, 0.1, v
    
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        if query_type == "resolve":
            verb = constraints.get("verb")
            if verb:
                concept, conf, norm = self.resolve(verb)
                return {
                    "concept": concept.value,
                    "confidence": conf,
                    "normalized": norm
                }
        return {"concept": Concept.UNKNOWN.value, "confidence": 0}
    
    def update(self, intent: StructuredIntent) -> bool:
        # Apprentissage via clarification
        if intent.semantic.get("sub_intent") == "verb_clarification":
            verb = intent.attributes.get("verb")
            concept = intent.attributes.get("concept")
            if verb and concept:
                source = SourceInfo(
                    type=SourceWeight.SELF,
                    speaker=intent.speaker
                )
                self.add(verb.value, Concept(concept.value), source)
                return True
        return False
    
    def consolidate(self) -> Dict:
        """Consolidation nocturne."""
        with self._lock:
            # Pas de nettoyage pour les verbes (permanents)
            return {"verbs": len(self.verbs)}
    
    def relevance(self, query_type: str, constraints: Dict) -> float:
        return 0.9 if query_type == "resolve" else 0.1
    
    def _save(self):
        data = {
            v: {
                "infinitive": e.infinitive,
                "concept": e.concept.value,
                "synonyms": list(e.synonyms),
                "source_weights": e.source_weights,
                "frequency": e.frequency
            } for v, e in self.verbs.items()
        }
        self.storage.save_json("verb_memory", data)
    
    def _load(self):
        data = self.storage.load_json("verb_memory")
        if data:
            for v, vdata in data.items():
                self.verbs[v] = VerbEntry(
                    infinitive=vdata["infinitive"],
                    concept=Concept(vdata["concept"]),
                    synonyms=set(vdata.get("synonyms", [])),
                    source_weights=vdata.get("source_weights", {}),
                    frequency=vdata.get("frequency", 0)
                )

# ============================================================
# MÉMOIRE: MOTS
# ============================================================

@dataclass
class WordEntry:
    word: str
    pos: str
    synonyms: Set[str] = field(default_factory=set)
    definitions: List[str] = field(default_factory=list)
    source_weights: Dict[str, float] = field(default_factory=dict)

class WordMemory(MemoryInterface):
    """Mémoire des mots et synonymes."""
    
    def __init__(self, storage: CompressedStorage, gem: Gem):
        self.storage = storage
        self.gem = gem
        self.words: Dict[str, WordEntry] = {}
        self._lock = threading.RLock()
        self._load()
    
    def add(self, word: str, pos: str, source: SourceInfo):
        w = word.lower()
        with self._lock:
            if w not in self.words:
                self.words[w] = WordEntry(word=w, pos=pos)
            self.words[w].source_weights[source.type.name] = \
                self.words[w].source_weights.get(source.type.name, 0) + source.type.value
            self._save()
    
    def add_synonym(self, word: str, synonym: str, source: SourceInfo):
        w = word.lower()
        s = synonym.lower()
        with self._lock:
            if w in self.words:
                self.words[w].synonyms.add(s)
            if s in self.words:
                self.words[s].synonyms.add(w)
            self._save()
    
    def add_definition(self, word: str, definition: str, source: SourceInfo):
        w = word.lower()
        with self._lock:
            if w in self.words:
                self.words[w].definitions.append(definition)
            self._save()
    
    def has(self, word: str) -> bool:
        return word.lower() in self.words
    
    def expand(self, word: str) -> Set[str]:
        w = word.lower()
        if w not in self.words:
            return {w}
        return {w} | self.words[w].synonyms
    
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        if query_type == "definition":
            word = constraints.get("word", "").lower()
            if word in self.words:
                entry = self.words[word]
                return {
                    "word": entry.word,
                    "pos": entry.pos,
                    "definitions": entry.definitions,
                    "synonyms": list(entry.synonyms),
                    "confidence": min(1.0, sum(entry.source_weights.values()) / 5)
                }
        return {}
    
    def update(self, intent: StructuredIntent) -> bool:
        if intent.semantic.get("sub_intent") == "new_word":
            word = intent.attributes.get("word")
            pos = intent.attributes.get("pos", Attribute(type="string", value="nom"))
            source = SourceInfo(type=SourceWeight.SELF, speaker=intent.speaker)
            self.add(word.value, pos.value, source)
            return True
        return False
    
    def consolidate(self) -> Dict:
        with self._lock:
            return {"words": len(self.words)}
    
    def relevance(self, query_type: str, constraints: Dict) -> float:
        return 0.9 if query_type == "definition" else 0.1
    
    def _save(self):
        data = {
            w: {
                "word": e.word,
                "pos": e.pos,
                "synonyms": list(e.synonyms),
                "definitions": e.definitions,
                "source_weights": e.source_weights
            } for w, e in self.words.items()
        }
        self.storage.save_json("word_memory", data)
    
    def _load(self):
        data = self.storage.load_json("word_memory")
        if data:
            for w, wdata in data.items():
                self.words[w] = WordEntry(
                    word=wdata["word"],
                    pos=wdata.get("pos", "nom"),
                    synonyms=set(wdata.get("synonyms", [])),
                    definitions=wdata.get("definitions", []),
                    source_weights=wdata.get("source_weights", {})
                )

# ============================================================
# MÉMOIRE: ERREURS
# ============================================================

class ErrorMemory(MemoryInterface):
    """Fautes et corrections apprises."""
    
    def __init__(self, db_path: Path, gem: Gem):
        self.db_path = Path(db_path)
        self.gem = gem
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    wrong TEXT PRIMARY KEY,
                    correct TEXT NOT NULL,
                    concept TEXT,
                    confidence REAL DEFAULT 0.8,
                    count INTEGER DEFAULT 1,
                    first_seen REAL,
                    last_seen REAL,
                    source_weights TEXT,
                    contexts TEXT
                )
            """)
            conn.execute("CREATE INDEX idx_errors_last_seen ON errors(last_seen)")
    
    def add(self, wrong: str, correct: str, concept: Optional[str] = None,
            source: SourceInfo = None, context: str = None):
        wrong = wrong.lower()
        correct = correct.lower()
        now = time.time()
        source = source or SourceInfo(type=SourceWeight.SELF)
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT confidence, count, source_weights FROM errors WHERE wrong = ?",
                (wrong,)
            )
            row = cur.fetchone()
            
            if row:
                old_conf, count, weights_json = row
                weights = json.loads(weights_json) if weights_json else {}
                weights[source.type.name] = weights.get(source.type.name, 0) + 1
                new_conf = min(0.95, old_conf + 0.03)
                conn.execute("""
                    UPDATE errors 
                    SET confidence = ?, count = ?, last_seen = ?, source_weights = ?
                    WHERE wrong = ?
                """, (new_conf, count + 1, now, json.dumps(weights), wrong))
            else:
                weights = {source.type.name: 1}
                conn.execute("""
                    INSERT INTO errors 
                    (wrong, correct, concept, confidence, count, first_seen, last_seen, source_weights)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                """, (wrong, correct, concept, source.type.value, now, now, json.dumps(weights)))
    
    def get(self, wrong: str) -> Optional[Tuple[str, float, Optional[str]]]:
        wrong = wrong.lower()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT correct, confidence, concept FROM errors WHERE wrong = ?",
                (wrong,)
            )
            row = cur.fetchone()
            if row:
                return row[0], row[1], row[2]
        return None
    
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        if query_type == "correction":
            word = constraints.get("word", "")
            result = self.get(word)
            if result:
                correct, conf, concept = result
                return {
                    "correct": correct,
                    "confidence": conf,
                    "concept": concept
                }
        return {}
    
    def update(self, intent: StructuredIntent) -> bool:
        if intent.semantic.get("sub_intent") == "correction":
            wrong = intent.attributes.get("wrong")
            correct = intent.attributes.get("correct")
            if wrong and correct:
                source = SourceInfo(type=SourceWeight.SELF, speaker=intent.speaker)
                self.add(wrong.value, correct.value, source=source)
                return True
        return False
    
    def consolidate(self) -> Dict:
        """Nettoyage nocturne des erreurs."""
        max_age = self.gem.duree_memoire_episodique
        cutoff = time.time() - (max_age * 24 * 3600)
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                DELETE FROM errors 
                WHERE last_seen < ? AND confidence < 0.3
            """, (cutoff,))
            deleted = cur.rowcount
            return {"errors_deleted": deleted}
    
    def relevance(self, query_type: str, constraints: Dict) -> float:
        return 0.8 if query_type == "correction" else 0.1

# ============================================================
# MÉMOIRE: SOCIALE
# ============================================================

@dataclass
class Person:
    id: str
    name: str
    nicknames: Set[str]
    relation: str
    gender: str
    weight: float
    metadata: Dict[str, Any]
    first_met: float
    last_interaction: float
    interaction_count: int

class SocialMemory(MemoryInterface):
    """Personnes et relations."""
    
    def __init__(self, db_path: Path, gem: Gem):
        self.db_path = Path(db_path)
        self.gem = gem
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    nicknames TEXT,
                    relation TEXT,
                    gender TEXT,
                    weight REAL DEFAULT 1.0,
                    metadata TEXT,
                    first_met REAL,
                    last_interaction REAL,
                    interaction_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX idx_persons_name ON persons(name)")
    
    def add_person(self, name: str, relation: str = "unknown",
                   source: SourceInfo = None) -> str:
        pid = f"person_{uuid.uuid4().hex[:8]}"
        now = time.time()
        weight = source.type.value if source else 1.0
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO persons 
                (id, name, nicknames, relation, gender, weight, metadata, first_met, last_interaction, interaction_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (pid, name, json.dumps([]), relation, "unknown",
                  weight, json.dumps({}), now, now, 1))
        return pid
    
    def find_by_name(self, name: str) -> Optional[Dict]:
        name_lower = name.lower()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM persons WHERE LOWER(name) = ?",
                (name_lower,)
            )
            row = cur.fetchone()
            if row:
                return self._row_to_dict(row)
            
            # Recherche dans les nicknames
            cur = conn.execute("SELECT id, nicknames FROM persons")
            for pid, nick_json in cur:
                nicknames = json.loads(nick_json)
                if name_lower in [n.lower() for n in nicknames]:
                    cur2 = conn.execute("SELECT * FROM persons WHERE id = ?", (pid,))
                    row2 = cur2.fetchone()
                    if row2:
                        return self._row_to_dict(row2)
        return None
    
    def add_interaction(self, person_id: str):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE persons 
                SET weight = weight + 1, last_interaction = ?, interaction_count = interaction_count + 1
                WHERE id = ?
            """, (time.time(), person_id))
    
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        if query_type == "person":
            name = constraints.get("name")
            if name:
                person = self.find_by_name(name)
                if person:
                    return {
                        "person": person,
                        "confidence": min(1.0, person["weight"] / 10)
                    }
        return {}
    
    def update(self, intent: StructuredIntent) -> bool:
        if intent.semantic.get("sub_intent") == "new_person":
            name = intent.attributes.get("name")
            if name:
                source = SourceInfo(type=SourceWeight.SELF, speaker=intent.speaker)
                self.add_person(name.value, source=source)
                return True
        return False
    
    def consolidate(self) -> Dict:
        """Nettoyage social - poids protège de l'oubli."""
        max_age = self.gem.duree_memoire_sociale
        cutoff = time.time() - (max_age * 24 * 3600)
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                DELETE FROM persons 
                WHERE last_interaction < ? AND weight < 5
            """, (cutoff,))
            deleted = cur.rowcount
            return {"social_deleted": deleted}
    
    def relevance(self, query_type: str, constraints: Dict) -> float:
        return 0.9 if query_type == "person" else 0.1
    
    def _row_to_dict(self, row) -> Dict:
        return {
            "id": row[0],
            "name": row[1],
            "nicknames": json.loads(row[2]),
            "relation": row[3],
            "gender": row[4],
            "weight": row[5],
            "metadata": json.loads(row[6]),
            "first_met": row[7],
            "last_interaction": row[8],
            "interaction_count": row[9]
        }

# ============================================================
# MÉMOIRE: TEMPORELLE
# ============================================================

@dataclass
class TemporalExpression:
    expression: str
    type: str
    resolution: str
    synonyms: Set[str] = field(default_factory=set)

class TemporalMemory(MemoryInterface):
    """Expressions temporelles."""
    
    def __init__(self, storage: CompressedStorage):
        self.storage = storage
        self.expressions: Dict[str, TemporalExpression] = {}
        self._lock = threading.RLock()
        self._init_defaults()
        self._load()
    
    def _init_defaults(self):
        defaults = [
            ("aujourd'hui", "relative", "today"),
            ("demain", "relative", "tomorrow"),
            ("hier", "relative", "yesterday"),
            ("maintenant", "relative", "now"),
            ("ce matin", "relative", "this_morning"),
            ("ce midi", "relative", "this_noon"),
            ("ce soir", "relative", "this_evening"),
            ("cette nuit", "relative", "tonight"),
        ]
        for expr, typ, res in defaults:
            self.expressions[expr] = TemporalExpression(expr, typ, res)
        self.expressions["aujourd'hui"].synonyms = {"ajd", "auj"}
        self.expressions["demain"].synonyms = {"dem", "2m1"}
    
    def resolve(self, expr: str, base: Optional[datetime] = None) -> Optional[TemporalInfo]:
        expr_lower = expr.lower()
        if expr_lower in self.expressions:
            e = self.expressions[expr_lower]
            return TemporalInfo(
                raw=expr,
                confidence=0.9,
                source="temporal_memory"
            )
        return None
    
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        if query_type == "resolve":
            expr = constraints.get("expression")
            if expr:
                result = self.resolve(expr)
                if result:
                    return {"temporal": asdict(result)}
        return {}
    
    def update(self, intent: StructuredIntent) -> bool:
        return False
    
    def consolidate(self) -> Dict:
        return {"temporal_expressions": len(self.expressions)}
    
    def relevance(self, query_type: str, constraints: Dict) -> float:
        return 0.7 if query_type == "resolve" else 0.1
    
    def _save(self):
        data = {
            e: {
                "expression": expr.expression,
                "type": expr.type,
                "resolution": expr.resolution,
                "synonyms": list(expr.synonyms)
            } for e, expr in self.expressions.items()
        }
        self.storage.save_json("temporal_memory", data)
    
    def _load(self):
        data = self.storage.load_json("temporal_memory")
        if data:
            for e, edata in data.items():
                self.expressions[e] = TemporalExpression(
                    expression=edata["expression"],
                    type=edata["type"],
                    resolution=edata["resolution"],
                    synonyms=set(edata.get("synonyms", []))
                )

# ============================================================
# MÉMOIRE: ÉPISODIQUE
# ============================================================

class EpisodicMemory(MemoryInterface):
    """Expériences vécues avec TTL."""
    
    def __init__(self, db_path: Path, gem: Gem):
        self.db_path = Path(db_path)
        self.gem = gem
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic (
                    id TEXT PRIMARY KEY,
                    concept TEXT,
                    verb TEXT,
                    agent TEXT,
                    objects TEXT,
                    location TEXT,
                    start_time REAL,
                    end_time REAL,
                    salience REAL,
                    source_weight REAL,
                    intent_id TEXT,
                    raw_text TEXT,
                    ts REAL
                )
            """)
            conn.execute("CREATE INDEX idx_episodic_concept ON episodic(concept)")
            conn.execute("CREATE INDEX idx_episodic_ts ON episodic(ts)")
    
    def add_event(self, concept: Concept, verb: str, agent: List[str],
                  objects: List[str], location: Optional[str],
                  start_time: Optional[float], end_time: Optional[float],
                  salience: float, source: SourceInfo,
                  intent_id: str, raw_text: str) -> str:
        eid = f"ep_{uuid.uuid4().hex[:8]}"
        now = time.time()
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO episodic 
                (id, concept, verb, agent, objects, location, start_time, end_time,
                 salience, source_weight, intent_id, raw_text, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (eid, concept.value, verb, json.dumps(agent), json.dumps(objects),
                  location, start_time, end_time, salience, source.type.value,
                  intent_id, raw_text, now))
        return eid
    
    def query_by_concept(self, concept: Concept, days: int = 30) -> List[Dict]:
        cutoff = time.time() - (days * 24 * 3600)
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT * FROM episodic 
                WHERE concept = ? AND ts > ?
                ORDER BY ts DESC
            """, (concept.value, cutoff))
            
            for row in cur:
                results.append({
                    "id": row[0],
                    "concept": row[1],
                    "verb": row[2],
                    "agent": json.loads(row[3]),
                    "objects": json.loads(row[4]),
                    "location": row[5],
                    "start_time": row[6],
                    "end_time": row[7],
                    "salience": row[8],
                    "source_weight": row[9],
                    "intent_id": row[10],
                    "raw_text": row[11],
                    "ts": row[12]
                })
        return results
    
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        if query_type == "by_concept":
            concept_name = constraints.get("concept")
            days = constraints.get("days", 30)
            if concept_name:
                try:
                    concept = Concept(concept_name)
                    results = self.query_by_concept(concept, days)
                    return {
                        "events": results,
                        "count": len(results),
                        "confidence": 0.8
                    }
                except ValueError:
                    pass
        return {"events": [], "count": 0}
    
    def update(self, intent: StructuredIntent) -> bool:
        if intent.semantic.get("type") == IntentType.INFORMATION:
            # Les intents information sont automatiquement stockés
            concept_name = intent.analysis.get("verb", {}).get("concept")
            if concept_name:
                try:
                    concept = Concept(concept_name)
                    self.add_event(
                        concept=concept,
                        verb=intent.analysis.get("verb", {}).get("lemma", "unknown"),
                        agent=[intent.speaker] if intent.speaker else [],
                        objects=[],
                        location=intent.context.get("where"),
                        start_time=None,
                        end_time=None,
                        salience=intent.memory_hint.get("salience", 0.3),
                        source=SourceInfo(type=SourceWeight.OBSERVATION),
                        intent_id=intent.id,
                        raw_text=intent.attributes.get("text", {}).value if "text" in intent.attributes else ""
                    )
                    return True
                except ValueError:
                    pass
        return False
    
    def consolidate(self) -> Dict:
        """Nettoyage des vieux souvenirs peu importants."""
        max_age = self.gem.duree_memoire_episodique
        cutoff = time.time() - (max_age * 24 * 3600)
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                DELETE FROM episodic 
                WHERE ts < ? AND salience < ?
            """, (cutoff, CONFIG["min_salience"]))
            deleted = cur.rowcount
            return {"episodic_deleted": deleted}
    
    def relevance(self, query_type: str, constraints: Dict) -> float:
        return 0.9 if query_type == "by_concept" else 0.1

# ============================================================
# MÉMOIRE: ROMANS (éphémère)
# ============================================================

@dataclass
class Roman:
    id: str
    title: str
    author: str
    genre: str
    year: Optional[int]
    characters: List[str]
    summary: str
    themes: List[str]
    source_weight: float
    temperature: float = 1.0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

class RomanMemory(MemoryInterface):
    """Romans - éphémère, refroidit."""
    
    def __init__(self, storage: CompressedStorage, gem: Gem):
        self.storage = storage
        self.gem = gem
        self.romans: Dict[str, Roman] = {}
        self.last_cooling = time.time()
        self._lock = threading.RLock()
        self._load()
    
    def add(self, title: str, author: str, genre: str,
            characters: List[str], summary: str, themes: List[str],
            source: SourceInfo, year: int = None) -> str:
        rid = f"roman_{uuid.uuid4().hex[:8]}"
        affinite = self.gem.affinites_litterature.get(genre, 0.5)
        temp_initiale = 0.5 + (affinite * 0.5)
        
        with self._lock:
            self.romans[rid] = Roman(
                id=rid,
                title=title,
                author=author,
                genre=genre,
                year=year,
                characters=characters,
                summary=summary,
                themes=themes,
                source_weight=source.type.value,
                temperature=temp_initiale
            )
            self._save()
            return rid
    
    def get(self, roman_id: str) -> Optional[Roman]:
        with self._lock:
            if roman_id in self.romans:
                r = self.romans[roman_id]
                r.access_count += 1
                r.last_accessed = time.time()
                affinite = self.gem.affinites_litterature.get(r.genre, 0.5)
                r.temperature = min(1.0, r.temperature + (0.1 * affinite))
                self._save()
                return r
            return None
    
    def search(self, query: str) -> List[Roman]:
        q = query.lower()
        with self._lock:
            results = []
            for r in self.romans.values():
                if r.temperature < CONFIG["freezing_threshold"]:
                    continue
                if q in r.title.lower() or q in r.author.lower():
                    results.append(r)
                    affinite = self.gem.affinites_litterature.get(r.genre, 0.5)
                    r.temperature = min(1.0, r.temperature + (0.05 * affinite))
            self._save()
            return results
    
    def apply_cooling(self):
        now = time.time()
        hours = (now - self.last_cooling) / 3600
        
        with self._lock:
            to_remove = []
            for rid, r in self.romans.items():
                affinite = self.gem.affinites_litterature.get(r.genre, 0.5)
                facteur = 1.0 - (affinite * 0.5)
                r.temperature = max(0.0, r.temperature - (CONFIG["cooling_rate_hourly"] * hours * facteur))
                if r.temperature < CONFIG["freezing_threshold"]:
                    to_remove.append(rid)
            
            for rid in to_remove:
                logger.info(f"❄️ Roman oublié: {self.romans[rid].title}")
                del self.romans[rid]
            
            self.last_cooling = now
            self._save()
    
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        if query_type == "search":
            query = constraints.get("query", "")
            results = self.search(query)
            return {
                "romans": [asdict(r) for r in results],
                "count": len(results)
            }
        return {"romans": []}
    
    def update(self, intent: StructuredIntent) -> bool:
        if intent.semantic.get("sub_intent") == "new_roman":
            title = intent.attributes.get("title")
            author = intent.attributes.get("author")
            if title and author:
                source = SourceInfo(
                    type=SourceWeight.FICTION,
                    speaker=intent.speaker
                )
                self.add(
                    title=title.value,
                    author=author.value,
                    genre=intent.attributes.get("genre", Attribute(type="string", value="roman")).value,
                    characters=intent.attributes.get("characters", Attribute(type="list", value=[])).value,
                    summary=intent.attributes.get("summary", Attribute(type="string", value="")).value,
                    themes=intent.attributes.get("themes", Attribute(type="list", value=[])).value,
                    source=source
                )
                return True
        return False
    
    def consolidate(self) -> Dict:
        self.apply_cooling()
        return {"romans": len(self.romans)}
    
    def relevance(self, query_type: str, constraints: Dict) -> float:
        return 0.8 if query_type == "search" else 0.1
    
    def _save(self):
        data = {
            rid: {
                "id": r.id,
                "title": r.title,
                "author": r.author,
                "genre": r.genre,
                "year": r.year,
                "characters": r.characters,
                "summary": r.summary,
                "themes": r.themes,
                "source_weight": r.source_weight,
                "temperature": r.temperature,
                "last_accessed": r.last_accessed,
                "access_count": r.access_count
            } for rid, r in self.romans.items()
        }
        self.storage.save_json("roman_memory", data)
    
    def _load(self):
        data = self.storage.load_json("roman_memory")
        if data:
            for rid, rdata in data.items():
                self.romans[rid] = Roman(
                    id=rdata["id"],
                    title=rdata["title"],
                    author=rdata["author"],
                    genre=rdata.get("genre", "roman"),
                    year=rdata.get("year"),
                    characters=rdata.get("characters", []),
                    summary=rdata.get("summary", ""),
                    themes=rdata.get("themes", []),
                    source_weight=rdata.get("source_weight", 0.3),
                    temperature=rdata.get("temperature", 1.0),
                    last_accessed=rdata.get("last_accessed", time.time()),
                    access_count=rdata.get("access_count", 0)
                )

# ============================================================
# MÉMOIRE: ÉDUCATIVE (permanente)
# ============================================================

@dataclass
class EducationalWork:
    id: str
    title: str
    author: str
    period: str
    genre: str
    importance: float
    content: Dict[str, Any]
    source_weight: float
    learned_date: float = field(default_factory=time.time)

class EducationalMemory(MemoryInterface):
    """Littérature éducative - permanente."""
    
    def __init__(self, storage: CompressedStorage):
        self.storage = storage
        self.works: Dict[str, EducationalWork] = {}
        self._lock = threading.RLock()
        self._load()
    
    def add(self, title: str, author: str, period: str,
            genre: str, importance: float, content: Dict,
            source: SourceInfo) -> str:
        wid = f"edu_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self.works[wid] = EducationalWork(
                id=wid,
                title=title,
                author=author,
                period=period,
                genre=genre,
                importance=importance,
                content=content,
                source_weight=source.type.value
            )
            self._save()
            return wid
    
    def search(self, query: str) -> List[EducationalWork]:
        q = query.lower()
        with self._lock:
            return [w for w in self.works.values() 
                    if q in w.title.lower() or q in w.author.lower()]
    
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        if query_type == "search":
            query = constraints.get("query", "")
            results = self.search(query)
            return {
                "works": [asdict(w) for w in results],
                "count": len(results)
            }
        return {"works": []}
    
    def update(self, intent: StructuredIntent) -> bool:
        if intent.semantic.get("sub_intent") == "new_knowledge":
            title = intent.attributes.get("title")
            if title:
                source = SourceInfo(
                    type=SourceWeight.EDUCATIVE,
                    speaker=intent.speaker
                )
                self.add(
                    title=title.value,
                    author=intent.attributes.get("author", Attribute(type="string", value="")).value,
                    period=intent.attributes.get("period", Attribute(type="string", value="contemporain")).value,
                    genre=intent.attributes.get("genre", Attribute(type="string", value="essai")).value,
                    importance=intent.attributes.get("importance", Attribute(type="number", value=0.8)).value,
                    content=intent.attributes.get("content", Attribute(type="object", value={})).value,
                    source=source
                )
                return True
        return False
    
    def consolidate(self) -> Dict:
        with self._lock:
            return {"educational": len(self.works)}
    
    def relevance(self, query_type: str, constraints: Dict) -> float:
        return 0.7 if query_type == "search" else 0.1
    
    def _save(self):
        data = {
            wid: {
                "id": w.id,
                "title": w.title,
                "author": w.author,
                "period": w.period,
                "genre": w.genre,
                "importance": w.importance,
                "content": w.content,
                "source_weight": w.source_weight,
                "learned_date": w.learned_date
            } for wid, w in self.works.items()
        }
        self.storage.save_json("educational_memory", data)
    
    def _load(self):
        data = self.storage.load_json("educational_memory")
        if data:
            for wid, wdata in data.items():
                self.works[wid] = EducationalWork(
                    id=wdata["id"],
                    title=wdata["title"],
                    author=wdata["author"],
                    period=wdata["period"],
                    genre=wdata["genre"],
                    importance=wdata["importance"],
                    content=wdata.get("content", {}),
                    source_weight=wdata.get("source_weight", 0.9),
                    learned_date=wdata.get("learned_date", time.time())
                )

# ============================================================
# NARRATIVE MEMORY (Livre de vie)
# ============================================================

@dataclass
class NarrativePage:
    page_num: int
    start_day: int
    end_day: int
    events: List[Dict]
    summary: str = ""

@dataclass
class NarrativeSection:
    month: int
    year: int
    pages: Dict[int, NarrativePage]
    summary: str = ""

@dataclass
class NarrativeChapter:
    year: int
    sections: Dict[int, NarrativeSection]
    summary: str = ""
    title: str = ""

@dataclass
class NarrativeBook:
    book_id: str
    embodiment_name: str
    start_date: float
    end_date: Optional[float]
    chapters: Dict[int, NarrativeChapter]
    summary: str = ""

class NarrativeMemory(MemoryInterface):
    """Livre de vie structuré."""
    
    def __init__(self, base_path: Path, storage: CompressedStorage):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.storage = storage
        self.current_book: Optional[NarrativeBook] = None
        self._lock = threading.RLock()
        self._load_current()
    
    def start_new_book(self, embodiment_name: str) -> str:
        book_id = f"book_{uuid.uuid4().hex[:8]}"
        now = time.time()
        
        with self._lock:
            self.current_book = NarrativeBook(
                book_id=book_id,
                embodiment_name=embodiment_name,
                start_date=now,
                end_date=None,
                chapters={}
            )
            self._save_current()
            return book_id
    
    def add_event(self, event: Dict, source: SourceInfo):
        if not self.current_book:
            self.start_new_book("default")
        
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day
        page_num = 1 if day <= 15 else 2
        
        with self._lock:
            if year not in self.current_book.chapters:
                self.current_book.chapters[year] = NarrativeChapter(year=year, sections={})
            
            chapter = self.current_book.chapters[year]
            
            if month not in chapter.sections:
                chapter.sections[month] = NarrativeSection(
                    month=month, year=year, pages={}
                )
            
            section = chapter.sections[month]
            
            if page_num not in section.pages:
                start_day = 1 if page_num == 1 else 16
                end_day = 15 if page_num == 1 else 31
                section.pages[page_num] = NarrativePage(
                    page_num=page_num,
                    start_day=start_day,
                    end_day=end_day,
                    events=[]
                )
            
            event["timestamp"] = time.time()
            event["day"] = day
            event["source"] = source.type.value
            section.pages[page_num].events.append(event)
            self._save_current()
    
    def archive_month(self, year: int, month: int):
        if not self.current_book:
            return
        if year not in self.current_book.chapters:
            return
        
        chapter = self.current_book.chapters[year]
        if month not in chapter.sections:
            return
        
        section = chapter.sections[month]
        
        archive_data = {
            "book_id": self.current_book.book_id,
            "embodiment": self.current_book.embodiment_name,
            "year": year,
            "month": month,
            "pages": {}
        }
        
        for page_num, page in section.pages.items():
            archive_data["pages"][page_num] = {
                "start_day": page.start_day,
                "end_day": page.end_day,
                "events": page.events,
                "summary": page.summary
            }
        
        archive_path = self.base_path / f"archive_{year}_{month:02d}.json.gz"
        with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
            json.dump(archive_data, f, ensure_ascii=False, indent=2)
        
        with self._lock:
            for page in section.pages.values():
                page.events = []
                page.summary = ""
            self._save_current()
        
        logger.info(f"📚 Mois {year}-{month:02d} archivé")
    
    def query(self, query_type: str, constraints: Dict, context: Dict) -> Dict:
        return {"book": asdict(self.current_book) if self.current_book else None}
    
    def update(self, intent: StructuredIntent) -> bool:
        if intent.semantic.get("type") == IntentType.INFORMATION:
            source = SourceInfo(
                type=SourceWeight.OBSERVATION,
                speaker=intent.speaker
            )
            self.add_event({
                "intent_id": intent.id,
                "type": intent.semantic.get("sub_intent"),
                "attributes": {k: v.value for k, v in intent.attributes.items()}
            }, source)
            return True
        return False
    
    def consolidate(self) -> Dict:
        now = datetime.now()
        last_month = now.month - 1 if now.month > 1 else 12
        last_year = now.year if now.month > 1 else now.year - 1
        self.archive_month(last_year, last_month)
        return {"narrative_archived": f"{last_year}-{last_month:02d}"}
    
    def relevance(self, query_type: str, constraints: Dict) -> float:
        return 0.1  # Peu pertinent pour les requêtes directes
    
    def _save_current(self):
        if self.current_book:
            data = {
                "book_id": self.current_book.book_id,
                "embodiment_name": self.current_book.embodiment_name,
                "start_date": self.current_book.start_date,
                "end_date": self.current_book.end_date,
                "chapters": {}
            }
            
            for year, chapter in self.current_book.chapters.items():
                chapter_data = {
                    "year": chapter.year,
                    "sections": {},
                    "summary": chapter.summary,
                    "title": chapter.title
                }
                for month, section in chapter.sections.items():
                    section_data = {
                        "month": section.month,
                        "year": section.year,
                        "pages": {},
                        "summary": section.summary
                    }
                    for page_num, page in section.pages.items():
                        section_data["pages"][page_num] = {
                            "page_num": page.page_num,
                            "start_day": page.start_day,
                            "end_day": page.end_day,
                            "events": page.events,
                            "summary": page.summary
                        }
                    chapter_data["sections"][month] = section_data
                data["chapters"][year] = chapter_data
            
            self.storage.save_json("current_book", data)
    
    def _load_current(self):
        data = self.storage.load_json("current_book")
        if data:
            book = NarrativeBook(
                book_id=data["book_id"],
                embodiment_name=data["embodiment_name"],
                start_date=data["start_date"],
                end_date=data["end_date"],
                chapters={}
            )
            
            for year_str, chapter_data in data.get("chapters", {}).items():
                year = int(year_str)
                chapter = NarrativeChapter(
                    year=chapter_data["year"],
                    sections={},
                    summary=chapter_data.get("summary", ""),
                    title=chapter_data.get("title", "")
                )
                for month_str, section_data in chapter_data.get("sections", {}).items():
                    month = int(month_str)
                    section = NarrativeSection(
                        month=section_data["month"],
                        year=section_data["year"],
                        pages={},
                        summary=section_data.get("summary", "")
                    )
                    for page_num_str, page_data in section_data.get("pages", {}).items():
                        page_num = int(page_num_str)
                        section.pages[page_num] = NarrativePage(
                            page_num=page_data["page_num"],
                            start_day=page_data["start_day"],
                            end_day=page_data["end_day"],
                            events=page_data.get("events", []),
                            summary=page_data.get("summary", "")
                        )
                    chapter.sections[month] = section
                book.chapters[year] = chapter
            
            self.current_book = book

# ============================================================
# CONCEPT RESOLVER
# ============================================================

class ConceptResolver:
    """Résout les verbes en concepts en utilisant VerbMemory."""
    
    def __init__(self, verb_memory: VerbMemory):
        self.verb_memory = verb_memory
        self.pending_clarifications: List[Dict] = []
    
    def resolve(self, verb: str, context: Dict) -> Tuple[Concept, float, Optional[str]]:
        """Résout un verbe en concept."""
        return self.verb_memory.resolve(verb)
    
    def handle_unknown(self, verb: str, context: Dict) -> Optional[StructuredIntent]:
        """Gère un verbe inconnu en proposant une clarification."""
        # Chercher des verbes similaires
        similar = []
        for v, entry in self.verb_memory.verbs.items():
            if len(v) > 2 and (v in verb or verb in v):
                similar.append((v, entry.concept))
        
        if similar:
            # Proposer le plus similaire
            best = similar[0]
            return StructuredIntent(
                id=f"clar_{uuid.uuid4().hex[:8]}",
                timestamp=time.time(),
                conversation_id=context.get("conversation_id", ""),
                speaker="system",
                semantic={
                    "intent": "clarification",
                    "sub_intent": "verb_clarification",
                    "type": IntentType.CLARIFICATION,
                    "confidence": 0.8
                },
                attributes={
                    "verb": Attribute(
                        type="string",
                        value=verb,
                        source=SourceInfo(type=SourceWeight.OBSERVATION)
                    ),
                    "suggestion": Attribute(
                        type="string",
                        value=best[0],
                        source=SourceInfo(type=SourceWeight.OBSERVATION)
                    ),
                    "concept": Attribute(
                        type="string",
                        value=best[1].value,
                        source=SourceInfo(type=SourceWeight.OBSERVATION)
                    )
                }
            )
        
        return None

# ============================================================
# CURIOSITY ENGINE
# ============================================================

class CuriosityEngine:
    """Gère les connaissances manquantes."""
    
    def __init__(self, gem: Gem, concept_resolver: ConceptResolver):
        self.gem = gem
        self.concept_resolver = concept_resolver
        self.pending: List[Dict] = []
    
    def check_intent(self, intent: StructuredIntent) -> List[StructuredIntent]:
        """Vérifie un intent et retourne d'éventuelles clarifications."""
        clarifications = []
        
        # Vérifier les verbes inconnus
        if "verb" in intent.analysis:
            verb_info = intent.analysis["verb"]
            if verb_info.get("concept") == Concept.UNKNOWN.value:
                if self.gem.curiosite_verbes > self.gem.seuil_curiosite:
                    clarif = self.concept_resolver.handle_unknown(
                        verb_info.get("lemma", ""),
                        {"conversation_id": intent.conversation_id}
                    )
                    if clarif:
                        clarifications.append(clarif)
        
        # Vérifier les mots inconnus (à implémenter)
        # Vérifier les personnes inconnues (à implémenter)
        
        return clarifications

# ============================================================
# MOTEUR D'INFÉRENCE
# ============================================================

class InferenceEngine:
    """Moteur qui interroge les mémoires pour répondre."""
    
    def __init__(self, memories: Dict[str, MemoryInterface]):
        self.memories = memories
    
    def answer(self, intent: StructuredIntent) -> Optional[StructuredIntent]:
        """Produit une réponse à un intent."""
        
        if intent.semantic.get("type") != IntentType.QUESTION:
            return None
        
        sub_intent = intent.semantic.get("sub_intent")
        
        # Construire les requêtes en fonction du type de question
        if sub_intent == "time":
            # Réponse directe (pas besoin de mémoires)
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
                    ),
                    "hour": Attribute(
                        type="number",
                        value=now.hour,
                        source=SourceInfo(type=SourceWeight.OBSERVATION)
                    ),
                    "minute": Attribute(
                        type="number",
                        value=now.minute,
                        source=SourceInfo(type=SourceWeight.OBSERVATION)
                    )
                },
                in_response_to=intent.id
            )
        
        elif sub_intent == "person":
            # Chercher dans SocialMemory
            person_name = intent.attributes.get("person", {}).value if "person" in intent.attributes else None
            if person_name:
                result = self.memories["social"].query(
                    "person",
                    {"name": person_name},
                    {}
                )
                if result.get("person"):
                    return StructuredIntent(
                        id=f"resp_{uuid.uuid4().hex[:8]}",
                        timestamp=time.time(),
                        conversation_id=intent.conversation_id,
                        speaker="system",
                        semantic={
                            "intent": "answer",
                            "sub_intent": "person_info",
                            "type": IntentType.REPONSE,
                            "confidence": result["confidence"]
                        },
                        attributes={
                            "person": Attribute(
                                type="person",
                                value=result["person"],
                                source=SourceInfo(type=SourceWeight.OBSERVATION)
                            )
                        },
                        in_response_to=intent.id
                    )
        
        elif sub_intent == "fact":
            # Chercher dans EpisodicMemory par concept
            concept = intent.analysis.get("verb", {}).get("concept")
            if concept:
                result = self.memories["episodic"].query(
                    "by_concept",
                    {"concept": concept, "days": 30},
                    {}
                )
                if result.get("events"):
                    return StructuredIntent(
                        id=f"resp_{uuid.uuid4().hex[:8]}",
                        timestamp=time.time(),
                        conversation_id=intent.conversation_id,
                        speaker="system",
                        semantic={
                            "intent": "answer",
                            "sub_intent": "facts",
                            "type": IntentType.REPONSE,
                            "confidence": 0.8
                        },
                        attributes={
                            "events": Attribute(
                                type="list",
                                value=result["events"],
                                source=SourceInfo(type=SourceWeight.OBSERVATION)
                            )
                        },
                        in_response_to=intent.id
                    )
        
        return None

# ============================================================
# COGNITION CORE PRINCIPAL
# ============================================================

class CognitionCore:
    """
    Cœur cognitif principal - point d'entrée unique du système.
    
    Utilisation:
        core = CognitionCore(data_path="./data", gem_path="./gem.json")
        core.start()
        
        intent = core.process_text("demain midi, je vais au restaurant avec Paul")
        # intent est un StructuredIntent prêt pour le pipeline d'enrichissement
        
        response = core.answer(intent)  # Optionnel, si on veut répondre directement
    """
    
    def __init__(self, data_path: Path, gem_path: Path):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Charger le Gem (l'âme immuable)
        logger.info(f"📖 Chargement du Gem depuis {gem_path}")
        self.gem = Gem.from_file(gem_path)
        
        # Initialiser les chemins
        self.files_path = self.data_path / "files"
        self.db_path = self.data_path / "db"
        self.narrative_path = self.data_path / "narrative"
        
        self.files_path.mkdir(exist_ok=True)
        self.db_path.mkdir(exist_ok=True)
        self.narrative_path.mkdir(exist_ok=True)
        
        # Stockage compressé
        self.storage = CompressedStorage(self.files_path)
        
        # Initialiser toutes les mémoires
        self.verbs = VerbMemory(self.storage, self.gem)
        self.words = WordMemory(self.storage, self.gem)
        self.errors = ErrorMemory(self.db_path / "errors.db", self.gem)
        self.temporal = TemporalMemory(self.storage)
        self.social = SocialMemory(self.db_path / "social.db", self.gem)
        self.episodic = EpisodicMemory(self.db_path / "episodic.db", self.gem)
        self.romans = RomanMemory(self.storage, self.gem)
        self.educational = EducationalMemory(self.storage)
        self.narrative = NarrativeMemory(self.narrative_path, self.storage)
        
        # Registry des mémoires pour l'inférence
        self.memories = {
            "verbs": self.verbs,
            "words": self.words,
            "errors": self.errors,
            "temporal": self.temporal,
            "social": self.social,
            "episodic": self.episodic,
            "romans": self.romans,
            "educational": self.educational,
            "narrative": self.narrative
        }
        
        # Composants
        self.concept_resolver = ConceptResolver(self.verbs)
        self.curiosity = CuriosityEngine(self.gem, self.concept_resolver)
        self.inference = InferenceEngine(self.memories)
        
        # Contexte
        self.current_conversation: Optional[ContextFrame] = None
        self.system_context = SystemContext()
        
        # Thread de maintenance nocturne
        self._running = False
        self._maintenance_thread = None
        
        logger.info(f"✅ CognitionCore initialisé - Gem: {self.gem.nom} v{self.gem.version}")
    
    def start(self):
        """Démarre le thread de maintenance nocturne."""
        self._running = True
        self._maintenance_thread = threading.Thread(
            target=self._nightly_loop,
            name="nightly-maintenance",
            daemon=True
        )
        self._maintenance_thread.start()
        logger.info("✅ CognitionCore démarré")
    
    def stop(self):
        """Arrête le système."""
        self._running = False
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)
        logger.info("✅ CognitionCore arrêté")
    
    def _nightly_loop(self):
        """Boucle de maintenance nocturne."""
        while self._running:
            now = datetime.now()
            if now.hour == CONFIG["nightly_hour"] and now.minute < 5:
                logger.info("🌙 Début de la maintenance nocturne...")
                self.nightly_maintenance()
                time.sleep(3600)  # Attendre 1h pour ne pas refaire tout de suite
            time.sleep(60)  # Vérifier chaque minute
    
    def nightly_maintenance(self):
        """Cristallisation nocturne de toutes les connaissances."""
        logger.info("🌙 Cristallisation nocturne...")
        
        results = {}
        for name, memory in self.memories.items():
            try:
                res = memory.consolidate()
                results[name] = res
                logger.info(f"  ✓ {name}: {res}")
            except Exception as e:
                logger.error(f"  ✗ {name}: {e}")
        
        # Archiver le narratif
        self.narrative.consolidate()
        
        logger.info(f"🌙 Cristallisation terminée: {results}")
        return results
    
    def process_text(self, text: str, speaker: str = None) -> StructuredIntent:
        """
        Traite un texte brut et retourne un intent structuré.
        C'est le point d'entrée principal.
        """
        # Créer ou récupérer le contexte de conversation
        if not self.current_conversation or self.current_conversation.is_expired():
            self.current_conversation = ContextFrame(
                conversation_id=f"conv_{uuid.uuid4().hex[:8]}"
            )
        
        # Mettre à jour le contexte système
        self.system_context.refresh()
        
        # Créer l'intent de base
        intent_id = f"intent_{uuid.uuid4().hex[:8]}"
        intent = StructuredIntent(
            id=intent_id,
            timestamp=time.time(),
            conversation_id=self.current_conversation.conversation_id,
            speaker=speaker or "unknown",
            semantic={
                "intent": "unknown",  # Sera enrichi par le pipeline
                "sub_intent": "unknown",
                "type": IntentType.INFORMATION,
                "confidence": 0.5
            },
            attributes={
                "text": Attribute(
                    type="string",
                    value=text,
                    source=SourceInfo(type=SourceWeight.OBSERVATION, speaker=speaker)
                )
            },
            context={
                "conversation": asdict(self.current_conversation),
                "system": {
                    "now": self.system_context.now.isoformat(),
                    "here": self.system_context.here
                }
            }
        )
        
        # Vérifier les corrections d'erreurs
        for word in text.split():
            correction = self.errors.get(word)
            if correction:
                correct, conf, concept = correction
                intent.attributes[f"correction_{word}"] = Attribute(
                    type="correction",
                    value={"original": word, "corrected": correct},
                    source=SourceInfo(type=SourceWeight.EDUCATIVE, confidence=conf)
                )
        
        # Mettre à jour le contexte
        self.current_conversation.update(
            who=speaker,
            history=[intent_id]
        )
        
        logger.info(f"📥 Intent créé: {intent_id}")
        return intent
    
    def answer(self, intent: StructuredIntent) -> Optional[StructuredIntent]:
        """
        Produit une réponse à un intent.
        Retourne un intent de réponse ou None.
        """
        # Vérifier les clarifications nécessaires
        clarifications = self.curiosity.check_intent(intent)
        if clarifications:
            # Retourner la première clarification
            return clarifications[0]
        
        # Sinon, essayer de répondre
        return self.inference.answer(intent)
    
    def update_memories(self, intent: StructuredIntent):
        """Met à jour toutes les mémoires à partir d'un intent."""
        for name, memory in self.memories.items():
            try:
                memory.update(intent)
            except Exception as e:
                logger.error(f"Erreur mise à jour {name}: {e}")
    
    def get_stats(self) -> Dict:
        """Statistiques du système."""
        return {
            "gem": {
                "nom": self.gem.nom,
                "version": self.gem.version,
                "naissance": self.gem.date_naissance
            },
            "memories": {
                "verbs": len(self.verbs.verbs),
                "words": len(self.words.words),
                "romans": len(self.romans.romans),
                "educational": len(self.educational.works)
            }
        }

# ============================================================
# FICHIERS DE CONFIGURATION EXEMPLES
# ============================================================

"""
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
    
    # Créer un gem exemple si nécessaire
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
    
    # Initialiser le core
    core = CognitionCore(Path("./data"), gem_path)
    core.start()
    
    # Test simple
    test_phrase = "demain midi, je vais au restaurant avec Paul"
    print(f"\n📥 Test: {test_phrase}")
    
    intent = core.process_text(test_phrase, speaker="user_001")
    print(f"📤 Intent: {json.dumps(intent.to_dict(), indent=2, ensure_ascii=False)[:500]}...")
    
    # Simuler une réponse
    response = core.answer(intent)
    if response:
        print(f"📢 Réponse: {json.dumps(response.to_dict(), indent=2, ensure_ascii=False)}")
    
    # Statistiques
    print(f"\n📊 Stats: {json.dumps(core.get_stats(), indent=2)}")
    
    # Arrêt
    core.stop()