# PIVOT REFERENCE — Fractal, Audiomancer, Orbit, Dotfiles
## Document de référence pour Claude Code — Mars 2026

---

## CONTEXTE GLOBAL

Guillaume est joueur de poker pro basé à Annecy, avec un background en physique des particules (doctorat + postdoc). Il développe plusieurs projets tech en vibe coding (Claude Code + VS Code). Ce document décrit un pivot stratégique majeur suite à une analyse approfondie du marché et des politiques YouTube 2026.

### Projets concernés par ce pivot

| Projet | Repo actuel | Action |
|--------|-------------|--------|
| Fractal (audio) | `goliviero/fractal` | **PIVOT** → Fractal devient un générateur de visuels procéduraux |
| Audiomancer (nouveau) | `goliviero/audiomancer` | **NOUVEAU** — backup du code audio de Fractal, recentré en toolkit de scripts |
| Orbit | `goliviero/orbit` | **MISE À JOUR** — ajouter panneau Audiomancer, mettre à jour panneau Fractal |
| Dotfiles | `goliviero/dotfiles` | **MISE À JOUR** — refléter les changements dans la mémoire centrale |

### Motivation du pivot

- **Fractal audio** était un DAW headless Python (numpy/scipy/pedalboard/soundfile/audiocraft). Après état de l'art, le projet est redondant avec DawDreamer (1100+ stars, papier ISMIR, JUCE/C++, VST, Faust, JAX) et claw-daw (terminal-first MIDI DAW, agent-friendly).
- **Le vrai besoin** : produire du contenu audio ET visuel 100% original pour les chaînes YouTube de Guillaume (Akasha Portal = ambient/meditation, Element Zero = jeu vidéo, futures chaînes faceless).
- **YouTube 2026** fait la chasse au contenu "effortless" et "AI slop". Musique Suno + images Midjourney = ~55% risque de rejet monétisation. Contenu 100% original (piano/guitare/field recordings + visuels procéduraux codés) = ~15% risque.

---

## 1. FRACTAL — PIVOT VERS VISUELS PROCÉDURAUX

### Nouveau scope

Fractal devient un **générateur de visuels procéduraux en Python** pour produire des vidéos ambient/meditation (fractales, particules, attracteurs étranges, champs de bruit) destinées aux chaînes YouTube de Guillaume.

### Identité

- **Nom** : Fractal (inchangé — le nom colle encore mieux maintenant)
- **Tagline** : "Procedural visual generation toolkit for ambient content creation"
- **Repo** : `goliviero/fractal`
- **Stack technique** : Python + Blender API (prioritaire), avec explorations futures en GLSL/TouchDesigner

### Ce que Fractal FAIT (scope réduit et précis)

1. **Fractales animées** — Mandelbrot, Julia, attracteurs étranges (Lorenz, Rössler), rendus frame-by-frame ou via Blender
2. **Systèmes de particules** — Flux, nébuleuses, poussière cosmique, via Blender Python API + geometry nodes
3. **Champs de bruit** — Perlin noise, simplex noise, champs vectoriels animés
4. **Pipeline de rendu** — Scripts Python qui pilotent Blender en headless pour rendre en 4K/60fps via EEVEE ou Cycles
5. **Intégration NASA/JWST** — Scripts pour télécharger, traiter et utiliser comme backdrops les images domaine public de la NASA

### Ce que Fractal NE FAIT PAS

- Pas de DAW, pas de traitement audio (c'est Audiomancer)
- Pas de GUI, pas de webapp
- Pas de génération IA (pas de Stable Diffusion, pas de MidJourney dans le pipeline)
- Pas de streaming temps réel (rendu offline uniquement pour le moment)

### Architecture cible

```
fractal/
├── README.md
├── pyproject.toml
├── fractal/
│   ├── __init__.py
│   ├── core/
│   │   ├── fractals.py      # Mandelbrot, Julia, attracteurs
│   │   ├── particles.py     # Systèmes de particules 2D/3D
│   │   ├── noise.py         # Perlin, simplex, champs vectoriels
│   │   └── colors.py        # Palettes, gradients, color mapping
│   ├── blender/
│   │   ├── scene.py         # Setup scène Blender
│   │   ├── materials.py     # Matériaux procéduraux
│   │   ├── render.py        # Pipeline de rendu headless
│   │   └── geometry.py      # Geometry nodes helpers
│   ├── nasa/
│   │   ├── download.py      # Fetch images NASA/JWST
│   │   └── process.py       # Traitement et animation
│   └── export/
│       ├── frames.py        # Export frame-by-frame (PNG sequence)
│       └── video.py         # Assembly via ffmpeg
├── scripts/
│   ├── mandelbrot_zoom.py   # Exemple : zoom fractal animé
│   ├── particle_nebula.py   # Exemple : nébuleuse de particules
│   ├── lorenz_attractor.py  # Exemple : attracteur de Lorenz 3D
│   └── nasa_backdrop.py     # Exemple : image JWST + particules
├── renders/                  # Output (gitignored)
└── tests/
```

### Niveaux de difficulté (pour roadmap)

| Niveau | Description | Difficulté (pour Guillaume) | Résultat visuel |
|--------|-------------|----------------------------|-----------------|
| 1 - Python pur | numpy + PIL, rendu frame-by-frame, ffmpeg | 3/10 | 5/10 — 2D, basique |
| 2 - Blender API | Particules, geometry nodes, EEVEE/Cycles | 5/10 | 8/10 — Pro-grade |
| 3 - Shaders GLSL | Shadertoy, raymarching, fractales volumétriques | 7/10 | 10/10 — Époustouflant |

**Recommandation** : commencer par Niveau 1 (premiers scripts en 1-2 jours) puis monter vers Niveau 2 (Blender, 2-3 semaines pour être à l'aise).

### Dépendances

```
# Niveau 1 - Python pur
numpy
Pillow
matplotlib  # optionnel, pour preview
noise       # Perlin noise

# Niveau 2 - Blender
# Blender installé séparément (gratuit)
# Scripts exécutés via : blender --background --python script.py

# Export
# ffmpeg installé séparément
```

---

## 2. AUDIOMANCER — TOOLKIT AUDIO SCRIPTS JETABLES

### Concept

Audiomancer est le **backup et pivot** du code audio de l'ancien Fractal. Ce n'est PAS un framework ni un DAW. C'est une collection de scripts Python utilitaires pour produire des sons et textures audio pour les projets YouTube et le jeu vidéo de Guillaume.

### Identité

- **Nom** : Audiomancer
- **Tagline** : "Minimal Python audio scripts for ambient sound design"
- **Repo** : `goliviero/audiomancer` (nouveau)
- **Philosophie** : scripts jetables > framework maintenable. Chaque script = un son ou une texture. Pas d'architecture ambitieuse.

### Ce que Audiomancer FAIT

1. **Synthèse basique** — Génération d'ondes (sinus, saw, square), drones, pads via scipy/numpy
2. **Binauraux** — Génération de beats binauraux avec fréquences configurables (déjà fonctionnel)
3. **Traitement audio** — Chaînes d'effets via pedalboard (Spotify) : reverb, delay, chorus, compression
4. **Field recording processing** — Nettoyage, normalisation, effets sur des enregistrements terrain
5. **Layering** — Superposition de stems audio avec contrôle de volume et fade in/out

### Ce que Audiomancer NE FAIT PAS

- Pas de DAW (utiliser REAPER pour ça)
- Pas de VST hosting (utiliser DawDreamer pour ça)
- Pas de génération IA (pas d'AudioCraft, pas de Suno)
- Pas de MIDI (utiliser REAPER + Vital pour ça)
- Pas de temps réel

### Architecture cible

```
audiomancer/
├── README.md
├── pyproject.toml
├── audiomancer/
│   ├── __init__.py
│   ├── synth.py           # Synthèse basique (ondes, drones, pads)
│   ├── binaural.py        # Beats binauraux (code existant de Fractal)
│   ├── effects.py         # Wrappers pedalboard (reverb, delay, etc.)
│   ├── layers.py          # Layering et mixage de stems
│   └── utils.py           # I/O audio, normalisation, fade
├── scripts/
│   ├── make_binaural.py   # Script : générer un binaural 432Hz
│   ├── drone_pad.py       # Script : créer un drone ambient
│   ├── process_field.py   # Script : traiter un field recording
│   └── layer_stems.py     # Script : superposer audio + binaural + field
├── output/                 # Fichiers WAV générés (gitignored)
└── tests/
```

### Dépendances

```
numpy
scipy
soundfile
pedalboard    # Spotify, effets audio pro-grade
```

### Migration depuis Fractal audio

Lors de la création du repo Audiomancer :
1. Copier TOUT le contenu actuel de `fractal/` vers `audiomancer/` (backup intégral)
2. Réorganiser selon l'architecture ci-dessus
3. Supprimer tout ce qui est hors scope (tentatives de DAW, routing complexe, etc.)
4. Garder et nettoyer : synthèse basique, binauraux, effets pedalboard
5. Ajouter un README clair qui explique le pivot

---

## 3. ORBIT — MISE À JOUR DASHBOARD

### Changements requis

Orbit est le dashboard interactif personnel de Guillaume. Il doit refléter les changements de projets.

### Modifications

1. **Renommer le panneau "Fractal"** — mettre à jour la description : "Générateur de visuels procéduraux" au lieu de "DAW headless Python"
2. **Ajouter un panneau "Audiomancer"** — nouveau projet avec description : "Scripts audio Python pour sound design ambient"
3. **Mettre à jour les statuts** :
   - Fractal : status = "active", type = "visual generation"
   - Audiomancer : status = "active", type = "audio toolkit"
4. **Mettre à jour les liens repo** si les URLs changent

### Données du panneau Audiomancer

```json
{
  "name": "Audiomancer",
  "description": "Minimal Python audio scripts for ambient sound design",
  "status": "active",
  "repo": "goliviero/audiomancer",
  "stack": ["Python", "numpy", "scipy", "pedalboard", "soundfile"],
  "scope": "Scripts jetables pour synthèse, binauraux, effets, layering",
  "priority": "medium",
  "linked_to": ["Akasha Portal", "Element Zero"]
}
```

### Données mises à jour du panneau Fractal

```json
{
  "name": "Fractal",
  "description": "Procedural visual generation toolkit for ambient content creation",
  "status": "active",
  "repo": "goliviero/fractal",
  "stack": ["Python", "Blender API", "numpy", "Pillow", "ffmpeg"],
  "scope": "Fractales, particules, noise fields, rendu 4K pour YouTube",
  "priority": "high",
  "linked_to": ["Akasha Portal", "Element Zero"]
}
```

---

## 4. DOTFILES — MÉMOIRE CENTRALE

### Changements requis

Les dotfiles servent de mémoire centrale. Mettre à jour les sections suivantes :

### Sections à modifier

1. **Projets actifs** — Mettre à jour la liste :
   - Fractal : "Générateur de visuels procéduraux (Python + Blender)" au lieu de "DAW headless Python"
   - Audiomancer : ajouter comme nouveau projet actif
   - Supprimer toute mention de "Fractal audio" ou "DAW headless" dans les descriptions

2. **Stack créative** — Mettre à jour :
   - Retirer AudioCraft de la stack (pas d'IA générative audio)
   - Ajouter Blender comme outil visuel principal
   - Ajouter REAPER + Vital comme outils audio (DAW gratuit + synthé gratuit)
   - Garder pedalboard, scipy, numpy, soundfile

3. **Philosophie créative** — Ajouter/modifier :
   - "Pas de musique IA générative (Suno, AudioCraft) pour le contenu monétisé"
   - "Visuels procéduraux codés > images IA (MidJourney limité à l'inspiration)"
   - "Contenu 100% original et revendicable pour YouTube"

4. **Pipeline Akasha Portal** — Mettre à jour :
   - Audio : piano P-45 + guitare F310 + field recordings + binauraux maison + Freesound CC0 + REAPER/Vital
   - Visuels : Fractal (procédural codé) + NASA/JWST (domaine public) + photos/vidéos perso
   - Retirer : Suno de la stack audio, MidJourney des visuels finaux

---

## 5. STACK DE PRODUCTION AKASHA PORTAL (RÉFÉRENCE)

### Audio — par ordre de difficulté

| # | Source | Difficulté | Risque YT | Outils | Notes |
|---|--------|-----------|-----------|--------|-------|
| 1 | Field recordings téléphone | 1/10 | 0% | Téléphone + Audacity | Eau, oiseaux, vent, cloches d'Annecy |
| 2 | Piano P-45 + effets | 2/10 | 0% | P-45 USB-MIDI → REAPER → Valhalla Supermassive (gratuit) | Accords lents + reverb massive = ambient |
| 3 | Guitare F310 drone | 2/10 | 0% | F310 + micro/DI → REAPER → reverb/delay | Notes ouvertes, sustain, optionnel e-bow (40-60€) |
| 4 | Binauraux Python | 2/10 | 0% | Audiomancer (scipy) | Ondes sinusoïdales, déjà fonctionnel |
| 5 | Freesound CC0 | 1/10 | ~2% | freesound.org | Bols tibétains, gongs — vérifier chaque licence |
| 6 | Synthèse Python | 3/10 | 0% | Audiomancer (scipy + pedalboard) | Drones, pads, textures par code |
| 7 | REAPER + VST gratuits | 4/10 | 0% | REAPER + Vital + Surge XT + Dexed | Game changer — synthèse complète |
| 8 | Granular/spectral | 6/10 | 0% | Paulstretch (Audacity) ou plugins | Transformer field recordings en nappes cosmiques |

### Visuels — par ordre de difficulté

| # | Source | Difficulté | Risque YT | Outils | Notes |
|---|--------|-----------|-----------|--------|-------|
| 1 | Images NASA/JWST animées | 1/10 | ~5% | NASA.gov → DaVinci Resolve (gratuit) | Domaine public, animer en parallax/zoom |
| 2 | Timelapses smartphone | 2/10 | 0% | Téléphone actuel | Lac d'Annecy, montagnes, ciel |
| 3 | Fractales 2D Python pur | 3/10 | 0% | Fractal (numpy + PIL + ffmpeg) | Mandelbrot, Julia, attracteurs |
| 4 | Blender + Python API | 5/10 | 0% | Fractal + Blender | Particules 3D, nébuleuses, geometry nodes |
| 5 | TouchDesigner | 6/10 | 0% | TouchDesigner (gratuit non-commercial) | Visuels audio-réactifs temps réel |
| 6 | Shaders GLSL | 7/10 | 0% | Shadertoy / custom | Fractales volumétriques, raymarching |

### Visuels — CE QU'ON ÉVITE pour la monétisation

- **MidJourney** : OK pour inspiration/moodboard, PAS dans les vidéos finales
- **Stable Diffusion / DALL-E** : idem
- **Templates stock / footage stock** : risque "reused content"
- **Slideshows d'images statiques** : flag YouTube instantané en 2026

---

## 6. INSTRUCTIONS POUR CLAUDE CODE

### Ordre d'exécution recommandé

1. **Créer le repo `audiomancer`** sur GitHub
2. **Copier** tout le contenu actuel de `fractal/` vers `audiomancer/` (backup intégral)
3. **Réorganiser** audiomancer selon l'architecture décrite en section 2
4. **Vider** le repo `fractal/` (garder le .git pour l'historique)
5. **Restructurer** fractal selon l'architecture décrite en section 1
6. **Écrire les README** des deux projets
7. **Mettre à jour Orbit** selon section 3
8. **Mettre à jour Dotfiles** selon section 4
9. **Commit et push** chaque repo séparément avec des messages clairs

### Convention de commit

```
feat: pivot fractal from audio DAW to procedural visual generation
feat: create audiomancer repo from fractal audio backup
feat(orbit): add audiomancer panel, update fractal description
feat(dotfiles): update project registry and creative stack
```

### Notes importantes

- Ne PAS supprimer l'historique git de Fractal — le pivot doit être traçable
- Le code audio existant dans Fractal peut avoir des bugs/être incomplet — c'est OK, Audiomancer reprend tout tel quel puis nettoie
- Les exemples/scripts dans les deux repos doivent être fonctionnels en <10 lignes de code (preuve de concept)
- Pas de sur-engineering — scripts simples > architecture complexe

---

## 7. RÉSUMÉ DÉCISIONNEL

| Décision | Raison |
|----------|--------|
| Fractal pivote de audio → visuel | DawDreamer fait déjà tout en mieux. Le nom "Fractal" colle parfaitement aux visuels procéduraux. |
| Audiomancer créé comme backup + toolkit | Le code audio existant a de la valeur, mais comme scripts utilitaires, pas comme framework. |
| Pas de Suno dans Akasha | Risque Content ID + risque "reused content" + procès majors + pas aligné avec les valeurs de Guillaume. |
| Pas de MidJourney dans les vidéos finales | YouTube flag les slideshows d'images IA. OK pour inspiration uniquement. |
| Piano/guitare/field recordings comme base audio | 100% original, 0% risque copyright, storytelling authentique ("enregistré au lac d'Annecy à l'aube"). |
| Blender + Python comme base visuelle | Gratuit, puissant, rendu 4K pro, scriptable, communauté massive. Parfait match avec le profil de Guillaume. |
| REAPER + Vital comme DAW | Gratuit/donationware, VST gratuits de qualité pro. Pas besoin de coder un DAW. |
| Scope réduit partout | 5 projets en parallèle = risque de n'en finir aucun. Scope minimal = livraison réelle. |

---

*Document généré le 25 mars 2026. Source : analyse Claude avec Guillaume.*
