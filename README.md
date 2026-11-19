# AssetForge

Editable, game-ready 3D models from a single photo.

AssetForge lets you sketch over a photo or concept drawing to build a 3D model out of
separate, parametric parts (based on the 3-Sweep method, Chen et al., SIGGRAPH Asia 2013).
Each part keeps its cross-section, axis and size, so edits are exact and local.
AI helps with tedious steps (object selection, part suggestions) but never overwrites your work.

Bilkent University · CS491/492 Senior Design Project · Team Sprite
Website: https://assetforge.fyrkn.com

## Team
- Defne Lal Oruncak
- Leyla Özpınar
- Can Akpınar
- Fevzi Yarkın Müftüoğlu

Supervisor: Yusuf Sahillioğlu · Innovation Expert: Cem Çimenbiçer

## Getting started
    git clone https://github.com/assetforge-bilkent/assetforge.git
    cd assetforge
    python3.11 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Repository layout
    assetforge/core     geometry: sweep, frames, mesher, camera, part graph
    assetforge/image    masks, edges, profile fitting
    assetforge/ui       2D canvas and 3D viewer
    assetforge/export   OBJ / glTF export, LOD, colliders
    tests/              unit tests (pytest)
    data/images/        test photos
    docs/               reports and meeting notes

## Workflow
Never commit to `main`. Create a branch per task, open a pull request,
get one approval, then squash-merge.
