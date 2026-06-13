# repo-bab4

Dokumen Bab IV dan Bab V akan disimpan di repository ini.

## Kerr black hole animation (`black-hole.html`)

A real-time, physically based **rotating (Kerr) black hole** rendered with
[Three.js](https://threejs.org/). Every pixel's light ray is integrated as a
null geodesic through curved spacetime (Hamiltonian formulation of the Kerr
metric in Boyer–Lindquist coordinates, RK4), so the accretion disk lenses up
and over the shadow and frame-dragging warps the shadow into its characteristic
asymmetric shape.

Features:
- Kerr geodesic ray tracing (spin a/M = 0.84), event horizon + photon ring + shadow
- Accretion disk with relativistic **Doppler beaming** + **gravitational redshift**
- Procedural lensed starfield background
- **OrbitControls** — drag to orbit, scroll to zoom, right-drag to pan
- HDR **UnrealBloom** glare, ACES tonemapping, gentle auto-rotation

### Running

Just open **`black-hole.html`** in a modern browser (WebGL2). It loads the
pre-built `black-hole.bundle.js`, so it works straight from disk — no server
needed.

### Rebuilding the bundle

The bundle (Three.js + addons + the app in `src/black-hole.js`) is produced with:

```bash
npm install
npm run build
```
