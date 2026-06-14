# boltzmann_generators

Personal study repo for **Boltzmann Generators**.

Material de referencia:

1. **Original** — Noé, Olsson, Köhler, Wu, *"Boltzmann Generators: Sampling Equilibrium States of Many-Body Systems with Deep Learning"*, Science 2019 (arXiv 1812.01729). Repo: https://github.com/noegroup/paper_boltzmann_generators
2. **Transferable** — Klein & Noé, *"Transferable Boltzmann Generators"*, NeurIPS 2024. https://openreview.net/forum?id=AYq6GxxrrY

Aproximación: teoría primero (markdown notes), implementación propia en PyTorch desde cero, validación contra el repo original.

## Roadmap

- [x] **M1** — Setup folder, uv project, clonar repo de referencia
- [x] **M2** — Teoría: statistical mechanics + normalizing flows (`theory/01`, `theory/02`)
- [x] **M3** — RealNVP desde cero (`src/bg/flows/`, notebook 02 — Gaussian mixture, NLL 2.23 nats vs entropía 2.13)
- [x] **M4** — Teoría: Boltzmann Generators (`theory/03`)
- [x] **M5** — BG en double-well (notebook 03, ESS 98%, ΔF=+0.004 kT vs true 0)
- [x] **M6** — BG en Müller-Brown (notebook 04, ESS 96%, populations within 0.2% de ground truth)
- [x] **M7** — Diff vs repo original (notebook 05, ESS 92%)
- [x] **M8** — Teoría: flow matching + transferable BG (`theory/04`, `theory/05`)
- [x] **M9** — Flow matching desde cero (`src/bg/flows/cnf.py`, notebook 06)
- [x] **M10** — Dipeptide sintético (notebook 07, Ramachandran-flavored, sin OpenMM)

Notebook 01 (`01_toy_double_well.ipynb`) del plan original quedó absorbido por el 03 — la exploración del double-well + MCMC baseline está integrada allí.

## Layout

```
theory/      notas markdown por tema
src/bg/      paquete propio
  flows/     base + RealNVP + CNF
  energies/  double-well, Müller-Brown, Ramachandran dipeptide
  losses.py  KL_x, KL_z, mixed
  sampling.py  sample + reweight + ESS + ΔF
notebooks/   experimentos numerados (02–07)
references/  PDFs + clone del repo de Noé
data/        trayectorias/checkpoints (gitignored)
```

## Setup

```sh
uv sync                       # instala deps + bg en modo editable
uv run jupyter lab            # arranca notebooks
```

Si `references/paper_boltzmann_generators/` no existe (gitignored):

```sh
git clone --depth 1 https://github.com/noegroup/paper_boltzmann_generators.git references/paper_boltzmann_generators
```

## Stack

PyTorch 2.12 (MPS en Apple Silicon, CUDA en NVIDIA), numpy, matplotlib, jupyterlab, pandas. **Sin dependencias de MD** — todo el roadmap usa potenciales analíticos 2D. `torchdiffeq` NO fue necesario: el CNF en `bg.flows.cnf` usa integración Euler manual (suficiente para 2D). Para escalar a más dimensiones, considerar añadirlo.

## Resultados destacados

| Notebook | Sistema | Métrica | Valor |
|---|---|---|---|
| 02 | 4-mode Gaussian mixture | NLL final | 2.23 nats (entropía 2.13) |
| 03 | Double-well 2D (barrier 4 kT) | ESS / ΔF | 98% / +0.004 kT (true 0) |
| 04 | Müller-Brown (3 minima) | ESS / pops | 96% / true [80.37, 7.74, 11.89]%, BG [80.23, 7.71, 12.05]% |
| 05 | Double-well (Fig 2 paper) | ESS | 92% |
| 06 | Gaussian mixture (CFM) | Visual | 4 modos cubiertos vía ODE Euler |
| 07 | Ramachandran sintético | Visual + pops | BG-reweighted ≈ ground truth en 4 basins |

## Para continuar

- **M10b** — periodicity correcta vía embedding angular (notebook 07 sección 7).
- **M10c** — trayectorias MD reales de alanine dipeptide vía `mdshare` (sin OpenMM).
- E(3)-equivariant flows (Köhler 2020, Klein 2023).
- FAB / Annealed Importance Sampling Bootstrap (Midgley 2023) — mejora reweighting.
