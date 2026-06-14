# 03 — Boltzmann Generators

Paper: Noé, Olsson, Köhler, Wu, *"Boltzmann Generators: Sampling Equilibrium States of Many-Body Systems with Deep Learning"*, Science **365**, eaaw1147 (2019). arXiv: 1812.01729.

## 1. Idea en una frase

Entrenar un normalizing flow `q_\theta(x) = q_Z(f_\theta^{-1}(x)) |\det J_{f_\theta^{-1}}(x)|` para que aproxime la distribución de Boltzmann `p(x) = e^{-u(x)} / Z`, **usando solo la función de energía `u(x)`** (que sabemos evaluar) y opcionalmente algunas muestras MD/MCMC.

Una vez entrenado:

1. Generar `z ~ N(0,I)`, `x = f_\theta(z)` → muestras i.i.d. en una pasada. No hay barreras temporales.
2. Reweighting con `w(x) = e^{-u(x)} / q_\theta(x)` corrige el sesgo residual y rinde estimadores exactos para cualquier observable, incluyendo diferencias de energía libre.

## 2. Las dos losses

### 2.1 `KL_x` (forward KL, "KL by example")

Dadas muestras `{x_i}` del target (vía MD, por ejemplo):

$$
\mathcal{L}_{KL_x} = \mathrm{KL}(p \,\|\, q_\theta) = -\mathbb{E}_{x \sim p}[\log q_\theta(x)] + \mathrm{const}
$$

Es la NLL estándar de un flow entrenado por MLE. Propiedades:

- **Mode-covering:** penaliza fuertemente regiones donde `p > 0` y `q_\theta ≈ 0`. El modelo intenta cubrir todos los modos vistos en los datos.
- Necesita muestras — replica los sesgos del simulador (si MD no cruzó una barrera, esa región no se entrena).
- Es lo que hicimos en el notebook 02.

### 2.2 `KL_z` (reverse KL, "KL by energy")

No necesita muestras del target; solo necesita evaluar `u(x)`. La derivación clave:

$$
\mathrm{KL}(q_\theta \,\|\, p) = \mathbb{E}_{x \sim q_\theta}\!\left[\log q_\theta(x) - \log p(x)\right]
$$

Como `log p(x) = -u(x) - \log Z`:

$$
\mathrm{KL}(q_\theta \,\|\, p) = \mathbb{E}_{x \sim q_\theta}[\log q_\theta(x) + u(x)] + \log Z
$$

`log Z` es constante respecto a `\theta` y desaparece del gradiente. Y como muestrear `x ~ q_\theta` equivale a muestrear `z ~ N(0,I)` y aplicar `x = f_\theta(z)`, con `log q_\theta(x) = \log q_Z(z) - \log|\det J_{f_\theta}(z)|`:

$$
\mathcal{L}_{KL_z} = \mathbb{E}_{z \sim N(0,I)}\!\left[ u(f_\theta(z)) - \log|\det J_{f_\theta}(z)| \right] + \mathrm{const}
$$

(El término `\log q_Z(z)` también es constante respecto a `\theta`.)

**Esto es magia:** entrenamos un modelo generativo sin un solo sample del target. La loss es totalmente diferenciable en `\theta` siempre que `u` lo sea (lo es: viene de un force field analítico).

Propiedades:

- **Mode-seeking:** penaliza regiones donde `q_\theta > 0` y `p ≈ 0` (zonas de alta energía). Tendencia a **colapsar en un solo modo** si la geometría no ayuda.
- No requiere datos. Pero puede dar samples sesgados (un modo dominante).
- Suele combinarse con `KL_x` para evitar el colapso.

### 2.3 Loss combinada

$$
\mathcal{L} = w_{ML} \cdot \mathcal{L}_{KL_x} + w_{KL} \cdot \mathcal{L}_{KL_z} + w_{RC} \cdot \mathcal{L}_{RC}
$$

`L_{RC}` es opcional (reaction-coordinate loss, paper Sec. S5) — fuerza al modelo a poblar regiones específicas del espacio (útil cuando se conoce una coordenada relevante).

Estrategia típica del paper:

1. **Pretrain** con `KL_x` puro (necesita algunos samples MD/MCMC, aunque sea de un solo basin).
2. **Fine-tune** con `KL_z` añadido, peso pequeño al principio y subido gradualmente.
3. (Opcional) Añadir `RC` para forzar exploración cuando se conoce la geometría.

En notebooks 03–04 entrenaremos puramente con `KL_z` (más limpio pedagógicamente, sistemas pequeños donde el colapso es manejable).

## 3. Reweighting

Con un modelo entrenado `q_\theta`, las muestras `x_i ~ q_\theta` no se distribuyen exactamente como `p`. Para estimar observables en el Boltzmann exacto:

$$
\langle O \rangle_p \approx \frac{\sum_i w_i \, O(x_i)}{\sum_i w_i}, \qquad w_i = \frac{e^{-u(x_i)}}{q_\theta(x_i)}
$$

Notar:

- No necesitamos `Z` (se cancela en la normalización).
- Calidad medida por **Effective Sample Size (ESS)**:

$$
\mathrm{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2} \in [1, N]
$$

Si `q ≈ p`, ESS ≈ N (todos los pesos similares). Si `q` está muy lejos, ESS → 1 (un solo sample domina). ESS/N es el indicador de calidad estándar.

## 4. Estimación de diferencias de energía libre

Dadas dos regiones `A, B ⊂ R^d` (por ejemplo, dos cuencas de un double-well):

$$
\Delta F_{AB} = -k_B T \log \frac{Z_B}{Z_A} = -k_B T \log \frac{\int_B e^{-u(x)} dx}{\int_A e^{-u(x)} dx}
$$

Con samples reweighted de `q_\theta`:

$$
\Delta F_{AB} \approx -k_B T \log \frac{\sum_{i: x_i \in B} w_i}{\sum_{i: x_i \in A} w_i}
$$

Comparado con métodos clásicos (Bennett Acceptance Ratio, Free Energy Perturbation, thermodynamic integration), BG no necesita estados intermedios ni trayectorias de no-equilibrio.

## 5. Mode-seeking vs mode-covering: intuición geométrica

Considerar `p` con dos modos bien separados, ambos de igual masa. Si `q` solo cubre un modo:

- `KL(p || q)` (forward) = ∞ (porque en el otro modo `p>0` y `q≈0`).
- `KL(q || p)` (reverse) = pequeña (porque donde `q>0`, `p` también lo es).

Reverse-KL **acepta** soluciones que ignoran modos. Por eso `KL_z` solo es peligroso. Hace falta:

- Pretrain con muestras (forward) que cubran todos los modos relevantes, OR
- Arquitecturas con expresividad suficiente y muchos pasos, OR
- Loss de exploración explícita (RC, replicas a temperatura alta, etc.).

## 6. Comparación con otros métodos generativos para muestreo

| Método | Necesita muestras | Necesita energía | Densidad exacta | Sample i.i.d. |
|---|---|---|---|---|
| MD/MCMC | — | sí | no (implícita) | no (correlacionados) |
| VAE | sí | no | no (ELBO) | sí |
| GAN | sí | no | no (implícita) | sí |
| Score / Diffusion | sí (para score matching) | opcional | no (ratio) | sí (caro) |
| **Normalizing Flow / BG** | **opcional** | **sí o no** | **sí** | **sí** |

Es la combinación de **densidad exacta + sample i.i.d.** lo que permite el reweighting exacto.

## 7. Lo que añade el paper original (Sec. 4)

Aplicaciones demostradas:

- **Sistemas de partículas 2D condensadas** (Lennard-Jones-like): exploran múltiples estructuras cristalinas.
- **Polímeros condensados (BPTI-like en CG)**: campan los free-energy minima difíciles de cruzar por MD.
- **Estimación de ΔF entre fases**: error <1 `k_B T` vs métodos clásicos en órdenes de magnitud menos compute.

## 8. Limitaciones (que motivan papers posteriores)

- Un modelo por sistema → **Transferable BG** (Klein & Noé 2024, M8).
- Coordenadas internas (paper original) limitan geometrías → flow matching en cartesianas (Klein 2023+).
- Escala mal a sistemas grandes → equivariant architectures (E(3)-equivariant flows, Köhler et al.).

## 9. Referencias

- Noé, Olsson, Köhler, Wu 2019 — paper canónico.
- Wirnsberger, Ballard, Papamakarios et al. 2020 — BG para sólidos cristalinos.
- Midgley, Stimper, Simm, Schölkopf, Hernández-Lobato 2023 — "Flow Annealed Importance Sampling Bootstrap" (FAB), mejora la calidad del reweighting.
