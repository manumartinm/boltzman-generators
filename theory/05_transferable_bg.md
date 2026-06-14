# 05 — Transferable Boltzmann Generators

Paper: Klein & Noé 2024, *"Transferable Boltzmann Generators"*, NeurIPS 2024. https://openreview.net/forum?id=AYq6GxxrrY

## 1. El problema con BG clásico

BG original (Noé 2019): un modelo `q_\theta(x)` por sistema. Cada vez que cambia la molécula (un átomo más, una mutación, un solvente distinto), hay que reentrenar. Caro y poco práctico para screening.

Análogo: imaginar tener que reentrenar AlphaFold desde cero para cada nueva proteína. Inviable.

## 2. Qué cambia en Transferable BG (TBG)

Dos ideas combinadas:

1. **Coordenadas cartesianas + flow matching** (Klein 2023 ya lo había explorado): en vez de coordenadas internas Z-matrix (que requieren topología fija), trabajar directamente con posiciones cartesianas `x ∈ R^{3N}`.
2. **Condicionar el campo `v_\theta` en la identidad molecular**: el modelo recibe no solo coordenadas, sino también los tipos atómicos, conectividad, cargas — toda la "topología" del sistema. Aprende un mapeo común a través del espacio químico.

Resultado: un único modelo entrenado en un conjunto de moléculas pequeñas (dipéptidos en el paper) genera muestras Boltzmann para moléculas **no vistas** — zero-shot.

## 3. Arquitectura

### 3.1 Equivariance E(3)

Las propiedades físicas son invariantes bajo:

- **Traslaciones** del centro de masa.
- **Rotaciones** del sistema entero.
- (Opcional) **Reflexiones** (E(3) vs SO(3)).

Si `v_\theta(R x + t, t) = R \cdot v_\theta(x, t)`, el flow respeta estas simetrías → no necesita aprender datos rotados como augmentation, generaliza mejor, y satisface las invariancias del Boltzmann.

Arquitecturas comunes:

- **EGNN** (Satorras et al. 2021) — equivariante, basada en distancias entre pares.
- **PaiNN** (Schütt et al. 2021) — vectores equivariantes + features escalares.
- **GVP** (Jing et al. 2021) — gated vector perceptron.

TBG usa una EGNN modificada para condicionar en tipos atómicos y producir un campo equivariante `v_\theta(x, t; \text{topology})`.

### 3.2 Tokenización molecular

Cada átomo se embebe como vector que codifica:

- Tipo (C, N, O, H, S, ...).
- Carga formal.
- Hibridación (sp², sp³).
- Pertenencia a residuo / grupo funcional.

Los enlaces se representan en el message-passing (edges del grafo molecular).

Esto es **idéntico al pipeline de los modernos modelos moleculares** (Boltz-2, RoseTTAFold-AA, AlphaFold3) — solo cambia el target (en TBG, regresión de campo vectorial; en AF, estructura).

## 4. Loss y entrenamiento

Loss base = CFM (conditional flow matching) con path lineal OT:

$$
\mathcal{L}_{TBG} = \mathbb{E}_{m \sim \text{molecules}, x_1 \sim p_m, t, z_0} \left\| v_\theta(x_t, t; m) - (x_1 - z_0) \right\|^2
$$

donde `m` es la molécula (con su topología codificada) y `p_m` es su Boltzmann.

Datos de entrenamiento: trayectorias MD de muchas moléculas distintas (dipéptidos = 20 × 20 = 400 combinaciones de aminoácidos, suficiente diversidad química).

## 5. Zero-shot inference

Para una molécula `m*` no vista:

1. Tokenizar `m*` con el mismo schema.
2. Muestrear `z_0 ~ N(0, I) ⊗ R^{3N*}`.
3. Integrar `dx/dt = v_\theta(x, t; m*)` desde `t=0` hasta `t=1`.
4. (Opcional) Reweighting con la energía MM de `m*` para corregir el sesgo residual.

El reweighting es crítico: TBG sin reweighting da una **aproximación** al Boltzmann; con reweighting da un estimador asintóticamente exacto siempre que el ESS no sea catastrófico.

## 6. Resultados clave del paper

- Entrenado en un subset de dipéptidos, evaluado en dipéptidos no vistos: distribuciones de Ramachandran (`φ, ψ`) muy próximas a las MD.
- ESS típico tras reweighting: ~50% en moléculas vistas, ~10–30% en no vistas (varía con la similitud química).
- Tiempo de sampling: órdenes de magnitud más rápido que MD para la misma calidad estadística (i.i.d.).

## 7. Limitaciones honestas

- Escala con `N` (átomos) limitada: dipéptidos hasta ~50 átomos, no proteínas enteras.
- Reweighting solo funciona si la energía MM es evaluable rápidamente → para QM no sirve directamente.
- Distribución de tamaños: si una molécula es mucho más grande que las del training, hay que re-evaluar.

## 8. Implicaciones para nuestro M10

M10 (opcional) en este repo intentaría:

a. Cargar trayectorias MD precomputadas de un dipéptido (alanine dipeptide es el canónico).
b. Entrenar un FM **no-equivariante** (MLP simple, ignora simetrías) para validar el pipeline.
c. Comparar Ramachandran del modelo vs MD.

Saltarse equivariance limita la calidad pero permite implementar todo sin GNN libraries pesadas. Ver `notebooks/07_transferable_dipeptide.ipynb` para la implementación.

## 9. Referencias

- Klein, Krämer, Noé 2023, *"Equivariant Flow Matching"*, NeurIPS — predecesor directo (mismo sistema, sin transferabilidad).
- Klein & Noé 2024 — TBG, transferabilidad.
- Köhler, Krämer, Noé 2020, *"Equivariant flows: exact likelihood generative learning for symmetric densities"*, ICML — fundamentos teóricos de equivariance + densidad exacta.
- Satorras, Hoogeboom, Welling 2021, *"E(n) Equivariant GNNs"*, ICML — backbone arquitectónico habitual.
