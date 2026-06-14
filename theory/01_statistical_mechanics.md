# 01 — Statistical mechanics: el problema que BG resuelve

## 1. La distribución de Boltzmann

En el ensemble canónico (NVT: número de partículas, volumen, temperatura fijos), la probabilidad de encontrar el sistema en una configuración `x` (coordenadas de todos los átomos) está dada por:

$$
p(x) = \frac{e^{-U(x) / (k_B T)}}{Z}, \qquad Z = \int e^{-U(x) / (k_B T)} \, dx
$$

donde:

- `U(x)` es la energía potencial (función conocida, dada por el force field clásico o por un Hamiltoniano cuántico aproximado).
- `k_B` es la constante de Boltzmann, `T` la temperatura.
- `Z` es la **función de partición** — normaliza la distribución.

Conviene introducir la **energía reducida** `u(x) = U(x) / (k_B T)` (adimensional), con lo que:

$$
p(x) = \frac{e^{-u(x)}}{Z}
$$

A partir de ahora `u(x)` es siempre adimensional. Esta es la convención que el paper de Noé (2019) usa y que adoptaremos en el código.

## 2. Por qué `Z` es intratable

`Z` es una integral sobre el espacio de configuraciones. Para un sistema con `N` átomos en 3D, `x ∈ R^{3N}`. Para una proteína pequeña `N ~ 100` átomos → integral en `R^{300}`. Cuadratura clásica imposible (10^k puntos por dimensión = explosión combinatoria).

Consecuencia práctica: **podemos evaluar `e^{-u(x)}` puntualmente, pero no la normalización**. Es decir, conocemos `p(x)` solo hasta una constante multiplicativa desconocida.

Esto bloquea:

- Cálculo directo de medias `<O> = ∫ O(x) p(x) dx`.
- Cálculo de energía libre `F = -k_B T \log Z`.
- Cálculo de diferencias `ΔF = F_B - F_A = -k_B T \log(Z_B / Z_A)` entre estados.

## 3. Cómo se resuelve en la práctica (sin ML)

### 3.1 Molecular Dynamics (MD)

Integramos las ecuaciones de Newton con un termostato (Langevin, Nosé–Hoover). En equilibrio, la trayectoria muestrea la distribución de Boltzmann. Estimamos medias como promedios temporales.

Problema: en sistemas con **barreras energéticas altas** (transiciones conformacionales, plegamiento de proteínas, isomerización), la trayectoria queda atrapada en un mínimo. El tiempo de cruce entre cuencas escala como `exp(ΔU* / k_B T)` (Kramers / Arrhenius). Para barreras de 10 `k_B T`, cruces cada ~22000 pasos en promedio. Para 30 `k_B T`, prácticamente nunca.

### 3.2 MCMC (Metropolis–Hastings)

Propuesta aleatoria `x' ~ q(x' | x)`, aceptación con probabilidad `min(1, e^{-(u(x') - u(x))})`. La constante `Z` se cancela. En equilibrio muestrea `p(x)`.

Mismo problema: cadenas locales tardan exponencialmente en cruzar barreras. Se usan trucos (parallel tempering, umbrella sampling, metadynamics) que requieren conocer a priori las coordenadas relevantes.

### 3.3 Importance sampling y reweighting

Si tenemos muestras de una distribución auxiliar `q(x)` (fácil de muestrear), podemos estimar medias en `p`:

$$
\langle O \rangle_p = \int O(x) \, p(x) \, dx = \int O(x) \frac{p(x)}{q(x)} q(x) \, dx = \mathbb{E}_{x \sim q}\!\left[ O(x) \, w(x) \right]
$$

con peso `w(x) = p(x) / q(x)`. Para `p` no normalizada se usa la versión auto-normalizada:

$$
\langle O \rangle_p \approx \frac{\sum_i O(x_i) \tilde{w}(x_i)}{\sum_i \tilde{w}(x_i)}, \qquad \tilde{w}(x) = \frac{e^{-u(x)}}{q(x)}
$$

**Este es el truco clave que BG explota**: si entrenamos un modelo generativo `q_\theta(x)` cuya densidad **podemos evaluar exactamente**, podemos generar muestras i.i.d. (sin barreras temporales) y corregir el sesgo via reweighting hacia el Boltzmann exacto.

Restricción crítica: `q_\theta(x)` debe permitir **(a) muestreo eficiente** y **(b) cálculo exacto de la densidad** `q_\theta(x)`. VAEs y GANs fallan en (b). Los **normalizing flows** cumplen ambos requisitos → vehículo natural para BG.

## 4. Energía libre y diferencias entre estados

La energía libre de Helmholtz es:

$$
F = -k_B T \log Z
$$

Para dos estados A y B (definidos por restricciones a regiones distintas del espacio de configuraciones):

$$
\Delta F_{AB} = F_B - F_A = -k_B T \log \frac{Z_B}{Z_A}
$$

`ΔF` controla equilibrios de binding, plegamiento, cambios de fase. Su cálculo es el santo grial de la termodinámica computacional.

Con muestras y energías exactas, BG estima `ΔF` directamente vía reweighting entre las dos regiones, sin necesidad de simulaciones de no equilibrio (Jarzynski) ni de coordenadas colectivas predefinidas.

## 5. Conexión con el resto de la teoría

- **Cap. 02 (flows):** la maquinaria para construir `q_\theta(x)` con densidad tratable.
- **Cap. 03 (BG):** las dos losses (`KL_x` con muestras MD, `KL_z` con energía) y cómo se combinan.
- **Cap. 04 (flow matching):** alternativa moderna a coupling flows.
- **Cap. 05 (transferable):** cómo `q_\theta` se condiciona en la identidad química para generalizar.

## 6. Referencias para ampliar

- Frenkel & Smit, *Understanding Molecular Simulation* (2002). Cap. 7 (free energy), Cap. 5–6 (MC/MD).
- Tuckerman, *Statistical Mechanics: Theory and Molecular Simulation* (2010).
- Noé, Wu, Olsson, Köhler 2019, Science 365 — paper original, Sec. 1 motiva el problema en estos términos.
