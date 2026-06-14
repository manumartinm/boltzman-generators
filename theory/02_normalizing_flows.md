# 02 — Normalizing flows

## 1. Idea central

Un normalizing flow construye una distribución compleja `p_X(x)` aplicando una transformación invertible y diferenciable `f: Z → X` a un prior simple `p_Z(z)` (típicamente `N(0, I)`).

Si `x = f(z)` con `f` biyectiva y diferenciable:

$$
p_X(x) = p_Z(f^{-1}(x)) \, \left| \det J_{f^{-1}}(x) \right|
$$

equivalentemente, generando `z ~ p_Z` y haciendo `x = f(z)`:

$$
\log p_X(x) = \log p_Z(z) - \log \left| \det J_f(z) \right|
$$

Lo notable: **densidad exacta en `x` evaluable** si sabemos calcular el Jacobiano. Eso es lo que VAEs no tienen (solo cota inferior) y lo que GANs no tienen (densidad implícita).

## 2. Composición de transformaciones

Sea `f = f_K ∘ f_{K-1} ∘ ... ∘ f_1`. Por la regla de la cadena:

$$
\log \left| \det J_f(z) \right| = \sum_{k=1}^{K} \log \left| \det J_{f_k}(h_{k-1}) \right|
$$

con `h_0 = z`, `h_k = f_k(h_{k-1})`. Los log-det son **aditivos**. Esto permite apilar capas simples (cada una invertible y con jacobiano barato) para obtener un flow profundo.

## 3. El problema del Jacobiano

Para `x ∈ R^d`, `J_f` es `d × d`. Calcular `det` de una matriz arbitraria cuesta `O(d^3)`. Para `d = 300` (proteína pequeña), inaceptable en cada paso de entrenamiento.

Solución: diseñar `f` para que `J_f` sea **triangular** → `det` = producto de la diagonal → `O(d)`.

Estas son las dos familias clásicas:

- **Autoregresivos** (MAF, IAF): `f_i(z) = g(z_i; \text{params}(z_{<i}))`. Triangular por construcción.
- **Coupling layers** (NICE, RealNVP, Glow): split en dos mitades, una pasa intacta, la otra se transforma condicionada en la primera. Triangular en bloques.

## 4. RealNVP coupling layer (Dinh, Sohl-Dickstein, Bengio 2017)

Esta es la pieza que implementaremos en M3. Es el bloque básico del paper de Noé.

### 4.1 Definición

Dado `z ∈ R^d`, particionamos en dos mitades `z = (z_A, z_B)` con `|A| + |B| = d` (típicamente mitad/mitad). El **affine coupling**:

$$
\begin{aligned}
x_A &= z_A \\
x_B &= z_B \odot \exp(s_\theta(z_A)) + t_\theta(z_A)
\end{aligned}
$$

donde `s_\theta`, `t_\theta : R^{|A|} → R^{|B|}` son redes neuronales arbitrarias (MLPs), `\odot` es producto elemento a elemento, `\exp` es elementwise.

### 4.2 Inversa (cerrada, sin resolver sistemas)

$$
\begin{aligned}
z_A &= x_A \\
z_B &= (x_B - t_\theta(x_A)) \odot \exp(-s_\theta(x_A))
\end{aligned}
$$

**Las mismas redes `s` y `t`** se usan en forward e inverse — no necesitan ser invertibles. Solo el coupling lo es.

### 4.3 Jacobiano

$$
J_f = \begin{pmatrix} I_{|A|} & 0 \\ \frac{\partial x_B}{\partial z_A} & \mathrm{diag}(\exp(s(z_A))) \end{pmatrix}
$$

Triangular inferior por bloques con identidad arriba-izq. Por tanto:

$$
\log |\det J_f| = \sum_{i \in B} s_\theta(z_A)_i
$$

Coste `O(|B|)` por capa. Las redes `s, t` pueden ser tan grandes como queramos sin afectar al coste del log-det.

### 4.4 Por qué hace falta apilar varias

Una sola coupling deja `x_A = z_A` intacto → no es expresiva. Para alternar qué mitad se transforma se usan **máscaras** que rotan: capa 1 transforma mitad B condicionada en A; capa 2 transforma A condicionada en B; etc. Tras `K` capas (típicamente 4–10) todas las dimensiones se han mezclado.

Variantes prácticas: máscaras checkerboard (en imágenes), permutaciones aleatorias fijas, o permutaciones aprendidas via 1x1 conv invertibles (Glow).

## 5. Entrenamiento por MLE (forward KL)

Si tenemos un dataset `{x_i}` muestreado de un target desconocido `p^*(x)`, minimizamos:

$$
\mathcal{L}_{\text{MLE}} = -\mathbb{E}_{x \sim p^*}[\log p_X(x)] = -\mathbb{E}_{x \sim p^*}\!\left[ \log p_Z(f^{-1}(x)) - \log |\det J_f(f^{-1}(x))| \right] + \text{const}
$$

(el cambio de signo en el log-det viene de que estamos evaluando densidad en `x`, no en `z`).

Equivalente a minimizar `KL(p^* || p_X)` ("forward KL", mode-covering): penaliza fuertemente las regiones donde `p^* > 0` pero `p_X ≈ 0` → fuerza al modelo a **cubrir todos los modos**.

## 6. Conexión con BG

En el problema de muestreo de Boltzmann, **no tenemos** muestras del target `p^* = p_{\text{Boltz}}`. Tenemos algo más raro: la **función de energía** `u(x)`, evaluable puntualmente.

Esto cambia el juego: entrenamos minimizando la **reverse KL**:

$$
\mathcal{L}_{KL_z} = \mathrm{KL}(p_X || p_{\text{Boltz}}) = \mathbb{E}_{z \sim p_Z}\!\left[ u(f(z)) - \log |\det J_f(z)| \right] + \text{const}
$$

(detalle en cap. 03). Esta loss NO requiere samples del target — solo evaluar `u(f(z))`. Esa es la magia del Boltzmann Generator.

## 7. Otras familias de flows (referencia)

- **NICE** (Dinh 2014): coupling solo aditivo (`s = 0`). Volume-preserving (`det = 1`), menos expresivo.
- **Glow** (Kingma & Dhariwal 2018): RealNVP + 1×1 conv invertible + ActNorm. SOTA en imágenes (2018).
- **MAF / IAF** (Papamakarios 2017, Kingma 2016): autoregresivos. MAF rápido al evaluar densidad, lento al samplear; IAF al revés.
- **Neural Spline Flows** (Durkan 2019): coupling con splines monotónicos racionales en vez de affine. Más expresivos por capa, jacobiano sigue siendo triangular.
- **Continuous Normalizing Flows** (Chen 2018, Grathwohl 2018): `dx/dt = v_\theta(x, t)`, log-det via integración del divergente. Coste alto → motivó flow matching (cap. 04).

Para BG original (M3–M7) basta RealNVP. Para transferable BG (M8+) toca flow matching.

## 8. Referencias

- Dinh, Sohl-Dickstein, Bengio 2017, *"Density estimation using Real NVP"*, ICLR — paper canónico del coupling affine.
- Papamakarios, Nalisnick, Rezende, Mohamed, Lakshminarayanan 2021, *"Normalizing Flows for Probabilistic Modeling and Inference"*, JMLR — survey completo.
- Kobyzev, Prince, Brubaker 2020, IEEE TPAMI — survey alternativo, más enfocado a imagen.
