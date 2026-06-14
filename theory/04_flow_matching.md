# 04 — Flow matching

Refs:
- Chen et al. 2018, *"Neural ODEs"* — base de CNF.
- Grathwohl et al. 2018, *"FFJORD"* — CNF entrenable con MLE vía Hutchinson trace.
- Lipman, Chen, Ben-Hamu, Nickel, Le 2023, *"Flow Matching for Generative Modeling"*, ICLR.
- Tong et al. 2024, *"Conditional Flow Matching"*.

## 1. Continuous normalizing flows (CNF)

En lugar de una composición discreta `f_K ∘ ... ∘ f_1`, definimos un campo vectorial `v_\theta(x, t)` y la transformación como flujo de la ODE:

$$
\frac{dx(t)}{dt} = v_\theta(x(t), t), \qquad x(0) = z \sim p_Z, \quad x(1) = x \sim p_X
$$

El cambio instantáneo de log-densidad sigue:

$$
\frac{d \log p(x(t), t)}{dt} = -\nabla \cdot v_\theta(x(t), t)
$$

Entonces:

$$
\log p_X(x(1)) = \log p_Z(z) - \int_0^1 \nabla \cdot v_\theta(x(t), t) \, dt
$$

**Ventaja vs flows discretos:** no se necesita arquitectura especial (cualquier MLP/CNN para `v`); la invertibilidad la garantiza la ODE.

**Desventaja:** computar `\nabla \cdot v` exacto es `O(d^2)` (Jacobiano completo). FFJORD usa la estimación de Hutchinson:

$$
\nabla \cdot v \approx \epsilon^T \frac{\partial v}{\partial x} \epsilon, \qquad \epsilon \sim N(0, I)
$$

con `O(d)` (un producto Jacobiano-vector vía autodiff). Aun así, entrenamiento por MLE requiere integrar la ODE en cada batch — lento y delicado con adaptive ODE solvers.

## 2. Flow matching: regresión directa del campo

Idea central de Lipman 2023: en vez de entrenar `v` minimizando MLE (caro), **regresar `v` directamente sobre un campo objetivo conocido**.

### 2.1 Path probabilístico

Definimos una familia de distribuciones `p_t(x)` con `p_0 = p_Z` (prior) y `p_1 = p_X` (target). Para cada `t`, hay un campo `u_t(x)` que **genera** `p_t` (es decir, integrar `dx/dt = u_t(x)` desde `z ~ p_0` produce `x_t ~ p_t`).

El **flow matching loss** (no práctico aún):

$$
\mathcal{L}_{FM} = \mathbb{E}_{t \sim U[0,1], x \sim p_t} \left\| v_\theta(x, t) - u_t(x) \right\|^2
$$

Problema: no conocemos `u_t(x)` (es la cantidad que queremos aprender).

### 2.2 Conditional flow matching (CFM)

Truco clave: en vez de un `p_t` marginal, condicionamos en un punto del target `x_1`. Para cada `x_1`, construimos un path **condicional** `p_t(x | x_1)` con campo `u_t(x | x_1)` que sí conocemos. Lipman demuestra:

$$
\nabla_\theta \mathcal{L}_{FM} = \nabla_\theta \mathcal{L}_{CFM}
$$

donde:

$$
\mathcal{L}_{CFM} = \mathbb{E}_{t \sim U[0,1], x_1 \sim p_X, x \sim p_t(\cdot | x_1)} \left\| v_\theta(x, t) - u_t(x | x_1) \right\|^2
$$

Ambas losses tienen los mismos gradientes — entrenar con la condicional es equivalente y mucho más simple.

### 2.3 Optimal transport path (lineal)

La elección más usada (Lipman Sec. 4.1): path lineal entre un `z_0 ~ p_Z` y `x_1 ~ p_X`:

$$
x_t = (1 - t) \cdot z_0 + t \cdot x_1, \qquad u_t(x_t | x_1, z_0) = x_1 - z_0
$$

El campo condicional es **constante en `t`** — la velocidad necesaria para ir de `z_0` a `x_1` en tiempo 1. Loss:

$$
\mathcal{L}_{CFM-OT} = \mathbb{E}_{t \sim U[0,1], z_0 \sim p_Z, x_1 \sim p_X} \left\| v_\theta((1-t) z_0 + t x_1, t) - (x_1 - z_0) \right\|^2
$$

**Esto es un problema de regresión vanilla.** Una pasada forward, MSE loss, sin integrar ODEs, sin Hutchinson. Sólo en inferencia (sampling) integramos `dx/dt = v_\theta(x, t)`.

## 3. Por qué reemplazó MLE en flows continuos

| Aspecto | CNF + MLE | Flow matching |
|---|---|---|
| Loss | ODE inversa + Hutchinson trace | MSE plano |
| Coste / step | alto (varios pasos ODE) | bajo (1 forward) |
| Inferencia | ODE solve | ODE solve (igual) |
| Estabilidad | depende del solver | sólida |
| Densidad exacta `log p` | sí (caro) | sí, vía ODE en inferencia |

Para *training* moderno: FM gana en velocidad y estabilidad. Para *evaluar `log p`* exacta sigue requiriendo integrar.

## 4. Conexión con difusión

Si `p_t` es la difusión Gaussiana del target hacia ruido, el campo `u_t` equivale al **score** `\nabla \log p_t`. FM con ese path = **score matching** estándar. FM es más general: cualquier path interpolante, no solo difusión.

Esto unifica la literatura: stochastic interpolants, rectified flows, score matching, FM, CFM — todas son casos particulares de regresión de campo vectorial sobre una familia de paths.

## 5. Implementación mínima en PyTorch (M9)

```python
def cfm_step(model, x1, t=None):
    n = x1.shape[0]
    if t is None:
        t = torch.rand(n, device=x1.device)
    z0 = torch.randn_like(x1)
    x_t = (1 - t)[:, None] * z0 + t[:, None] * x1
    u_target = x1 - z0
    v_pred = model(x_t, t)  # MLP taking (x, t) -> velocity
    return ((v_pred - u_target) ** 2).sum(dim=-1).mean()
```

Sampling = integrar `dx/dt = v(x, t)` desde `z ~ N(0,I)`, `t: 0 → 1`. Solver: Euler fijo (rápido, suficiente para validar) o `torchdiffeq.odeint` (alta precisión).

## 6. Aplicación a Boltzmann Generators

FM por sí mismo entrena con `KL_x` (necesita samples). Para BG con FM:

1. **Pretrain FM** con samples MD/MCMC (`KL_x` estándar via FM).
2. **Fine-tune con KL_z** evaluando `log q_\theta(x)` vía ODE (lento pero factible) o vía proxies (energía-based losses sin densidad explícita).

Klein & Noé 2024 usa FM en cartesianas para BG transferable (cap. 05).

## 7. Detalles para M9

- Modelo: MLP `(x ∈ R^d, t ∈ [0,1]) → R^d`. Concatenar `t` como input, o usar Fourier features `[sin(2π k t), cos(2π k t)]_k`.
- Optimizer: Adam(3e-4), batch 256–512.
- Solver inferencia: Euler 50–200 pasos, o RK4 50 pasos.
- Sanity check: ajustar 2D gaussian mixture y reproducir resultado del notebook 02 RealNVP.
