import numpy as np
import matplotlib.pyplot as plt
from math import erf
# ============================================================
# TP3 - 
# Ecuación del calor con tres esquemas:
#   - Explícito
#   - Implícito
#   - Crank–Nicolson


#PUNTO 1
# Parámetros del modelo, los que nos dieron en el tp
K     = 100.0      # strike
r     = 0.05       # tasa libre de riesgo
sigma = 0.20       # volatilidad
T     = 1.0        # madurez


# Payoff en S

def payoff_call(S, K=K):
    return np.maximum(S - K, 0.0)

def payoff_put(S, K=K):
    return np.maximum(K - S, 0.0)


def condicion_inicial_y(x, tipo_opcion="call"):
    S = np.exp(x)          
    descuento = np.exp(-r*T)
    if tipo_opcion == "call":
        return descuento * payoff_call(S)
    elif tipo_opcion == "put":
        return descuento * payoff_put(S)
    else:
        raise ValueError("tipo_opcion debe ser 'call' o 'put'")


def construir_malla(S_min=1e-3, S_max=4*K, M=200, N=400):
    x_min = np.log(S_min)
    x_max = np.log(S_max)

    tau_max = 0.5 * sigma**2 * T

    dx = (x_max - x_min) / M
    dtau = tau_max / N

    x   = np.linspace(x_min,  x_max,  M+1)
    tau = np.linspace(0.0, tau_max, N+1)

    return x, tau, dx, dtau


# EXPLÍCITO (FTCS)
#
# y_j^{n+1} = y_j^n + s (y_{j+1}^n - 2 y_j^n + y_{j-1}^n)
# con   s = Δtau / (Δx)^2
# Condiciones de borde: y = 0 en x_min y x_max

# Esto va en el latex
def calor_explicito(tipo_opcion="call", S_min=1e-3, S_max=4*K, M=200, N=400):
    x, tau, dx, dtau = construir_malla(S_min, S_max, M, N)
    s = dtau / dx**2

    # matriz de solución:
    y = np.zeros((N+1, M+1))

    # condición inicial
    y[0, :] = condicion_inicial_y(x, tipo_opcion)

    # condiciones de borde 
    y[:, 0]   = 0.0
    y[:, -1]  = 0.0

    # iteración en el tiempo
    for n in range(N):
        for j in range(1, M):
            y[n+1, j] = y[n, j] + s * (y[n, j+1] - 2.0*y[n, j] + y[n, j-1])

    return x, tau, y, s


# 2) Esquema IMPLÍCITO (BTCS)
#
# y_j^{n+1} - s (y_{j+1}^{n+1} - 2 y_j^{n+1} + y_{j-1}^{n+1}) = y_j^n
#
# Para j = 1,...,M-1 se obtiene un sistema lineal tridiagonal:
#
#   -s y_{j-1}^{n+1} + (1+2s) y_j^{n+1} - s y_{j+1}^{n+1} = y_j^n
# latex
def calor_implicito(tipo_opcion="call", S_min=1e-3, S_max=4*K, M=200, N=400):
    x, tau, dx, dtau = construir_malla(S_min, S_max, M, N)
    s = dtau / dx**2

    y = np.zeros((N+1, M+1))
    y[0, :] = condicion_inicial_y(x, tipo_opcion)

    y[:, 0]  = 0.0
    y[:, -1] = 0.0

    # matriz tridiagonal A 1
    diag_principal = (1.0 + 2.0*s) * np.ones(M-1)
    diag_secund    = -s * np.ones(M-2)
    A = np.diag(diag_principal) + np.diag(diag_secund, 1) + np.diag(diag_secund, -1)

    for n in range(N):
        # lado derecho con valores en el tiempo n
        b = y[n, 1:M].copy()

        # aportes de las condiciones de borde 
        b[0]  += s * y[n+1, 0]     # x_min
        b[-1] += s * y[n+1, -1]    # x_max

        # resolución del sistema lineal
        y[n+1, 1:M] = np.linalg.solve(A, b)

    return x, tau, y, s


# 3) Esquema CRANK–NICOLSON
#
# y_j^{n+1} - (s/2)(y_{j+1}^{n+1} - 2 y_j^{n+1} + y_{j-1}^{n+1})
#   = y_j^n + (s/2)(y_{j+1}^n - 2 y_j^n + y_{j-1}^n)
#
# Se obtiene un sistema A y^{n+1} = B y^n (tridiagonal en A y B)
# Esto sirve para el latex
def calor_crank_nicolson(tipo_opcion="call", S_min=1e-3, S_max=4*K, M=200, N=400):
    x, tau, dx, dtau = construir_malla(S_min, S_max, M, N)
    s = dtau / dx**2

    y = np.zeros((N+1, M+1))
    y[0, :] = condicion_inicial_y(x, tipo_opcion)

    y[:, 0]  = 0.0
    y[:, -1] = 0.0

    # A lado implícito
    diagA_principal = (1.0 + s) * np.ones(M-1)
    diagA_secund    = -0.5 * s * np.ones(M-2)
    A = np.diag(diagA_principal) + np.diag(diagA_secund, 1) + np.diag(diagA_secund, -1)

    # B lado explícito
    diagB_principal = (1.0 - s) * np.ones(M-1)
    diagB_secund    = 0.5 * s * np.ones(M-2)
    B = np.diag(diagB_principal) + np.diag(diagB_secund, 1) + np.diag(diagB_secund, -1)

    for n in range(N):
        yn = y[n, :]

        # parte explícita B * y^n (solo nodos interiores)
        rhs = B @ yn[1:M]

        # aportes de las condiciones de borde (aquí 0)
        rhs[0]  += 0.5 * s * (yn[0]  + y[n+1, 0])
        rhs[-1] += 0.5 * s * (yn[-1] + y[n+1, -1])

        # resolvemos A * y^{n+1}_interior = rhs
        y[n+1, 1:M] = np.linalg.solve(A, rhs)

    return x, tau, y, s


# PUNTO 2: 
# Recuperar V(S, t) a partir de y(x, τ)
# Relaciones:
#   τ = σ²/2 (T - t)        →  t = T - 2 τ / σ²
#   x = ln S + (r - σ²/2)(T - t)
#     → ln S = x - (r - σ²/2)(T - t)
#     → S    = exp( x - (r - σ²/2)(T - t) )
#   V(S,t) = e^{r t} y(x, τ)
# Esto sirve a en el latex, me lo hizo el chat

def retransformar_a_V(x, tau, y):
    # Número de nodos en tiempo (N+1) y espacio (M+1)
    N_plus_1, M_plus_1 = y.shape

    # 1) t a partir de τ:  t = T - 2 τ / σ²
    t = T - (2.0 / sigma**2) * tau      # tamaño N+1

    # 2) Construimos una grilla (t,x) 2D a partir de los 1D
    #    t_grid[n, j] = t_n
    #    x_grid[n, j] = x_j
    t_grid, x_grid = np.meshgrid(t, x, indexing="ij")

    # 3) S a partir de x y t:
    #    ln S = x - (r - σ²/2)(T - t)
    #    S    = exp( x - (r - σ²/2)(T - t) )
    factor = (r - 0.5 * sigma**2) * (T - t_grid)
    S_grid = np.exp(x_grid - factor)

    # 4) V a partir de y:
    #    V(S,t) = e^{r t} y(x, τ)
    V_grid = np.exp(r * t_grid) * y

    return S_grid, t_grid, V_grid

#punto 1
x, tau, y_call, s_call = calor_crank_nicolson(tipo_opcion="call")

#punto 2
S_call, t_call, V_call = retransformar_a_V(x, tau, y_call)





# acumulada normal estándar y Black–Scholes

def N_cdf_scalar(z):
    """
    acumulada de la normal estándar para un escalar z.
    Usa la función error 'erf' de la librería math.

    N(z) = 0.5 * (1 + erf(z / sqrt(2)))
    """
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def N_cdf(z):
    """
    Versión vectorizada de la acumulada.
    Funciona si z es escalar o array de NumPy.

    La idea es aplicar N_cdf_scalar elemento a elemento.
    """
    z = np.array(z, dtype=float)          # nos aseguramos de tener un array de floats
    return np.vectorize(N_cdf_scalar)(z)  # aplica N_cdf_scalar a cada entrada de z


def black_scholes_call(S, K=K, r=r, sigma=sigma, T=T, t=0.0):
 
    S = np.array(S, dtype=float)

    tau = T - t

    if tau <= 0.0:
        return payoff_call(S, K)

    sqrt_tau = np.sqrt(tau)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    return S * N_cdf(d1) - K * np.exp(-r * tau) * N_cdf(d2)


def black_scholes_put(S, K=K, r=r, sigma=sigma, T=T, t=0.0):

    S = np.array(S, dtype=float)

    tau = T - t

    if tau <= 0.0:
        return payoff_put(S, K)

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    return K * np.exp(-r * tau) * N_cdf(-d2) - S * N_cdf(-d1)


def graficar_curvas_V(S_grid, t_grid, V_grid, n_curvas=6, titulo="Call europea"):
    """
    Grafica varias curvas V(S, t_j) para distintos tiempos t_j.

    - Cada fila de S_grid y V_grid corresponde a un tiempo t fijo.
    - Elegimos 'n_curvas' tiempos, equiespaciados entre t_min y t_max.
    - Para cada uno, graficamos V(S, t_j) en función de S.
    """
    # Extraemos la "línea" de tiempos: todos los t de la primera columna son iguales
    t_line = t_grid[:, 0]   # shape (N_t,)

    # Cantidad de tiempos disponibles
    N_t = t_line.shape[0]

    # Elegimos índices equiespaciados para los tiempos que vamos a graficar
    indices = np.linspace(0, N_t - 1, n_curvas, dtype=int)

    plt.figure()

    for idx in indices:
        # Para el tiempo t_line[idx], tomamos la fila correspondiente
        S_line = S_grid[idx, :]   # precios S en ese tiempo
        V_line = V_grid[idx, :]   # valores V(S, t_j) en ese tiempo

        # Es posible que S_line no esté ordenado, así que lo ordenamos junto con V_line
        orden = np.argsort(S_line)
        S_ord = S_line[orden]
        V_ord = V_line[orden]

        t_val = t_line[idx]  # valor numérico del tiempo
        plt.plot(S_ord, V_ord, label=f"t = {t_val:.2f}")

    plt.xlabel("S (precio del subyacente)")
    plt.ylabel("V(S, t)")
    plt.title(f"Curvas V(S, t_j) - {titulo}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def graficar_error_relativo_t0(S_grid, t_grid, V_grid, tipo_opcion="call"):
    """
    Calcula y grafica el error relativo entre el precio numérico V(S, t)
    y el precio teórico de Black–Scholes, en el tiempo t = 0.

    1) Busca en la grilla el tiempo más cercano a t = 0.
    2) Para esa fila, toma S y V_num.
    3) Calcula V_BS(S, t=0) con la fórmula cerrada correspondiente.
    4) Define error relativo = |V_num - V_BS| / max(|V_BS|, eps).
    5) Grafica el error relativo como función de S.
    """
    # Línea de tiempos (todos los t de la primera columna)
    t_line = t_grid[:, 0]

    # Índice del tiempo más cercano a t = 0
    idx_t0 = np.argmin(np.abs(t_line - 0.0))

    # Curvas de S y V_num en t ~ 0
    S_t0 = S_grid[idx_t0, :]
    V_num_t0 = V_grid[idx_t0, :]

    # Precio teórico de Black–Scholes según el tipo de opción
    if tipo_opcion == "call":
        V_bs_t0 = black_scholes_call(S_t0, t=0.0)
        titulo = "Error relativo vs Black–Scholes en t = 0 (call)"
    elif tipo_opcion == "put":
        V_bs_t0 = black_scholes_put(S_t0, t=0.0)
        titulo = "Error relativo vs Black–Scholes en t = 0 (put)"
    else:
        raise ValueError("tipo_opcion debe ser 'call' o 'put'.")

    # Para evitar divisiones por cero cuando V_BS es muy pequeño, usamos un eps
    eps = 1e-10
    denom = np.maximum(np.abs(V_bs_t0), eps)

    # Error relativo
    err_rel = np.abs(V_num_t0 - V_bs_t0) / denom

    # Ordenamos por S para que el gráfico sea prolijo
    orden = np.argsort(S_t0)
    S_ord = S_t0[orden]
    err_ord = err_rel[orden]

    # Graficamos
    plt.figure()
    plt.plot(S_ord, err_ord)
    plt.xlabel("S (precio del subyacente)")
    plt.ylabel("Error relativo")
    plt.title(titulo)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


x_call, tau_call, y_call, s_call = calor_crank_nicolson(tipo_opcion="call")

x_put, tau_put, y_put, s_put = calor_crank_nicolson(tipo_opcion="put")

S_call, t_call, V_call = retransformar_a_V(x_call, tau_call, y_call)

S_put, t_put, V_put = retransformar_a_V(x_put, tau_put, y_put)

graficar_curvas_V(S_grid=S_call, t_grid=t_call, V_grid=V_call, n_curvas=6, titulo="Call (Crank–Nicolson)")

graficar_curvas_V(S_grid=S_put, t_grid=t_put, V_grid=V_put, n_curvas=6, titulo="Put (Crank–Nicolson)")

graficar_error_relativo_t0(S_call, t_call, V_call, tipo_opcion="call")

graficar_error_relativo_t0(S_put, t_put, V_put, tipo_opcion="put")



def error_ATM(M, N):
    """
    Calcula el error absoluto en el precio de la opción en S ≈ K, t = 0
    usando Crank–Nicolson con una malla (M, N).

    1) Resuelve la ecuación del calor con Crank–Nicolson en una malla de
       M nodos espaciales y N pasos de tiempo.
    2) Retransforma la solución a V(S,t).
    3) Toma el tiempo más cercano a t = 0.
    4) En ese tiempo, busca el nodo de S más cercano a K (ATM).
    5) Compara el precio numérico con Black–Scholes y devuelve el error
       absoluto en ese punto.
    """
    # 1) Resolvemos la ecuación del calor (y(x, tau)) con Crank–Nicolson
    x, tau, y_num, s = calor_crank_nicolson(tipo_opcion="call", M=M, N=N)

    # 2) Retransformamos a V(S, t)
    S_grid, t_grid, V_grid = retransformar_a_V(x, tau, y_num)

    # 3) Encontramos el índice del tiempo más cercano a t = 0
    t_line = t_grid[:, 0]                    # todos los tiempos (una dimensión)
    idx_t0 = np.argmin(np.abs(t_line - 0.0)) # índice de t ≈ 0

    # Curvas de S y V en ese tiempo
    S_t0 = S_grid[idx_t0, :]
    V_t0 = V_grid[idx_t0, :]

    # 4) Buscamos el nodo espacial con S más cercano a K (ATM)
    idx_atm = np.argmin(np.abs(S_t0 - K))
    S_atm = S_t0[idx_atm]
    V_num_atm = V_t0[idx_atm]

    # 5) Precio teórico de Black–Scholes en ese mismo punto
    V_bs_atm = black_scholes_call(S_atm, t=0.0)

    # Error absoluto en el punto ATM
    error_abs = np.abs(V_num_atm - V_bs_atm)
    return error_abs, S_atm


def estudio_convergencia():
    """
    - Elegimos varias mallas (M, N) cada vez más finas.
    - Para cada una calculamos el error absoluto en S ≈ K, t = 0.
    - Mostramos cómo decrece el error al refinar la grilla.

    """

    # Distintas mallas: al aumentar M y N, refinamos la discretización
    M_list = [50, 100, 200, 400]
    N_list = [100, 200, 400, 800]  # podés ajustar estos valores si querés

    errores = []

    for M, N in zip(M_list, N_list):
        # Para ver también los tamaños de paso dx y dtau de cada malla
        x, tau, dx, dtau = construir_malla(M=M, N=N)

        # Error ATM para esta malla
        err_atm, S_atm = error_ATM(M, N)
        errores.append(err_atm)

        # Imprimimos información útil para el informe
        print(f"M = {M:4d}, N = {N:4d}, "f"dx = {dx:.5f}, dtau = {dtau:.5f}, "f"S_ATM ≈ {S_atm:.4f}, error = {err_atm:.6e}")

# Llamamos al estudio de convergencia para que se ejecute al correr el script
estudio_convergencia()
