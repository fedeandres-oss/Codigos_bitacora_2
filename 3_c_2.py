import numpy as np
import matplotlib.pyplot as plt

# ---------------- Funciones auxiliares (sin scipy) ----------------
def square(x):
    """Onda cuadrada +1/-1 (equivalente a signal.square, duty 50%)."""
    return np.where(np.sin(x) >= 0, 1.0, -1.0)


def filtrar(coef, x):
    """FIR causal: y[n] = sum_k coef[k] * x[n-k]  (equivale a lfilter)."""
    return np.convolve(x, coef)[:len(x)]


def fft_mag(x, fs):
    N = len(x)
    f = np.fft.fftfreq(N, d=1 / fs)[:N // 2]
    X = (2 / N) * np.abs(np.fft.fft(x)[:N // 2])
    return f, X


# ---------------- 1. Parámetros ----------------
fs = 10**5                       # Hz (según enunciado)
dt = 1 / fs
t = np.arange(0, 0.2, dt)        # 200 ms

fc = 10**3                       # portadora
fm = 60                          # moduladora

# ---------------- 2. Señales ----------------
x1 = (2 + square(2 * np.pi * fm * t)) * np.sin(2 * np.pi * fc * t)   # A (AM)
x2 = square(2 * np.pi * fm * t)                                       # B

# ---------------- 3. Filtro de promedio móvil ----------------
# fc(-3 dB) ≈ 0.443 * fs / M   ->   M = 0.443 * fs / fc
cutoffs = [40, 70, 120]
ma_filters = {}
for fcut in cutoffs:
    M = int(round(0.443 * fs / fcut))
    ma_filters[fcut] = np.ones(M) / M
    print(f"fc = {fcut:3d} Hz -> M = {M} muestras")

# FFT de las originales (se calculan una sola vez)
f_x1, X1 = fft_mag(x1, fs)
f_x2, X2 = fft_mag(x2, fs)

# ---------------- 4. Una figura (pestaña) por frecuencia de corte ----------------
for fcut in cutoffs:
    coef = ma_filters[fcut]
    y1 = filtrar(coef, x1)
    y2 = filtrar(coef, x2)
    f_y1, Y1 = fft_mag(y1, fs)
    f_y2, Y2 = fft_mag(y2, fs)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.canvas.manager.set_window_title(f"Promedio móvil fc = {fcut} Hz")
    fig.suptitle(f"Filtro de promedio móvil fc = {fcut} Hz (M = {len(coef)})")

    # (0,0) A en el tiempo
    axs[0, 0].plot(t * 1000, x1, color="gray", label="A original", alpha=0.7)
    axs[0, 0].plot(t * 1000, y1, color="b", label="A filtrada", linewidth=1.5)
    axs[0, 0].set_title("Señal A (x1) - tiempo")
    axs[0, 0].set_xlabel("t (ms)")
    axs[0, 0].set_ylabel("Amplitud")
    axs[0, 0].set_xlim(0, 50)
    axs[0, 0].set_ylim(-3.2, 3.2)
    axs[0, 0].grid(True)
    axs[0, 0].legend(loc="upper right")

    # (0,1) A FFT
    axs[0, 1].plot(f_x1, X1, color="gray", label="A original")
    axs[0, 1].plot(f_y1, Y1, color="b", label="A filtrada")
    axs[0, 1].axvline(fcut, color="r", ls="--", lw=0.8, label=f"fc = {fcut} Hz")
    axs[0, 1].set_title("Señal A (x1) - FFT")
    axs[0, 1].set_xlabel("Frecuencia (Hz)")
    axs[0, 1].set_ylabel("Magnitud")
    axs[0, 1].set_xlim(0, 1500)
    axs[0, 1].set_ylim(0, 2.2)
    axs[0, 1].grid(True)
    axs[0, 1].legend(loc="upper right")

    # (1,0) B en el tiempo
    axs[1, 0].plot(t * 1000, x2, color="gray", label="B original", alpha=0.7)
    axs[1, 0].plot(t * 1000, y2, color="g", label="B filtrada", linewidth=1.5)
    axs[1, 0].set_title("Señal B (x2) - tiempo")
    axs[1, 0].set_xlabel("t (ms)")
    axs[1, 0].set_ylabel("Amplitud")
    axs[1, 0].set_xlim(0, 50)
    axs[1, 0].set_ylim(-1.5, 1.5)
    axs[1, 0].grid(True)
    axs[1, 0].legend(loc="upper right")

    # (1,1) B FFT
    axs[1, 1].plot(f_x2, X2, color="gray", label="B original")
    axs[1, 1].plot(f_y2, Y2, color="g", label="B filtrada")
    axs[1, 1].axvline(fcut, color="r", ls="--", lw=0.8, label=f"fc = {fcut} Hz")
    axs[1, 1].set_title("Señal B (x2) - FFT")
    axs[1, 1].set_xlabel("Frecuencia (Hz)")
    axs[1, 1].set_ylabel("Magnitud")
    axs[1, 1].set_xlim(0, 400)
    axs[1, 1].set_ylim(0, 1.5)
    axs[1, 1].grid(True)
    axs[1, 1].legend(loc="upper right")

    fig.tight_layout()

# ---------------- 5. Ventana extra: los filtros en sí ----------------
nfft = 2**18
colores = {40: "tab:blue", 70: "tab:orange", 120: "tab:green"}

fig, axs = plt.subplots(2, 2, figsize=(14, 8))
fig.canvas.manager.set_window_title("Filtros de promedio móvil")
fig.suptitle("Filtros de promedio móvil: 40, 70 y 120 Hz")

for fcut in cutoffs:
    h = ma_filters[fcut]
    M = len(h)
    H = np.fft.rfft(h, nfft)
    f_h = np.fft.rfftfreq(nfft, d=1 / fs)
    mag = np.abs(H)
    mag_db = 20 * np.log10(mag + 1e-12)
    fase = np.unwrap(np.angle(H))
    c = colores[fcut]
    lbl = f"fc={fcut} Hz (M={M})"

    # frecuencia real donde la magnitud cae a -3 dB
    idx = np.argmax(mag <= 1 / np.sqrt(2))
    print(f"M = {M:4d} -> -3 dB real en {f_h[idx]:.1f} Hz (objetivo {fcut} Hz)")

    # (0,0) respuesta al impulso
    axs[0, 0].plot(np.arange(M) / fs * 1000, h, color=c, label=lbl)
    # (0,1) magnitud lineal
    axs[0, 1].plot(f_h, mag, color=c, label=lbl)
    # (1,0) magnitud dB
    axs[1, 0].plot(f_h, mag_db, color=c, label=lbl)
    # (1,1) fase
    axs[1, 1].plot(f_h, fase, color=c, label=lbl)

axs[0, 0].set_title("Respuesta al impulso h[n] = 1/M")
axs[0, 0].set_xlabel("t (ms)")
axs[0, 0].set_ylabel("Amplitud")

axs[0, 1].set_title("Magnitud |H(f)| (lineal)")
axs[0, 1].set_xlabel("Frecuencia (Hz)")
axs[0, 1].set_xlim(0, 400)
axs[0, 1].axhline(1 / np.sqrt(2), color="r", ls="--", lw=0.8)

axs[1, 0].set_title("Magnitud |H(f)| (dB)")
axs[1, 0].set_xlabel("Frecuencia (Hz)")
axs[1, 0].set_ylabel("dB")
axs[1, 0].set_xlim(0, 400)
axs[1, 0].set_ylim(-40, 3)
axs[1, 0].axhline(-3, color="r", ls="--", lw=0.8, label="-3 dB")

axs[1, 1].set_title("Fase de H(f)")
axs[1, 1].set_xlabel("Frecuencia (Hz)")
axs[1, 1].set_ylabel("rad")
axs[1, 1].set_xlim(0, 400)

for ax in axs.ravel():
    ax.grid(True)
    ax.legend(loc="upper right")

fig.tight_layout()

plt.show()   # abre las 4 ventanas