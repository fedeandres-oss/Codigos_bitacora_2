import numpy as np
import matplotlib.pyplot as plt
import librosa
import time
from scipy.signal import butter, lfilter

# ==========================================
# 1. PARÁMETROS Y CARGA DEL AUDIO (Fs = 1 kHz)
# ==========================================
fs = 1000.0             # Frecuencia de muestreo (1 kHz)
archivo_audio = 'gato.mp3'

# Rango de tiempo para visualizar
t_inicio = 1.0
t_fin = 1.5

# Cargar audio ajustado a 1 kHz
x_completo, _ = librosa.load(archivo_audio, sr=int(fs))
x_completo = x_completo / np.max(np.abs(x_completo)) # Normalización

N = len(x_completo)
t = np.arange(N) / fs

# Modulación AM a 300 Hz
f_portadora = 300.0   
x_modulada = x_completo * np.cos(2 * np.pi * f_portadora * t)

# ==========================================
# 2. COEFICIENTES (Fs = 1 kHz, Ganancia A = 2)
# ==========================================

# --- A) CÁLCULO MANUAL (2º Orden, A = 2) ---
# fc = 40 Hz
b1_man = np.array([0.0267180, 0.0534360, 0.0267180])
a1_man = np.array([1.0, -1.6474630, 0.7008990])

# fc = 100 Hz
b2_man = np.array([0.1349106, 0.2698212, 0.1349106])
a2_man = np.array([1.0, -1.1429806, 0.4128016])

# --- B) MATLAB (4º Orden, Ganancia A = 2) ---
b1_mat = 2 * np.array([1.832160233696088e-04, 7.328640934784353e-04, 1.099296140217653e-03, 7.328640934784353e-04, 1.832160233696088e-04])
a1_mat = np.array([1.000000000000000e+00, -3.344067837711872e+00, 4.238863950884062e+00, -2.409342856586316e+00, 5.174781997880397e-01])
b2_mat = 2 * np.array([4.824343357716226e-03, 1.929737343086491e-02, 2.894606014629736e-02, 1.929737343086491e-02, 4.824343357716226e-03])
a2_mat = np.array([1.000000000000000e+00, -2.369513007182036e+00, 2.313988414415877e+00, -1.054665405878565e+00, 1.873794923681843e-01])

# --- C) PYTHON / SCIPY (4º Orden, Ganancia A = 2) ---
b1_py, a1_py = butter(4, 40 / (fs / 2), btype='low')
b2_py, a2_py = butter(4, 100 / (fs / 2), btype='low')
b1_py, b2_py = 2 * b1_py, 2 * b2_py

# ==========================================
# 3. ALGORITMO DE FILTRADO MANUAL (CICLO FOR)
# ==========================================
def aplicar_filtro_ciclo_for(x_in, b_coeff, a_coeff):
    N_samples = len(x_in)
    y_out = np.zeros(N_samples, dtype=np.float64)
    for n in range(N_samples):
        for i in range(len(b_coeff)):
            if n - i >= 0:
                y_out[n] += b_coeff[i] * x_in[n - i]
        for j in range(1, len(a_coeff)):
            if n - j >= 0:
                y_out[n] -= a_coeff[j] * y_out[n - j]
    return y_out

# Filtrado por ciclos for
t0 = time.time()
y40_man = aplicar_filtro_ciclo_for(x_modulada, b1_man, a1_man)
y40_mat = aplicar_filtro_ciclo_for(x_modulada, b1_mat, a1_mat)
y40_py  = aplicar_filtro_ciclo_for(x_modulada, b1_py, a1_py)
t_for = time.time() - t0

y100_man = aplicar_filtro_ciclo_for(x_modulada, b2_man, a2_man)
y100_mat = aplicar_filtro_ciclo_for(x_modulada, b2_mat, a2_mat)
y100_py  = aplicar_filtro_ciclo_for(x_modulada, b2_py, a2_py)

# ==========================================
# 4. FILTRADO CON FUNCIÓN RESERVADA (lfilter)
# ==========================================
t0 = time.time()
y40_lfilter = lfilter(b1_py, a1_py, x_modulada)
t_native = time.time() - t0

diferencia = y40_py - y40_lfilter
mse = np.mean(diferencia**2)

print("=" * 60)
print("       COMPARACIÓN: CICLO FOR MANUAL vs lfilter")
print("=" * 60)
print(f"Error Cuadrático Medio (MSE): {mse:.4e}")
print(f"Tiempo Ciclo 'for': {t_for:.5f} s")
print(f"Tiempo 'lfilter':   {t_native:.5f} s")
print(f"Aceleración con 'lfilter': {t_for/t_native:.1f}x más rápido")
print("=" * 60)

# ==========================================
# 5. FIGURA 1: COMPARACIÓN DE FUENTES DE COEFICIENTES
# ==========================================
fig1, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axs[0].plot(t, x_modulada, color='purple', label='Señal Modulada AM (fc = 300 Hz)')
axs[0].set_title('Entrada Modulada (Fs = 1 kHz)')
axs[0].set_ylabel('Amplitud')
axs[0].grid(True)
axs[0].legend(loc='upper right')

axs[1].plot(t, y40_man, color='red', label='Cálculo manual (2º Orden, A=2)', linewidth=1.8)
axs[1].plot(t, y40_mat, color='black', linestyle='--', label='MATLAB (4º Orden, A=2)', linewidth=1.5)
axs[1].plot(t, y40_py, color='cyan', linestyle=':', label='SciPy (4º Orden, A=2)', linewidth=1.5)
axs[1].set_title('Respuesta del Filtro fc = 40 Hz (Ciclo for)')
axs[1].set_ylabel('Amplitud')
axs[1].grid(True)
axs[1].legend(loc='upper right')

axs[2].plot(t, y100_man, color='blue', label='Cálculo manual (2º Orden, A=2)', linewidth=1.8)
axs[2].plot(t, y100_mat, color='black', linestyle='--', label='MATLAB (4º Orden, A=2)', linewidth=1.5)
axs[2].plot(t, y100_py, color='orange', linestyle=':', label='SciPy (4º Orden, A=2)', linewidth=1.5)
axs[2].set_xlabel('Tiempo (s)')
axs[2].set_ylabel('Amplitud')
axs[2].set_title('Respuesta del Filtro fc = 100 Hz (Ciclo for)')
axs[2].grid(True)
axs[2].legend(loc='upper right')

plt.xlim([t_inicio, t_fin])
plt.tight_layout()

# ==========================================
# 6. FIGURA 2: COMPARACIÓN DE MÉTODOS DE PYTHON (CICLO FOR VS LFILTER)
# ==========================================
fig2, axs_comp = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Subplot 1: Superposición de ambas señales
axs_comp[0].plot(t, y40_py, color='cyan', label='Implementación Ciclo For Manual', linewidth=2.0)
axs_comp[0].plot(t, y40_lfilter, color='black', linestyle='--', label='Función Reservada (scipy.signal.lfilter)', linewidth=1.2)
axs_comp[0].set_ylabel('Amplitud')
axs_comp[0].set_title('Comparación de Métodos de Filtrado en Python (Filtro 40 Hz)')
axs_comp[0].grid(True)
axs_comp[0].legend(loc='upper right')

# Subplot 2: Diferencia punto a punto (Error)
axs_comp[1].plot(t, diferencia, color='magenta', label='Diferencia (Error de Precisión)', linewidth=1.0)
axs_comp[1].set_xlabel('Tiempo (s)')
axs_comp[1].set_ylabel('Error [y_for - y_lfilter]')
axs_comp[1].set_title(f'Error de Filtrado Numérico (MSE = {mse:.2e})')
axs_comp[1].grid(True)
axs_comp[1].legend(loc='upper right')

plt.xlim([t_inicio, t_fin])
plt.tight_layout()
# ==========================================
# 7. FIGURA 3: ESPECTRO FFT LIMPIO (AUDIO BASE, LÍMITE 0 - 130 Hz)
# ==========================================
# FFT del audio base original sin modular
fft_x = np.abs(np.fft.rfft(x_completo))

# Filtrado sobre el audio base para observar la banda pasante
y40_man_base = aplicar_filtro_ciclo_for(x_completo, b1_man, a1_man)
y40_mat_base = aplicar_filtro_ciclo_for(x_completo, b1_mat, a1_mat)
y40_py_base  = aplicar_filtro_ciclo_for(x_completo, b1_py, a1_py)

y100_man_base = aplicar_filtro_ciclo_for(x_completo, b2_man, a2_man)
y100_mat_base = aplicar_filtro_ciclo_for(x_completo, b2_mat, a2_mat)
y100_py_base  = aplicar_filtro_ciclo_for(x_completo, b2_py, a2_py)

freqs = np.fft.rfftfreq(N, d=1/fs)

# ÚNICA VENTANA CON 2 SUBPLOTS
fig3, axs_fft = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# --- Subplot Superior: fc = 40 Hz ---
axs_fft[0].plot(freqs, fft_x, color='gray', alpha=0.3, label='Audio Original (Sin modular)')
axs_fft[0].plot(freqs, np.abs(np.fft.rfft(y40_man_base)), color='red', label='Cálculo manual (2º Orden, A=2)', linewidth=1.8)
axs_fft[0].plot(freqs, np.abs(np.fft.rfft(y40_mat_base)), color='black', linestyle='--', label='MATLAB (4º Orden, A=2)', linewidth=1.5)
axs_fft[0].plot(freqs, np.abs(np.fft.rfft(y40_py_base)), color='cyan', linestyle=':', label='SciPy (4º Orden, A=2)', linewidth=1.5)
axs_fft[0].axvline(40, color='red', linestyle='--', alpha=0.8, label='Corte fc = 40 Hz')
axs_fft[0].set_ylabel('Magnitud')
axs_fft[0].set_title('Espectro FFT - Respuesta del Filtro Pasa Bajos fc = 40 Hz')
axs_fft[0].grid(True)
axs_fft[0].legend(loc='upper right')

# --- Subplot Inferior: fc = 100 Hz ---
axs_fft[1].plot(freqs, fft_x, color='gray', alpha=0.3, label='Audio Original (Sin modular)')
axs_fft[1].plot(freqs, np.abs(np.fft.rfft(y100_man_base)), color='blue', label='Cálculo manual (2º Orden, A=2)', linewidth=1.8)
axs_fft[1].plot(freqs, np.abs(np.fft.rfft(y100_mat_base)), color='black', linestyle='--', label='MATLAB (4º Orden, A=2)', linewidth=1.5)
axs_fft[1].plot(freqs, np.abs(np.fft.rfft(y100_py_base)), color='orange', linestyle=':', label='SciPy (4º Orden, A=2)', linewidth=1.5)
axs_fft[1].axvline(100, color='blue', linestyle='--', alpha=0.8, label='Corte fc = 100 Hz')
axs_fft[1].set_xlabel('Frecuencia [Hz]')
axs_fft[1].set_ylabel('Magnitud')
axs_fft[1].set_title('Espectro FFT - Respuesta del Filtro Pasa Bajos fc = 100 Hz')
axs_fft[1].grid(True)
axs_fft[1].legend(loc='upper right')

# Límite ajustado a 130 Hz
plt.xlim([0, 130])
plt.tight_layout()

plt.show()