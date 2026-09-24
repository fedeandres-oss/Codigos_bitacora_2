import numpy as np
import matplotlib.pyplot as plt
import librosa

# ==========================================
# 0. PARÁMETROS DE CONFIGURACIÓN Y VISUALIZACIÓN
# ==========================================
duracion_audio = 3.0   # Duración total a cargar (segundos)
t_inicio = 1.0         # Tiempo inicial para la gráfica (s)
t_fin = 1.015         # Tiempo final para la gráfica (s) [e.g., zoom de 0.2 ms]

fs_target = 500000     # Frecuencia de muestreo (500 kHz)
f_corte = 50000        # Frecuencia de portadora (50 kHz)
offset = 1.5           # DC Offset para AM de onda completa
archivo_audio = 'gato.mp3'

# ==========================================
# 1. CARGA Y RE-MUESTREO
# ==========================================
# Cargar audio ajustado a la duración seleccionada
y_entrada, _ = librosa.load(archivo_audio, sr=fs_target, duration=duracion_audio)

# Normalización entre -1 y 1
y_entrada = y_entrada / np.max(np.abs(y_entrada))

dt = 1 / fs_target
N = len(y_entrada)
t = np.arange(0, N) * dt

# ==========================================
# 2. MODULACIÓN DE ONDA COMPLETA CON OFFSET
# ==========================================
portadora = np.cos(2 * np.pi * f_corte * t)
y_modulada = (offset + y_entrada) * portadora

# ==========================================
# 3. GRAFICACIÓN PARAMETRIZADA
# ==========================================
plt.figure(figsize=(10, 4))

plt.plot(t, y_modulada, color='tab:purple', label='Señal Modulada AM (50 kHz)')

plt.title(f'Modulación AM Onda Completa ({t_inicio} s a {t_fin} s)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')

# Definición dinámica de los límites del eje X
plt.xlim([t_inicio, t_fin])

plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()