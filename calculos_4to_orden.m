% ==========================================
% PARTE 2: Diseño de Filtro IIR 4to Orden (MATLAB)
% ==========================================
fs = 1000;          % Frecuencia de muestreo (Hz)
fc1 = 40;  
fc2= 100;% Frecuencia de corte (Hz)
orden = 4;

% Frecuencia normalizada Wn para Nyquist (Fs/2)
Wn1 = fc1 / (fs / 2);
Wn2 = fc2 / (fs / 2);
% Obtención de coeficientes Butterworth de 4to orden
[b1, a1] = butter(orden, Wn1, 'low');
[b2, a2] = butter(orden, Wn2, 'low');
format long;
disp('--- Copiar estos vectores a Python ---');
fprintf('b1_mat = 2 * np.array([%.15e, %.15e, %.15e, %.15e, %.15e])\n', b1);
fprintf('a1_mat = np.array([%.15e, %.15e, %.15e, %.15e, %.15e])\n', a1);
fprintf('b2_mat = 2 * np.array([%.15e, %.15e, %.15e, %.15e, %.15e])\n', b2);
fprintf('a2_mat = np.array([%.15e, %.15e, %.15e, %.15e, %.15e])\n', a2);