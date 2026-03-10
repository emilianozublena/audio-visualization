# Visualizador de Espacio Tímbrico

Un script en Python que analiza audio y genera una **visualización 3D animada** de sus características tímbricas. El resultado es una nube de puntos que traza un camino a través del "espacio tímbrico" — un sistema de coordenadas 3D donde cada eje representa una característica acústica diferente extraída de la señal de audio.

---

## ¿Qué es el "Espacio Tímbrico"?

Todo sonido tiene un **timbre** — la cualidad que permite distinguir una flauta de una guitarra incluso cuando tocan la misma nota al mismo volumen. El timbre está determinado por la distribución de energía a lo largo de las frecuencias y cómo esta cambia en el tiempo.

Este script toma tres aspectos medibles del timbre y los mapea a los tres ejes de un espacio 3D:

| Eje | Característica | Qué mide |
|-----|---------------|----------|
| **X** | Centroide Espectral (kHz) | El "centro de masa" del espectro de frecuencias. Valores altos significan que el sonido es más **brillante** (más energía en altas frecuencias); valores bajos significan que es más **oscuro**. Un platillo tiene un centroide alto; un bombo tiene uno bajo. |
| **Y** | Ancho de Banda Espectral (kHz) | Qué tan **dispersa** está la energía alrededor del centroide. Una onda sinusoidal pura tiene ancho de banda casi nulo (toda la energía en una frecuencia). Un sonido ruidoso como el ruido blanco tiene un ancho de banda muy alto. También se le llama **dispersión espectral**. |
| **Z** | Sonoridad RMS (dBFS) | El **nivel de energía** general del sonido en cada momento, medido en decibeles relativos a escala completa. Los momentos más fuertes están más arriba en este eje; el silencio cae al fondo. Esto se correlaciona con la **intensidad** percibida. |
| **Color** | Las tres características | Cada característica controla un canal de color HSV: centroide → **tono** (rojo=oscuro, azul=brillante), ancho de banda → **saturación** (enfocado=apagado, disperso=vívido), sonoridad → **valor** (suave=tenue, fuerte=brillante). Así el color refuerza la misma información que la posición. |

A medida que el audio se reproduce, cada cuadro de análisis se convierte en un **punto** posicionado en este espacio 3D. Los puntos consecutivos se conectan con una **línea**, formando una trayectoria — el camino que el sonido traza a través del espacio tímbrico en el tiempo.

---

## Cómo Funciona el Análisis de Audio

El script usa [librosa](https://librosa.org/), una biblioteca de Python ampliamente utilizada para análisis de audio y música.

### 1. Carga del audio

```python
y, sr = librosa.load(audio_path, sr=None, mono=True)
```

- `y` es un array 1D de numpy con las muestras de audio (convertido a mono si es estéreo).
- `sr` es la tasa de muestreo (ej. 44100 Hz). Usar `sr=None` preserva la tasa nativa del archivo en lugar de remuestrear.

### 2. Extracción de características cuadro a cuadro

El audio se divide en cuadros superpuestos usando una ventana deslizante. Dos parámetros controlan esto:

- **`N_FFT = 2048`** — El tamaño de cada ventana FFT en muestras. A 44100 Hz, esto equivale a unos 46 ms de audio. Ventanas más grandes dan mejor resolución en frecuencia pero peor resolución temporal.
- **`HOP_LENGTH = 512`** — Cuánto avanza la ventana entre cuadros, en muestras. A 44100 Hz, esto significa un cuadro de análisis cada ~11.6 ms, dando aproximadamente 86 cuadros por segundo de audio.

Para cada cuadro se calculan tres características:

#### Centroide Espectral

```python
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
```

El centroide espectral es la media ponderada de las frecuencias presentes en la señal, donde los pesos son las magnitudes de cada bin de frecuencia. Se calcula como:

```
centroide = sum(frecuencia[k] * magnitud[k]) / sum(magnitud[k])
```

El resultado está en Hz y se convierte a kHz para la visualización. Pensalo como "¿dónde está ubicada la energía promedio de frecuencia?" — un solo número que resume la brillantez del sonido en ese momento.

#### Ancho de Banda Espectral

```python
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048, hop_length=512)
```

El ancho de banda espectral (o dispersión espectral) mide qué tan amplia es la distribución de frecuencias alrededor del centroide. Es la desviación estándar ponderada de las frecuencias:

```
ancho_de_banda = sqrt( sum(magnitud[k] * (frecuencia[k] - centroide)^2) / sum(magnitud[k]) )
```

Un sonido tonal con energía concentrada en una sola frecuencia tiene un ancho de banda bajo. Un sonido ruidoso de banda ancha tiene un ancho de banda alto.

#### Energía RMS (Sonoridad)

```python
rms = librosa.feature.rms(y=y, hop_length=512)
rms_db = librosa.amplitude_to_db(rms, ref=np.max)
```

La energía RMS (Root Mean Square / Raíz Cuadrada de la Media) es la medida estándar de la amplitud de la señal:

```
rms = sqrt( (1/N) * sum(muestra[k]^2) )
```

Luego se convierte a decibeles relativos al valor RMS máximo del archivo (`ref=np.max`), de modo que el momento más fuerte queda en 0 dBFS y los momentos más suaves son negativos.

### 3. Alineación temporal

```python
times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP_LENGTH)
```

Cada cuadro de análisis se mapea a una marca de tiempo (en segundos) para que la visualización pueda sincronizarse con la reproducción del audio.

---

## Cómo Funciona la Visualización

La visualización está construida con [matplotlib](https://matplotlib.org/) usando sus capacidades de proyección 3D y `FuncAnimation` para la animación cuadro a cuadro.

### Elementos visuales

Cada cuadro de animación dibuja tres cosas:

1. **Línea de estela** — Una línea blanca delgada que conecta los últimos `TRAIL_LENGTH` (70) cuadros de análisis. Muestra la trayectoria reciente a través del espacio tímbrico.

2. **Puntos de estela** — Puntos de dispersión coloreados a lo largo de la estela. Su tamaño crece desde los más antiguos (pequeños, 6px) hasta los más nuevos (grandes, 38px), creando un efecto de "cola de cometa". El color se deriva de las tres características mediante el mapeo HSV (ver tabla arriba).

3. **Punto actual** — Un punto blanco más grande (110px) con un borde sutil que marca la posición actual en el espacio tímbrico.

### Rotación de cámara

La cámara 3D rota lentamente alrededor de la escena a `ROTATE_SPEED` (0.35) grados por cuadro de animación, dando una sensación de profundidad a medida que la trayectoria se construye.

### Tema oscuro

Todo el gráfico usa un esquema de colores oscuros (fondo `#080810`) con líneas de grilla sutiles, diseñado para que los puntos de datos coloreados resalten.

---

## Tres Scripts

### 1. `visualize.py` — Visualización de un archivo

```bash
python visualize.py canto_pajaro.wav                    # ventana en vivo + audio
python visualize.py canto_pajaro.wav --export salida.mp4   # guardar MP4 con audio
```

- **Modo en vivo** (por defecto): Abre una ventana interactiva de matplotlib y reproduce el audio por los parlantes usando [sounddevice](https://python-sounddevice.readthedocs.io/). La animación se sincroniza con el **tiempo real**, así que incluso si se pierden cuadros, el punto se mantiene sincronizado con lo que se escucha. Se puede rotar/hacer zoom en el gráfico 3D con el mouse.
- **Modo exportación** (`--export`): Renderiza los cuadros de forma determinística, envía el video crudo a `ffmpeg`, y luego combina el audio original encima (AAC a 192 kbps). No se abre ninguna ventana.

### 2. `compare.py` — Comparación de dos archivos

```bash
python compare.py pajaro1.wav pajaro2.wav                        # ventana en vivo
python compare.py pajaro1.wav pajaro2.wav --export comparacion.mp4  # guardar MP4
```

- Ambas pistas se animan **simultáneamente en el mismo espacio 3D** con escalas de ejes compartidas para que sus trayectorias sean directamente comparables.
- Cada pista tiene su propia **paleta de colores** — Pista A en cian/azul, Pista B en naranja/rojo — para poder distinguirlas de un vistazo.
- En modo exportación, ambas pistas de audio se mezclan juntas en el stream de audio del MP4.

### 3. `live.py` — Visualización en tiempo real desde el micrófono

```bash
python live.py                    # escuchar indefinidamente
python live.py --duration 30      # parar después de 30 segundos
python live.py --window 15        # ventana deslizante de 15 segundos
```

- Captura audio desde el **dispositivo de entrada por defecto** (micrófono) y anima el espacio tímbrico en tiempo real.
- Usa una ventana deslizante (10 segundos por defecto) — los datos viejos se desplazan a medida que llegan datos nuevos.
- Los límites de los ejes se **auto-escalan** suavemente usando un promedio móvil exponencial, así el gráfico se adapta a lo que sea que se esté grabando sin saltar bruscamente.
- Un indicador rojo pulsante "LIVE" confirma que el sistema está escuchando.

---

## Configuración

Todos los parámetros ajustables están definidos como constantes al inicio de cada script:

| Constante | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `HOP_LENGTH` | `512` | Salto de la ventana de análisis en muestras. Menor = más cuadros = animación más suave pero exportación más lenta. |
| `N_FFT` | `2048` | Tamaño de la ventana FFT. Mayor = mejor resolución en frecuencia. |
| `TRAIL_LENGTH` | `70` | Número de cuadros de análisis pasados que se muestran en la "cola de cometa". |
| `FPS` | `30` | Tasa de cuadros de la animación. |
| `ROTATE_SPEED` | `0.35` | Velocidad de rotación de la cámara en grados por cuadro. |
| `BG_COLOR` | `#080810` | Color de fondo (casi negro con un toque de azul). |
| `GRID_COLOR` | `#14142a` | Color de la grilla/bordes del panel 3D. |
| `SENSITIVITY` | `2.0` | Exponente de la curva gamma (>1 amplifica diferencias sutiles en las características). |
| `PCT_LO` / `PCT_HI` | `2` / `98` | Percentiles usados para el estiramiento de contraste (recorta outliers para que los datos llenen el espacio). |

---

## Instalación

### Prerrequisitos

- **Python 3.10+**
- **ffmpeg** (para exportar MP4): `brew install ffmpeg`
- **portaudio** (para reproducción de audio en vivo): `brew install portaudio`

### Configuración

```bash
# Crear y activar un entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias

| Paquete | Propósito |
|---------|-----------|
| `librosa` | Carga de audio y extracción de características espectrales |
| `numpy` | Arrays numéricos y operaciones |
| `matplotlib` | Gráficos 3D y animación |
| `sounddevice` | Reproducción de audio en tiempo real (opcional — el script corre en silencio sin él) |

---

## Ejemplos de Uso

```bash
# Archivo individual — visualización en vivo con reproducción de audio
python visualize.py grabacion.wav

# Archivo individual — exportar como video MP4 con audio
python visualize.py grabacion.wav --export espacio_timbrico.mp4

# Comparar dos archivos lado a lado
python compare.py pajaro1.wav pajaro2.wav
python compare.py pajaro1.wav pajaro2.wav --export comparacion.mp4

# Visualización en tiempo real desde el micrófono
python live.py
python live.py --duration 60 --window 20

# Funciona con cualquier formato que librosa soporte
python visualize.py canto.mp3
python visualize.py grabacion_campo.flac
```

---

## Cómo Interpretar la Visualización

- **Agrupaciones compactas** significan que el timbre del sonido es estable (ej. una nota sostenida).
- **Estelas largas y rápidas** significan que el timbre está cambiando rápidamente (ej. el trino de un pájaro barriendo frecuencias).
- **Arriba en el eje Z** = momentos fuertes; **abajo** = silencio/quietud.
- **A la derecha en el eje X** = sonidos brillantes/agudos; **a la izquierda** = sonidos oscuros/graves.
- **Arriba en el eje Y** = ruidoso/banda ancha; **abajo** = tonal/banda estrecha.
- **Colores vívidos** = espectro disperso (ruidoso/rico); **colores apagados** = espectro enfocado (tonal/puro).
- **Puntos brillantes** = momentos fuertes; **puntos tenues** = momentos suaves.
- **Cambios de tono** siguen la brillantez — tonos rojizos para sonidos oscuros, tonos azulados para sonidos brillantes.

Para el canto de pájaros específicamente, se suelen ver saltos y espirales rápidas a medida que el pájaro modula tono, volumen y contenido armónico en rápida sucesión — creando una trayectoria compleja y dinámica a través del espacio tímbrico.

En **modo comparación**, se puede ver cómo difieren dos grabaciones: si la trayectoria de un pájaro ocupa una región diferente o traza una forma distinta, eso revela diferencias concretas en su carácter tímbrico.
