# Visualizador de Espacio Timbrico

Un script en Python que analiza audio y genera una visualizacion 3D animada de sus caracteristicas timbricas. El resultado es una nube de puntos que traza un camino a traves del "espacio timbrico", un sistema de coordenadas 3D donde cada eje representa una caracteristica acustica distinta.

Desde lo que les mostre ayer hice algunos ajustes (como por ejemplo combinar las 3 características en la variabilidad del color, lo que hace que "la identidad del sonido", como concepto que unifica estas 3 cualidades acústicas, se vea reflejada en los colores).

---

## Que es el "Espacio Timbrico"?

El timbre es lo que nos permite distinguir una flauta de una guitarra aunque toquen la misma nota al mismo volumen. Depende de como se distribuye la energia en las frecuencias y de como eso cambia en el tiempo.

Este script toma tres aspectos medibles del timbre y los asigna a los tres ejes de un espacio 3D, no son los únicos pero me parecieron interesantes y distintivos:

| Eje | Caracteristica | Que mide |
|-----|---------------|----------|
| X | Centroide Espectral (kHz) | El "centro de masa" del espectro. Valores altos = sonido brillante (mas energia en agudos). Valores bajos = sonido oscuro. Un platillo tiene centroide alto; un bombo, bajo. |
| Y | Ancho de Banda Espectral (kHz) | Que tan dispersa esta la energia alrededor del centroide. Una sinusoidal pura tiene ancho de banda casi nulo. El ruido blanco tiene uno muy alto. |
| Z | Sonoridad RMS (dBFS) | El nivel de energia del sonido en cada momento, en decibeles. Los momentos fuertes quedan arriba; el silencio cae al fondo. |
| Color | Las tres juntas | Centroide controla el tono (rojo=oscuro, azul=brillante), ancho de banda controla la saturacion (enfocado=apagado, disperso=vivido), sonoridad controla el brillo del color (suave=tenue, fuerte=brillante). |

A medida que el audio avanza, cada cuadro de analisis se convierte en un punto en este espacio. Los puntos consecutivos se conectan con una linea, formando la trayectoria que el sonido traza a lo largo del tiempo.

---

## Como funciona el analisis

El script usa [librosa](https://librosa.org/) para el analisis de audio.

### 1. Carga del audio

```python
y, sr = librosa.load(audio_path, sr=None, mono=True)
```

- `y` es un array de numpy con las muestras de audio (convertido a mono si es estereo).
- `sr` es la tasa de muestreo (ej. 44100 Hz). Con `sr=None` se preserva la tasa nativa del archivo.

### 2. Extraccion de caracteristicas cuadro a cuadro

El audio se divide en cuadros superpuestos con una ventana deslizante. Los dos parametros que controlan esto son:

- `N_FFT = 2048`: tamanio de cada ventana FFT en muestras. A 44100 Hz son unos 46 ms de audio. Ventanas mas grandes dan mejor resolucion en frecuencia pero peor resolucion temporal.
- `HOP_LENGTH = 512`: cuanto avanza la ventana entre cuadros. A 44100 Hz, un cuadro cada ~11.6 ms, o sea ~86 cuadros por segundo.

Para cada cuadro se calculan tres cosas:

#### Centroide Espectral

```python
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
```

La media ponderada de las frecuencias presentes en la señal, usando las magnitudes como pesos:

```
sum(frecuencia[k] * magnitud[k]) / sum(magnitud[k])
```

El resultado esta en Hz y se convierte a kHz para mostrar. Es un numero que resume que tan brillante suena algo en ese instante.

#### Ancho de Banda Espectral

```python
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=2048, hop_length=512)
```

Mide que tan amplia es la distribucion de frecuencias alrededor del centroide. Es la desviacion estandar ponderada:

```
sqrt( sum(magnitud[k] * (frecuencia[k] - centroide)^2) / sum(magnitud[k]) )
```

Un sonido tonal tiene ancho de banda bajo. Un sonido ruidoso de banda ancha lo tiene alto.

#### Energia RMS (Sonoridad)

```python
rms = librosa.feature.rms(y=y, hop_length=512)
rms_db = librosa.amplitude_to_db(rms, ref=np.max)
```

La RMS (Root Mean Square) es la medida estandar de amplitud de señal:

```
rms = sqrt( (1/N) * sum(muestra[k]^2) )
```

Se convierte a decibeles relativos al maximo del archivo (`ref=np.max`), asi el momento mas fuerte queda en 0 dBFS y los mas suaves son negativos.

### 3. Alineacion temporal

```python
times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP_LENGTH)
```

Cada cuadro se mapea a un timestamp en segundos para sincronizar la visualizacion con la reproduccion.

---

## La visualizacion

Construida con [matplotlib](https://matplotlib.org/) usando proyeccion 3D y `FuncAnimation` para animar cuadro a cuadro.

### Elementos visuales

Cada cuadro de animacion dibuja:

1. Una linea blanca fina que conecta los ultimos 70 cuadros de analisis (la trayectoria reciente).
2. Puntos coloreados a lo largo de esa estela. Crecen en tamaño del mas viejo al mas nuevo, como una cola de cometa. El color sale del mapeo HSV descrito arriba.
3. Un punto blanco mas grande que marca la posicion actual.

### Rotacion de camara

La camara rota lentamente alrededor de la escena (0.35 grados por cuadro) para dar sensacion de profundidad.

### Tema oscuro

Fondo casi negro (`#080810`) con grilla sutil, para que los puntos de colores resalten.

---

## Los scripts

### 1. `visualize.py` — Un archivo

```bash
python visualize.py canto_pajaro.wav                       # ventana en vivo + audio
python visualize.py canto_pajaro.wav --export salida.mp4   # guardar MP4
```

- En vivo (por defecto): abre una ventana de matplotlib y reproduce el audio con [sounddevice](https://python-sounddevice.readthedocs.io/). La animacion se sincroniza con el reloj del sistema, asi que aunque se pierdan cuadros el punto sigue en sync con lo que se escucha. Se puede rotar y hacer zoom con el mouse.
- Exportacion (`--export`): renderiza los cuadros, los manda a `ffmpeg` y le agrega el audio original (AAC a 192 kbps). No abre ventana.

### 2. `compare.py` — Dos archivos

```bash
python compare.py pajaro1.wav pajaro2.wav                          # ventana en vivo
python compare.py pajaro1.wav pajaro2.wav --export comparacion.mp4 # guardar MP4
```

Las dos pistas se animan a la vez en el mismo espacio 3D con ejes compartidos, para poder compararlas directamente. Cada una tiene su paleta de colores (cian/azul vs naranja/rojo). En exportacion, los dos audios se mezclan en la pista de audio del MP4.

### 3. `live.py` — Microfono en tiempo real

```bash
python live.py                    # escuchar indefinidamente
python live.py --duration 30      # parar despues de 30 segundos
python live.py --window 15        # ventana deslizante de 15 segundos
```

Captura audio del microfono y anima el espacio timbrico en tiempo real. Usa una ventana deslizante (10 segundos por defecto) donde los datos viejos se van descartando. Los ejes se auto-escalan con un promedio movil exponencial para adaptarse al audio entrante sin saltos bruscos. Un indicador rojo pulsante confirma que esta escuchando.

### 4. `clean.py` — Sin ejes, solo la animacion

```bash
python clean.py canto_pajaro.wav                       # ventana en vivo + audio
python clean.py canto_pajaro.wav --export salida.mp4   # guardar MP4
```

Hace lo mismo que `visualize.py` pero sin ningun elemento de grafico: sin ejes, sin grilla, sin etiquetas, sin barra de color. Solo la nube de puntos, la linea y el fondo oscuro. Sirve para ver la trayectoria timbrica como algo mas visual y menos tecnico.

---

## Configuracion

Todos los parametros ajustables estan definidos como constantes al inicio de cada script:

| Constante | Default | Descripcion |
|-----------|---------|-------------|
| `HOP_LENGTH` | `512` | Salto de la ventana de analisis en muestras. Menor = mas cuadros = animacion mas suave pero exportacion mas lenta. |
| `N_FFT` | `2048` | Tamanio de ventana FFT. Mayor = mejor resolucion en frecuencia. |
| `TRAIL_LENGTH` | `70` | Cuantos cuadros pasados se muestran en la cola de cometa. |
| `FPS` | `30` | Cuadros por segundo de la animacion. |
| `ROTATE_SPEED` | `0.35` | Velocidad de rotacion de la camara (grados por cuadro). |
| `BG_COLOR` | `#080810` | Color de fondo. |
| `GRID_COLOR` | `#14142a` | Color de la grilla 3D. |
| `SENSITIVITY` | `2.0` | Exponente gamma. Valores >1 amplifican diferencias sutiles en las caracteristicas. |
| `PCT_LO` / `PCT_HI` | `2` / `98` | Percentiles para el estiramiento de contraste (recorta outliers). |

---

## Instalacion

### Prerrequisitos

- Python 3.10+
- ffmpeg (para exportar MP4): `brew install ffmpeg`
- portaudio (para reproduccion de audio): `brew install portaudio`

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencias

| Paquete | Para que |
|---------|----------|
| `librosa` | Carga de audio y extraccion de caracteristicas espectrales |
| `numpy` | Arrays y operaciones numericas |
| `matplotlib` | Graficos 3D y animacion |
| `sounddevice` | Reproduccion de audio en tiempo real (opcional) |

---

## Ejemplos

```bash
# Un archivo, visualizacion en vivo con audio
python visualize.py grabacion.wav

# Un archivo, exportar como MP4
python visualize.py grabacion.wav --export espacio_timbrico.mp4

# Comparar dos archivos
python compare.py pajaro1.wav pajaro2.wav
python compare.py pajaro1.wav pajaro2.wav --export comparacion.mp4

# Microfono en tiempo real
python live.py
python live.py --duration 60 --window 20

# Funciona con cualquier formato que soporte librosa
python visualize.py canto.mp3
python visualize.py grabacion_campo.flac
```

---

## Como leer la visualizacion

- Agrupaciones compactas = timbre estable (una nota sostenida, por ejemplo).
- Estelas largas y rapidas = timbre cambiando rapido (un trino de pajaro barriendo frecuencias).
- Arriba en Z = fuerte. Abajo = silencio.
- Derecha en X = brillante/agudo. Izquierda = oscuro/grave.
- Arriba en Y = ruidoso/banda ancha. Abajo = tonal/banda estrecha.
- Colores vividos = espectro disperso. Colores apagados = espectro enfocado.
- Puntos brillantes = momentos fuertes. Puntos tenues = momentos suaves.
- Tonos rojizos = sonidos oscuros. Tonos azulados = sonidos brillantes.

En el canto de pajaros se ven saltos y espirales rapidas porque el pajaro modula tono, volumen y armonicos en rapida sucesion.

En modo comparacion se puede ver si dos grabaciones ocupan regiones distintas del espacio o trazan formas diferentes, lo que revela diferencias concretas en caracter timbrico.

---

## Para leer mas

### Centroide Espectral

La media ponderada de las frecuencias en el espectro, usando magnitudes como pesos. Es el descriptor mas usado de brillantez espectral y se correlaciona con que tan "agudo" o "apagado" se percibe un sonido.

### Ancho de Banda Espectral

Mide la dispersion del espectro alrededor del centroide: la desviacion estandar ponderada de las frecuencias. Un tono puro tiene ancho de banda casi nulo; un sonido ruidoso o armonicamente rico lo tiene alto. Captura que tan enfocado o difuso es el contenido frecuencial.

### Energia RMS / Sonoridad

La medida estandar de amplitud a corto plazo. En decibeles, un aumento de 10 dB corresponde aproximadamente a duplicar la sonoridad percibida. Aca se usa como proxy de la sonoridad instantanea.

### Referencias

- Basso, G. (2006). *Percepcion auditiva*. Editorial Universidad Nacional de Quilmes. ISBN 978-987-558-082-4. Cubre percepcion de sonoridad, timbre y el oido como analizador espectral.
- Basso, G. (1999). *Analisis espectral: La transformada de Fourier en la musica*. EDULP, Universidad Nacional de La Plata. Analisis de Fourier aplicado a la musica, con las bases matematicas de la descomposicion espectral.
- Miyara, F. (2006). *Acustica y sistemas de sonido* (4ta ed.). UNR Editora, Universidad Nacional de Rosario. ISBN 978-950-673-557-9. Capitulo 2 sobre psicoacustica: sonoridad, escala de fonios, curvas de ponderacion, timbre y formantes.
- Roederer, J. G. (1997). *Acustica y psicoacustica de la musica*. Melos / Ricordi Americana. ISBN 978-987-611-219-2. Capitulos 3-4: fisica del sonido y percepcion psicoacustica.
- Peeters, G. (2004). *A large set of audio features for sound description*. Informe tecnico del IRCAM. Secciones 4.2-4.3 sobre centroide espectral, ancho de banda espectral y clasificacion timbrica.
- Grey, J. M. (1977). "Multidimensional perceptual scaling of musical timbres." *JASA*, 61(5), 1270-1277. Muestra que la distribucion de energia espectral es una dimension perceptual primaria del timbre.
- McAdams, S. et al. (1995). "Perceptual scaling of synthesized musical timbres." *Psychological Research*, 58, 177-192. La dispersion espectral como eje perceptual para diferenciar timbres.
- Fletcher, H. & Munson, W. A. (1933). "Loudness, its definition, measurement and calculation." *JASA*, 5(2), 82-108. El estudio clasico sobre percepcion de sonoridad y la relacion logaritmica (decibeles).
- Giannoulis, D., Massberg, M., & Reiss, J. D. (2012). "Digital dynamic range compressor design." *JAES*, 60(6), 399-408. Medicion RMS en procesamiento de audio.
- ITU-R BS.1770. Estandar internacional para medicion de sonoridad en radiodifusion.
- Documentacion de librosa: [spectral_centroid](https://librosa.org/doc/latest/generated/librosa.feature.spectral_centroid.html), [spectral_bandwidth](https://librosa.org/doc/latest/generated/librosa.feature.spectral_bandwidth.html), [rms](https://librosa.org/doc/latest/generated/librosa.feature.rms.html)
