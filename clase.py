import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread

plt.ion()

CARPETA = "resultados"      

# Funciones auxiliares

def nombreSeguro(texto):
    """Deja solo letras, números, guion y guion bajo para nombres de archivo."""
    limpio = "".join(c if (c.isascii() and c.isalnum()) or c in "-_" else "_"
                     for c in str(texto))
    while "__" in limpio:
        limpio = limpio.replace("__", "_")
    return limpio.strip("_")


def rutaResultado(nombre):
    """Crea la carpeta de resultados (si no existe) y devuelve la ruta."""
    os.makedirs(CARPETA, exist_ok=True)
    return os.path.join(CARPETA, nombre)


def guardarFigura(nombre):
    """Guarda la figura activa (png) dentro de CARPETA y devuelve la ruta."""
    ruta = rutaResultado(nombre)
    plt.savefig(ruta, dpi=150)
    return ruta


def guardarImagen(nombre, imagen):
    """Guarda una imagen con OpenCV dentro de CARPETA y devuelve la ruta."""
    ruta = rutaResultado(nombre)
    if not cv2.imwrite(ruta, imagen):
        raise ValueError("No se pudo guardar la imagen en {}".format(ruta))
    return ruta


def mostrarFigura():
    """Muestra la figura sin bloquear el menú y deja que se dibuje."""
    plt.show()
    plt.pause(0.1)


def normalizarUint8(corte):
    """Normaliza un corte (cualquier rango) a uint8 entre 0 y 255."""
    corte = np.asarray(corte, dtype=np.float32)
    if float(corte.max()) == float(corte.min()):
        return np.zeros(corte.shape, dtype=np.uint8)
    return cv2.normalize(corte, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def aplicarAspecto(imagen, aspecto, interpolacion=cv2.INTER_LINEAR):
    """Estira las filas de la imagen según el aspect (alto_mm / ancho_mm)."""
    filas, columnas = imagen.shape[:2]
    nuevasFilas = max(1, int(round(filas * aspecto)))
    return cv2.resize(imagen, (columnas, nuevasFilas),
                      interpolation=interpolacion)


def numeroInstancia(ds):
    """Número de corte del dataset (para ordenar los cortes de la serie)."""
    try:
        return int(ds.InstanceNumber)
    except (AttributeError, TypeError, ValueError):
        return 0


def carpetaTieneDicom(ruta):
    """True si la carpeta contiene (directamente) al menos un archivo DICOM."""
    for archivo in sorted(os.listdir(ruta)):
        rutaArchivo = os.path.join(ruta, archivo)
        if not os.path.isfile(rutaArchivo):
            continue
        try:
            ds = dcmread(rutaArchivo, stop_before_pixels=True)
        except Exception:
            continue
        if hasattr(ds, "Rows"):
            return True
    return False


def buscarCarpetasDicom(ruta):

    encontradas = []
    for raiz, subcarpetas, _ in os.walk(ruta):
        subcarpetas.sort()                  
        if carpetaTieneDicom(raiz):
            encontradas.append(raiz)
    return encontradas

class SerieDicom:
    ORIENTACIONES = ("transversal", "coronal", "sagital")

    def __init__(self, nombre, ruta):
        self.__nombre = nombre
        self.__ruta = ruta

        # Información extraída directamente del dataset
        self.__nombrePaciente = ""
        self.__modalidad = ""
        self.__filas = 0
        self.__columnas = 0
        self.__voxel = (1.0, 1.0, 1.0)           
        self.__voxelEscalado = (1.0, 1.0, 1.0)   
        self.__matrices = []                    

        self.__volumen = None                   
        self.__volumenCorregido = None           
        self.__angulo = 0.0
        self.__matrizRotacion = None

        self.__cargar(ruta)
        self.__reconstruir()

    def __leerArchivos(self, ruta):
        """Lee los DICOM con imagen de una carpeta (sin entrar a subcarpetas)."""
        datasets = []
        for archivo in sorted(os.listdir(ruta)):
            rutaArchivo = os.path.join(ruta, archivo)
            if not os.path.isfile(rutaArchivo):
                continue
            try:
                ds = dcmread(rutaArchivo)
            except Exception:
                continue                         
            if hasattr(ds, "PixelData"):
                datasets.append(ds)
        return datasets

    def __cargar(self, ruta):
        datasets = self.__leerArchivos(ruta)

        if not datasets:
            for raiz, subcarpetas, _ in os.walk(ruta):
                subcarpetas.sort()
                if raiz == ruta:
                    continue
                datasets = self.__leerArchivos(raiz)
                if datasets:
                    self.__ruta = raiz
                    break

        if not datasets:
            raise ValueError("No se encontraron archivos DICOM con imagen.")

        datasets.sort(key=numeroInstancia)

        cortes = []
        for ds in datasets:
            pixeles = ds.pixel_array
            if pixeles.ndim == 2:
                cortes.append((ds, pixeles))
            elif pixeles.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 1:
                for corte in pixeles:
                    cortes.append((ds, corte))

        if not cortes:
            raise ValueError("Los archivos no contienen cortes en escala de grises.")

        formas = {}
        for _, matriz in cortes:
            formas[matriz.shape] = formas.get(matriz.shape, 0) + 1
        forma = max(formas, key=formas.get)
        cortes = [par for par in cortes if par[1].shape == forma]

        ref = cortes[0][0]

        # Información tomada directamente del dataset
        self.__filas = int(getattr(ref, "Rows", forma[0]))
        self.__columnas = int(getattr(ref, "Columns", forma[1]))
        self.__nombrePaciente = str(getattr(ref, "PatientName", "")).strip() or "Desconocido"
        self.__modalidad = str(getattr(ref, "Modality", "")).strip() or "Desconocida"

        try:
            alto = float(ref.PixelSpacing[0])     # separación entre filas
            largo = float(ref.PixelSpacing[1])    # separación entre columnas
        except (AttributeError, IndexError, TypeError, ValueError):
            alto, largo = 1.0, 1.0

        ancho = self.__espesor(ref, datasets)
        self.__voxel = (alto, largo, ancho)
        self.__voxelEscalado = (alto / 2.0, largo / 2.0, ancho)
        self.__matrices = [matriz for _, matriz in cortes]

    def __espesor(self, ref, datasets):
        """Ancho del voxel = SliceThickness; si falta, se usa la separación de cortes."""
        try:
            espesor = float(ref.SliceThickness)
            if espesor > 0:
                return espesor
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            z1 = float(datasets[0].ImagePositionPatient[2])
            z2 = float(datasets[1].ImagePositionPatient[2])
            if abs(z2 - z1) > 0:
                return abs(z2 - z1)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        return 1.0

    def __reconstruir(self):
       
        escalados = []
        for matriz in self.__matrices:
            if matriz.dtype not in (np.uint8, np.uint16, np.int16):
                matriz = matriz.astype(np.float32)
            grande = cv2.resize(matriz, None, fx=2, fy=2,
                                interpolation=cv2.INTER_LINEAR)
            escalados.append(grande)

        self.__volumen = np.stack(escalados, axis=0)       # (cortes, filas, columnas)
        self.__volumenCorregido = self.__volumen            # sin corrección aún
        self.__angulo = 0.0
        self.__matrizRotacion = None

    def verNombre(self):
        return self.__nombre

    def verRuta(self):
        return self.__ruta

    def verNombrePaciente(self):
        return self.__nombrePaciente

    def verModalidad(self):
        return self.__modalidad

    def verFilas(self):
        return self.__filas

    def verColumnas(self):
        return self.__columnas

    def verVoxel(self):
        return self.__voxel

    def verVoxelEscalado(self):
        return self.__voxelEscalado

    def verMatrices(self):
        return self.__matrices

    def verVolumen(self):
        return self.__volumen

    def verVolumenCorregido(self):
        return self.__volumenCorregido

    def verAngulo(self):
        return self.__angulo

    def verMatrizRotacion(self):
        return self.__matrizRotacion

    # ---------------- información ----------------
    def verInformacion(self):
        alto, largo, ancho = self.__voxel
        altoE, largoE, anchoE = self.__voxelEscalado
        return (
            "\n===== INFORMACIÓN DE LA SERIE =====\n"
            "Serie: {}\n"
            "Carpeta: {}\n"
            "Paciente: {}\n"
            "Modalidad: {}\n"
            "Filas x columnas (originales): {} x {}\n"
            "Cantidad de cortes: {}\n"
            "Voxel original (alto x largo x ancho): {:.4f} x {:.4f} x {:.4f} mm\n"
            "Voxel tras escalar x2 (alto x largo x ancho): {:.4f} x {:.4f} x {:.4f} mm\n"
            "Volumen 3D (cortes, filas, columnas): {}\n"
            "Volumen corregido: {} (ángulo aplicado: {} grados)".format(
                self.__nombre, self.__ruta, self.__nombrePaciente,
                self.__modalidad, self.__filas, self.__columnas,
                len(self.__matrices), alto, largo, ancho,
                altoE, largoE, anchoE, self.__volumen.shape,
                self.__volumenCorregido.shape, self.__angulo))

    def __str__(self):
        return self.verInformacion()

    # ---------------- utilidades de cortes ----------------
    def __validar(self, orientacion, indice):
        if orientacion not in self.ORIENTACIONES:
            raise ValueError("Orientación no válida: {}".format(orientacion))
        eje = self.ORIENTACIONES.index(orientacion)
        cantidad = self.__volumenCorregido.shape[eje]
        if indice < 0 or indice >= cantidad:
            raise ValueError("El índice debe estar entre 0 y {}.".format(cantidad - 1))

    def __extraerCorte(self, volumen, orientacion, indice):
        """Corte 2D del volumen. Ejes: 0 = transversal, 1 = coronal, 2 = sagital."""
        self.__validar(orientacion, indice)
        if orientacion == "transversal":
            corte = volumen[indice, :, :]
        elif orientacion == "coronal":
            corte = volumen[:, indice, :]
        else:
            corte = volumen[:, :, indice]
        return np.array(corte)

    def __espaciado(self, orientacion):
        """(alto_mm, ancho_mm) de un píxel del corte, ya con el escalado x2."""
        alto, largo, espesor = self.__voxelEscalado
        if orientacion == "transversal":
            return alto, largo            # filas = y, columnas = x
        if orientacion == "coronal":
            return espesor, largo         # filas = z, columnas = x
        return espesor, alto              # sagital: filas = z, columnas = y

    # ---------------- 4. corregir ángulo ----------------
    def corregirAngulo(self, grados):
       
        grados = float(grados)
        cortes, alto, ancho = self.__volumen.shape

        centro = (ancho / 2.0, alto / 2.0)                       
        matriz = cv2.getRotationMatrix2D(centro, grados, 1.0)    
        fondo = float(self.__volumen.min())

        corregido = np.empty_like(self.__volumen)
        for i in range(cortes):
            corregido[i] = cv2.warpAffine(
                self.__volumen[i], matriz, (ancho, alto),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=fondo)

        self.__volumenCorregido = corregido
        self.__angulo = grados
        self.__matrizRotacion = matriz
        return matriz

    # ---------------- 5. ecualizar un corte ----------------
    def ecualizarCorte(self, orientacion, indice):
        """Normaliza a uint8 un corte de una copia del volumen y lo ecualiza.

        Grafica histogramas e imágenes, y guarda el gráfico y la imagen.
        Devuelve (ruta_imagen, ruta_grafico).
        """
        copia = self.__volumenCorregido.copy()
        corte = self.__extraerCorte(copia, orientacion, indice)
        del copia

        original = normalizarUint8(corte)
        ecualizada = cv2.equalizeHist(original)

        alto, ancho = self.__espaciado(orientacion)
        aspecto = alto / ancho

        fig = plt.figure(figsize=(11, 8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        ax1.imshow(original, cmap="gray", aspect=aspecto, vmin=0, vmax=255)
        ax1.set_title("Corte {} {} (uint8)".format(orientacion, indice))

        ax2.imshow(ecualizada, cmap="gray", aspect=aspecto, vmin=0, vmax=255)
        ax2.set_title("Imagen ecualizada")

        ax3.hist(original.ravel(), bins=256, range=(0, 255), color="gray")
        ax3.set_yscale("log")
        ax3.set_xlabel("Nivel de gris")
        ax3.set_ylabel("Frecuencia (escala log)")
        ax3.set_title("Histograma original")

        ax4.hist(ecualizada.ravel(), bins=256, range=(0, 255), color="tab:blue")
        ax4.set_yscale("log")
        ax4.set_xlabel("Nivel de gris")
        ax4.set_ylabel("Frecuencia (escala log)")
        ax4.set_title("Histograma ecualizado")

        plt.tight_layout()

        base = "{}_{}_{}".format(nombreSeguro(self.__nombre), orientacion, indice)
        rutaGrafico = guardarFigura("{}_ecualizacion.png".format(base))
        mostrarFigura()

        rutaImagen = guardarImagen("{}_ecualizada.png".format(base),
                                   aplicarAspecto(ecualizada, aspecto))
        return rutaImagen, rutaGrafico

    # ---------------- 6. segmentar un corte ----------------
    def segmentarCorte(self, orientacion, indice, umbral, kernel,
                       iteraciones, invertir=False):
    
        if umbral < 0 or umbral > 255:
            raise ValueError("El umbral debe estar entre 0 y 255.")
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("El kernel debe ser un entero impar.")
        if iteraciones < 1:
            raise ValueError("Las iteraciones deben ser al menos 1.")

        corte = self.__extraerCorte(self.__volumenCorregido, orientacion, indice)
        gris = normalizarUint8(corte)

        # Binarización por umbral
        tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
        _, binaria = cv2.threshold(gris, umbral, 255, tipo)

        nucleo = np.ones((kernel, kernel), np.uint8)
        apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, nucleo,
                                    iterations=iteraciones)

        total, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(
            apertura, connectivity=8)
        if total < 2:
            raise ValueError("No se encontró ningún elemento de interés; "
                             "pruebe otro umbral, kernel o la opción de invertir.")

    
        filas, columnas = apertura.shape
        mejor = -1
        mejorArea = 0
        tocaBorde = False
        for e in range(1, total):
            x = stats[e, cv2.CC_STAT_LEFT]
            y = stats[e, cv2.CC_STAT_TOP]
            w = stats[e, cv2.CC_STAT_WIDTH]
            h = stats[e, cv2.CC_STAT_HEIGHT]
            area = stats[e, cv2.CC_STAT_AREA]
            borde = x == 0 or y == 0 or x + w >= columnas or y + h >= filas
            if not borde and area > mejorArea:
                mejor, mejorArea = e, area
        if mejor == -1:
            tocaBorde = True
            for e in range(1, total):
                area = stats[e, cv2.CC_STAT_AREA]
                if area > mejorArea:
                    mejor, mejorArea = e, area

        x = int(stats[mejor, cv2.CC_STAT_LEFT])
        y = int(stats[mejor, cv2.CC_STAT_TOP])
        w = int(stats[mejor, cv2.CC_STAT_WIDTH])
        h = int(stats[mejor, cv2.CC_STAT_HEIGHT])
        areaPx = int(stats[mejor, cv2.CC_STAT_AREA])

        # Conversión a milímetros con el espaciado del corte
        altoMm, anchoMm = self.__espaciado(orientacion)
        area_mm2 = areaPx * altoMm * anchoMm
        dimensiones = (w * anchoMm, h * altoMm)      

        aspecto = altoMm / anchoMm
        mascara = aplicarAspecto(apertura, aspecto, cv2.INTER_NEAREST)
        filasM, columnasM = mascara.shape

        rgb = cv2.cvtColor(mascara, cv2.COLOR_GRAY2RGB)
        centro = (x + w // 2, int(round((y + h / 2.0) * aspecto)))
        radio = int(np.hypot(w, h * aspecto) / 2) + 5
        grosor = max(2, min(filasM, columnasM) // 150)
        cv2.circle(rgb, centro, radio, (255, 128, 0), grosor)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.imshow(rgb, aspect="equal")
        ax.set_title("Segmentación - corte {} {} (umbral {}, kernel {}, "
                     "iteraciones {})".format(orientacion, indice, umbral,
                                              kernel, iteraciones))
        ax.set_xlabel("Columnas (píxeles)")
        ax.set_ylabel("Filas (píxeles, con aspect aplicado)")
        ax.text(0.02, 0.98,
                "Dimensiones: {:.2f} x {:.2f} mm\nÁrea: {:.2f} mm²".format(
                    dimensiones[0], dimensiones[1], area_mm2),
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="orange", alpha=0.9))
        plt.tight_layout()

        base = "{}_{}_{}".format(nombreSeguro(self.__nombre), orientacion, indice)
        rutaImagen = guardarFigura("{}_segmentacion.png".format(base))
        mostrarFigura()

        rutaMascara = guardarImagen("{}_mascara.png".format(base), mascara)

        return {
            "imagen": rutaImagen,
            "mascara": rutaMascara,
            "area_mm2": area_mm2,
            "area_pixeles": areaPx,
            "dimensiones_mm": dimensiones,
            "toca_borde": tocaBorde,
        }

class GestorSeries:
    def __init__(self):
        self.__series = {}

    def cargarCarpeta(self, nombre, ruta):
        """Carga una carpeta DICOM y la guarda con el nombre indicado."""
        serie = SerieDicom(nombre, ruta)
        self.__series[nombre] = serie
        return serie

    def verNombres(self):
        return list(self.__series.keys())

    def verSerie(self, nombre):
        if nombre not in self.__series:
            raise KeyError("No existe la serie '{}'.".format(nombre))
        return self.__series[nombre]

    def verSeries(self):
        return self.__series.copy()
    