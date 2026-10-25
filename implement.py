import os
from clase import GestorSeries, buscarCarpetasDicom


def entero(mensaje, minimo=None, maximo=None):
    
    while True:
        try:
            valor = int(input(mensaje))

            if minimo is not None and valor < minimo:
                raise ValueError
            if maximo is not None and valor > maximo:
                raise ValueError

            return valor

        except ValueError:
            if minimo is not None and maximo is not None:
                print("Ingrese un entero entre {} y {}.".format(
                    minimo, maximo
                ))
            elif minimo is not None:
                print("Ingrese un entero mayor o igual a {}.".format(
                    minimo
                ))
            else:
                print("Ingrese un número entero válido.")


def decimal(mensaje):
    while True:
        try:
            valor = float(input(mensaje).replace(",", "."))
            if not (valor == valor and abs(valor) != float("inf")):
                raise ValueError
            return valor
        except ValueError:
            print("Ingrese un número válido.")


def elegirSerie(gestor):
    nombres = gestor.verNombres()

    if not nombres:
        print("No hay series cargadas. Use primero la opción 1 o la 7.")
        return None

    print("\nSeries cargadas:")
    for i, nombre in enumerate(nombres):
        print("{} - {}".format(i, nombre))

    indice = entero(
        "Seleccione una serie: ", 0, len(nombres) - 1
    )
    return gestor.verSerie(nombres[indice])


def elegirOrientacion():
    print(
        "1 - Transversal/axial\n"
        "2 - Coronal\n"
        "3 - Sagital"
    )

    opcion = entero("Orientación: ", 1, 3)

    return {
        1: "transversal",
        2: "coronal",
        3: "sagital"
    }[opcion]


def cantidadCortes(serie, orientacion):
    ejes = {
        "transversal": 0,
        "coronal": 1,
        "sagital": 2
    }
    return serie.verVolumenCorregido().shape[ejes[orientacion]]


def cargarCarpetasClase(gestor):
    
    base = input(
        "Ruta de la carpeta 'datos' (Enter = ./datos): "
    ).strip().strip('"').strip("'")

    if not base:
        base = "datos"

    if not os.path.isdir(base):
        print("La carpeta indicada no existe.")
        return

    for nombre in ("PPMI", "Sarcoma", "T2"):
        ruta = None

        for carpeta in sorted(os.listdir(base)):
            completa = os.path.join(base, carpeta)
            if os.path.isdir(completa) and nombre.lower() in carpeta.lower():
                ruta = completa
                break

        if ruta is None:
            print("No se encontró la carpeta {} dentro de {}.".format(
                nombre, base
            ))
            continue

        hojas = buscarCarpetasDicom(ruta)

        if not hojas:
            print("No hay archivos DICOM dentro de {}.".format(ruta))
            continue

        for hoja in hojas:
            if len(hojas) == 1:
                clave = nombre
            else:
                relativa = os.path.relpath(hoja, ruta)
                clave = "{}_{}".format(
                    nombre, relativa.replace(os.sep, "_")
                )

            try:
                serie = gestor.cargarCarpeta(clave, hoja)
                print("Serie {} cargada correctamente.".format(clave))
                print(serie)

            except Exception as error:
                print("No se pudo cargar {}: {}".format(clave, error))


def main():
    gestor = GestorSeries()

    while True:
        print(
            "\n========== MENÚ PRINCIPAL ==========\n"
            "1 - Cargar carpeta DICOM\n"
            "2 - Ver información de una serie\n"
            "3 - Corregir ángulo de una serie\n"
            "4 - Ecualizar corte y guardar histogramas\n"
            "5 - Segmentar corte, dibujar círculo y estimar área\n"
            "6 - Ver series cargadas\n"
            "7 - Cargar PPMI, Sarcoma y T2 \n"
            "8 - Salir"
        )

        opcion = entero("--> ", 1, 8)

        # --------------------------------------------------
        if opcion == 1:
            ruta = input(
                "Ruta de la carpeta DICOM: "
            ).strip().strip('"').strip("'")

            if not os.path.isdir(ruta):
                print("La carpeta indicada no existe.")
                continue

            nombre = input(
                "Nombre para identificar la serie "
                "(PPMI/Sarcoma/T2): "
            ).strip()

            if not nombre:
                nombre = os.path.basename(
                    os.path.normpath(ruta)
                ) or "Serie"

            try:
                serie = gestor.cargarCarpeta(nombre, ruta)
                print("Serie cargada correctamente.")
                print(serie)

            except Exception as error:
                print("No se pudo cargar la carpeta: {}".format(error))
        # --------------------------------------------------
        elif opcion == 2:
            serie = elegirSerie(gestor)

            if serie is not None:
                print(serie.verInformacion())
        # --------------------------------------------------
        elif opcion == 3:
            serie = elegirSerie(gestor)

            if serie is not None:
                grados = decimal(
                    "Ángulo de rotación en grados: "
                )

                try:
                    serie.corregirAngulo(grados)
                    print(
                        "Corrección aplicada y guardada "
                        "en el objeto."
                    )

                except Exception as error:
                    print(
                        "No se pudo corregir el ángulo: {}".format(
                            error
                        )
                    )
        # --------------------------------------------------
        elif opcion == 4:
            serie = elegirSerie(gestor)

            if serie is not None:
                orientacion = elegirOrientacion()
                cantidad = cantidadCortes(serie, orientacion)

                indice = entero(
                    "Índice del corte (0 a {}): ".format(
                        cantidad - 1
                    ),
                    0,
                    cantidad - 1
                )

                try:
                    imagen, grafico = serie.ecualizarCorte(
                        orientacion, indice
                    )

                    print("Imagen ecualizada: {}".format(imagen))
                    print("Gráfico de histogramas: {}".format(grafico))

                except Exception as error:
                    print(
                        "No se pudo ecualizar el corte: {}".format(
                            error
                        )
                    )
        # --------------------------------------------------
        elif opcion == 5:
            serie = elegirSerie(gestor)

            if serie is not None:
                orientacion = elegirOrientacion()
                cantidad = cantidadCortes(serie, orientacion)

                indice = entero(
                    "Índice del corte (0 a {}): ".format(
                        cantidad - 1
                    ),
                    0,
                    cantidad - 1
                )

                umbral = entero(
                    "Umbral uint8 (0 a 255): ", 0, 255
                )

                kernel = entero(
                    "Tamaño del kernel impar (3, 5, 7...): ",
                    1
                )

                while kernel % 2 == 0:
                    print("El kernel debe ser impar.")
                    kernel = entero(
                        "Tamaño del kernel impar: ", 1
                    )

                iteraciones = entero(
                    "Iteraciones de apertura: ", 1
                )

                invertir = (
                    input(
                        "¿Segmentar intensidades oscuras? (s/n): "
                    ).strip().lower() == "s"
                )

                try:
                    resultado = serie.segmentarCorte(
                        orientacion,
                        indice,
                        umbral,
                        kernel,
                        iteraciones,
                        invertir
                    )

                    print(
                        "Imagen segmentada: {}".format(
                            resultado["imagen"]
                        )
                    )
                    print(
                        "Máscara: {}".format(
                            resultado["mascara"]
                        )
                    )
                    print(
                        "Área aproximada: {:.2f} mm²".format(
                            resultado["area_mm2"]
                        )
                    )
                    print(
                        "Dimensiones aproximadas "
                        "(ancho x alto): {:.2f} x {:.2f} mm".format(
                            *resultado["dimensiones_mm"]
                        )
                    )

                    if resultado["toca_borde"]:
                        print(
                            "Aviso: todos los elementos tocan el borde "
                            "de la imagen; se usó el de mayor área."
                        )

                except Exception as error:
                    print(
                        "No se pudo segmentar: {}".format(error)
                    )

        # --------------------------------------------------
        elif opcion == 6:
            series = gestor.verSeries()

            if not series:
                print("No hay series cargadas.")
            else:
                print("\n===== SERIES CARGADAS =====")

                for nombre, serie in series.items():
                    print(
                        "- {}: {} cortes, forma {}".format(
                            nombre,
                            len(serie.verMatrices()),
                            serie.verVolumen().shape
                        )
                    )
        # --------------------------------------------------
        elif opcion == 7:
            cargarCarpetasClase(gestor)

    
        elif opcion == 8:
            print("Hasta luego.")
            break


if __name__ == "__main__":
    main()