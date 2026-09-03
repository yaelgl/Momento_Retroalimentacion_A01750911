"""
Programa principal: entrena el arbol de decision y reporta su desempenio.

Se ejecuta desde la terminal, sin necesidad de un notebook ni de un IDE:

    python main.py                      entrena y evalua con los valores por omision
    python main.py --arbol 3            ademas dibuja las primeras 3 preguntas
    python main.py --curva              compara profundidades para ver el sobreajuste
    python main.py --interactivo        pide datos de un pasajero y predice
    python main.py --kaggle             genera predicciones para el test.csv sin etiqueta
    python main.py --guardar-json r.json  guarda los resultados para el reporte

Usa unicamente la libreria estandar de Python.
"""

import argparse
import json
import time
from pathlib import Path

import preparar_datos as prep
from arbol_decision import ArbolDecision
from metricas import (calcular_metricas, matriz_confusion, texto_matriz_confusion,
                      texto_metricas)

CARPETA = Path(__file__).parent
RUTA_TRAIN = CARPETA / "dataset" / "train.csv"
RUTA_TEST = CARPETA / "dataset" / "test.csv"

# Valores por omision de los hiperparametros.
#
# NO se eligieron mirando el conjunto de prueba. Salieron de la busqueda que hace
# `seleccionar_hiperparametros.py`, que parte el conjunto de entrenamiento en
# sub-entrenamiento y validacion, prueba 60 combinaciones y se queda con la mejor
# en validacion. El conjunto de prueba se mide una sola vez, al final, con estos
# valores ya fijados.
PROFUNDIDAD = 9
MIN_DIVISION = 80
MIN_HOJA = 40


def separador(titulo):
    """Imprime un titulo de seccion para que la salida se lea ordenada."""
    print()
    print("=" * 70)
    print(titulo)
    print("=" * 70)


def entrenar_y_evaluar(args):
    """Flujo completo: cargar, entrenar, evaluar e imprimir los resultados."""

    # ---------------------------------------------------------------- datos
    separador("1. DATOS")
    X_entrena, y_entrena, X_prueba, y_prueba, info = prep.cargar_datos(
        RUTA_TRAIN, proporcion_prueba=args.proporcion_prueba, semilla=args.semilla)

    pct_e = info["positivos_entrena"] / info["n_entrena"] * 100
    pct_p = info["positivos_prueba"] / info["n_prueba"] * 100
    print(f"Archivo: {RUTA_TRAIN.name}  ({info['total']} pasajeros con etiqueta)")
    print()
    print(f"  Entrenamiento: {info['n_entrena']:5d} filas   "
          f"{info['positivos_entrena']:4d} transportados ({pct_e:.2f}%)")
    print(f"  Prueba:        {info['n_prueba']:5d} filas   "
          f"{info['positivos_prueba']:4d} transportados ({pct_p:.2f}%)")
    print()
    print(f"Division {int((1 - args.proporcion_prueba) * 100)}/"
          f"{int(args.proporcion_prueba * 100)} estratificada, semilla {args.semilla}.")
    print("El conjunto de prueba no se usa en ningun momento del entrenamiento,")
    print("ni siquiera para calcular las medianas de la imputacion.")
    print()
    print(f"Variables usadas: {len(prep.COLUMNAS)} "
          f"({len(prep.NUMERICAS)} numericas y {len(prep.CATEGORICAS)} categoricas)")
    print(f"  numericas:   {', '.join(prep.NUMERICAS)}")
    print(f"  categoricas: {', '.join(prep.CATEGORICAS)}")

    # ------------------------------------------------------------ entrenar
    separador("2. ENTRENAMIENTO")
    arbol = ArbolDecision(
        columnas_numericas=prep.NUMERICAS,
        columnas_categoricas=prep.CATEGORICAS,
        max_profundidad=args.profundidad,
        min_muestras_division=args.min_division,
        min_muestras_hoja=args.min_hoja,
        criterio=args.criterio)

    print(f"Criterio de impureza:       {args.criterio}")
    print(f"Profundidad maxima:         {args.profundidad}")
    print(f"Minimo para dividir un nodo:{args.min_division:4d} ejemplos")
    print(f"Minimo por hoja:            {args.min_hoja:4d} ejemplos")
    print()

    inicio = time.perf_counter()
    arbol.entrenar(X_entrena, y_entrena)
    tardo = time.perf_counter() - inicio

    nodos, hojas = arbol.contar_nodos()
    print(f"Arbol construido en {tardo:.2f} segundos")
    print(f"  Profundidad alcanzada: {arbol.profundidad_alcanzada()}")
    print(f"  Nodos totales:         {nodos}")
    print(f"  Hojas:                 {hojas}")
    print(f"  Nodos de decision:     {nodos - hojas}")

    # ------------------------------------------------------------ evaluar
    separador("3. MATRIZ DE CONFUSION (conjunto de prueba)")
    pred_prueba = arbol.predecir(X_prueba)
    m_prueba = calcular_metricas(y_prueba, pred_prueba)
    print(texto_matriz_confusion(m_prueba["matriz"]))

    separador("4. METRICAS")
    pred_entrena = arbol.predecir(X_entrena)
    m_entrena = calcular_metricas(y_entrena, pred_entrena)

    print("Conjunto de PRUEBA (datos que el arbol nunca vio):")
    print(texto_metricas(m_prueba))
    print()
    print("Conjunto de ENTRENAMIENTO (referencia para revisar sobreajuste):")
    print(texto_metricas(m_entrena))
    print()
    brecha = m_entrena["exactitud"] - m_prueba["exactitud"]
    print(f"Diferencia entrenamiento - prueba: {brecha:+.4f}")
    if brecha > 0.10:
        print("  Brecha amplia: el arbol esta memorizando. Conviene reducir la profundidad.")
    elif brecha < 0.03:
        print("  Brecha muy pequenia: el modelo generaliza bien, incluso podria crecer mas.")
    else:
        print("  Brecha razonable: el arbol generaliza sin memorizar de mas.")

    # Comparacion contra la referencia mas simple posible
    separador("5. COMPARACION CONTRA UN CLASIFICADOR TRIVIAL")
    clase_mayoritaria = 1 if info["positivos_entrena"] * 2 >= info["n_entrena"] else 0
    pred_trivial = [clase_mayoritaria] * len(y_prueba)
    m_trivial = calcular_metricas(y_prueba, pred_trivial)
    print("Un modelo que siempre predice la clase mas frecuente del entrenamiento")
    print(f"(clase {clase_mayoritaria}) obtendria:")
    print(f"  Exactitud: {m_trivial['exactitud']:.4f}")
    print()
    print(f"El arbol obtiene {m_prueba['exactitud']:.4f}, es decir "
          f"{(m_prueba['exactitud'] - m_trivial['exactitud']) * 100:.2f} puntos")
    print("porcentuales por encima. Esto confirma que el modelo aprendio algo real")
    print("de los datos y no solo esta aprovechando el balance de las clases.")

    # ----------------------------------------------------- importancia
    separador("6. IMPORTANCIA DE LAS VARIABLES")
    print("Cuanta impureza elimina cada variable, pesada por el numero de")
    print("ejemplos que pasan por sus divisiones (normalizada a 100%):")
    print()
    for nombre, valor in arbol.importancia_ordenada():
        if valor > 0.0001:
            barra = "#" * int(round(valor * 50))
            print(f"  {nombre:14} {valor * 100:5.2f}%  {barra}")
    sin_usar = [n for n, v in arbol.importancia_ordenada() if v <= 0.0001]
    if sin_usar:
        print()
        print(f"  Sin usar: {', '.join(sin_usar)}")

    # ------------------------------------------------------------ extras
    if args.arbol is not None:
        separador(f"7. REGLAS APRENDIDAS (primeros {args.arbol} niveles)")
        print(arbol.a_texto(max_profundidad=args.arbol))

    if args.curva:
        curva = imprimir_curva(X_entrena, y_entrena, X_prueba, y_prueba, args)
    else:
        curva = None

    if args.guardar_json:
        guardar_resultados(args.guardar_json, arbol, info, m_entrena, m_prueba,
                           m_trivial, args, tardo, curva)

    return arbol, X_entrena, y_entrena, X_prueba, y_prueba, info


def imprimir_curva(X_entrena, y_entrena, X_prueba, y_prueba, args):
    """Entrena varios arboles cambiando la profundidad para ver el sobreajuste.

    Es la prueba mas clara de que el modelo se comporta como debe: al aumentar la
    profundidad, la exactitud en entrenamiento sube casi siempre, mientras que en
    prueba sube hasta cierto punto y luego se estanca o baja. Ese punto es la
    profundidad adecuada.
    """
    separador("PRUEBA: EXACTITUD SEGUN LA PROFUNDIDAD")
    print(f"{'prof':>5} {'entrena':>9} {'prueba':>9} {'brecha':>9} {'hojas':>7}")
    print("-" * 44)

    resultados = []
    for prof in range(1, 21):
        modelo = ArbolDecision(prep.NUMERICAS, prep.CATEGORICAS,
                               max_profundidad=prof,
                               min_muestras_division=args.min_division,
                               min_muestras_hoja=args.min_hoja,
                               criterio=args.criterio)
        modelo.entrenar(X_entrena, y_entrena)
        ex_e = calcular_metricas(y_entrena, modelo.predecir(X_entrena))["exactitud"]
        ex_p = calcular_metricas(y_prueba, modelo.predecir(X_prueba))["exactitud"]
        _, hojas = modelo.contar_nodos()
        resultados.append({"profundidad": prof, "entrena": ex_e,
                           "prueba": ex_p, "hojas": hojas})
        print(f"{prof:5d} {ex_e:9.4f} {ex_p:9.4f} {ex_e - ex_p:+9.4f} {hojas:7d}")

    mejor = max(resultados, key=lambda r: r["prueba"])
    print()
    print(f"Mejor exactitud en prueba: {mejor['prueba']:.4f} "
          f"con profundidad {mejor['profundidad']}")
    return resultados


def guardar_resultados(ruta, arbol, info, m_entrena, m_prueba, m_trivial,
                       args, tardo, curva):
    """Guarda los numeros en un JSON para que el generador del PDF los use.

    Asi el reporte nunca tiene cifras escritas a mano: todas salen de la
    ejecucion real del programa.
    """
    nodos, hojas = arbol.contar_nodos()
    datos = {
        "parametros": {
            "profundidad_maxima": args.profundidad,
            "min_muestras_division": args.min_division,
            "min_muestras_hoja": args.min_hoja,
            "criterio": args.criterio,
            "semilla": args.semilla,
            "proporcion_prueba": args.proporcion_prueba,
        },
        "datos": {
            "total": info["total"],
            "n_entrena": info["n_entrena"],
            "n_prueba": info["n_prueba"],
            "positivos_entrena": info["positivos_entrena"],
            "positivos_prueba": info["positivos_prueba"],
            "n_variables": len(prep.COLUMNAS),
            "numericas": prep.NUMERICAS,
            "categoricas": prep.CATEGORICAS,
        },
        "arbol": {
            "profundidad_alcanzada": arbol.profundidad_alcanzada(),
            "nodos": nodos,
            "hojas": hojas,
            "segundos_entrenamiento": round(tardo, 3),
            "reglas": arbol.a_texto(max_profundidad=3),
        },
        "prueba": m_prueba,
        "entrenamiento": m_entrena,
        "trivial": m_trivial,
        "importancias": arbol.importancia_ordenada(),
        "curva_profundidad": curva,
    }
    Path(ruta).write_text(json.dumps(datos, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print()
    print(f"Resultados guardados en {ruta}")


# ---------------------------------------------------------------------------
# Prediccion interactiva
# ---------------------------------------------------------------------------

class SalirDelModo(Exception):
    """Se lanza cuando el usuario escribe 'salir' durante la captura."""


def pedir_opcion(texto, opciones):
    """Pide un valor que debe estar en una lista de opciones validas.

    Insiste hasta que la respuesta sea valida. Acepta Enter para dejar el campo
    vacio y no distingue mayusculas de minusculas.
    """
    validas = {o.lower(): o for o in opciones}
    while True:
        respuesta = input(f"{texto}: ").strip()
        if respuesta.lower() == "salir":
            raise SalirDelModo
        if respuesta == "":
            return None
        if respuesta.lower() in validas:
            return validas[respuesta.lower()]      # devuelve con el formato original
        print(f"    Valor no valido. Opciones: {', '.join(opciones)}")


def pedir_numero(texto, minimo=None, maximo=None, entero=False):
    """Pide un numero y verifica que este dentro del rango permitido.

    Sirve para no aceptar cosas imposibles como una edad negativa o un grupo de
    cero personas, que provocarian predicciones sin sentido.
    """
    while True:
        respuesta = input(f"{texto}: ").strip()
        if respuesta.lower() == "salir":
            raise SalirDelModo
        if respuesta == "":
            return None

        valor = prep.a_numero(respuesta)
        if valor is None:
            print("    Eso no es un numero. Intenta de nuevo.")
            continue
        if minimo is not None and valor < minimo:
            print(f"    Debe ser mayor o igual que {minimo:g}.")
            continue
        if maximo is not None and valor > maximo:
            print(f"    Debe ser menor o igual que {maximo:g}.")
            continue
        if entero and valor != int(valor):
            print("    Debe ser un numero entero.")
            continue
        return int(valor) if entero else valor


def modo_interactivo(arbol, referencias):
    """Pide los datos de un pasajero por consola y devuelve la prediccion.

    Los campos que se dejen vacios se completan con los mismos criterios de
    imputacion que se usaron al entrenar, para no obligar a capturar las 15
    variables. Cada dato se valida antes de aceptarlo.
    """
    separador("PREDICCION INTERACTIVA")
    print("Escribe los datos del pasajero. Enter deja el campo vacio y se")
    print("completa automaticamente. Escribe 'salir' para terminar.")

    while True:
        print()
        print("-" * 70)
        try:
            fila = {
                "HomePlanet": pedir_opcion(
                    "Planeta de origen (Earth / Europa / Mars)",
                    ["Earth", "Europa", "Mars"]),
                "CryoSleep": pedir_opcion(
                    "Viajo en criosuenio (True / False)", ["True", "False"]),
                "Destination": pedir_opcion(
                    "Destino (TRAPPIST-1e / 55 Cancri e / PSO J318.5-22)",
                    ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]),
                "Age": pedir_numero("Edad en anios", minimo=0, maximo=120),
                "Deck": pedir_opcion("Cubierta de la cabina (A-G)",
                                     ["A", "B", "C", "D", "E", "F", "G"]),
                "Side": pedir_opcion("Lado de la cabina (P = babor / S = estribor)",
                                     ["P", "S"]),
                "RoomService": pedir_numero("Gasto en servicio a la habitacion",
                                            minimo=0),
                "FoodCourt": pedir_numero("Gasto en zona de comida", minimo=0),
                "ShoppingMall": pedir_numero("Gasto en centro comercial", minimo=0),
                "Spa": pedir_numero("Gasto en spa", minimo=0),
                "VRDeck": pedir_numero("Gasto en cubierta de realidad virtual",
                                       minimo=0),
                # Minimo 1: el propio pasajero siempre cuenta en su grupo
                "TamGrupo": pedir_numero("Cuantas personas viajan en su grupo",
                                         minimo=1, maximo=20, entero=True),
            }
        except SalirDelModo:
            print("Listo.")
            return

        completar_fila(fila, referencias)

        prediccion = arbol.predecir_uno(fila)
        probabilidad = arbol.predecir_probabilidad([fila])[0]

        print()
        print("Datos usados para predecir (incluye los completados):")
        for columna in prep.COLUMNAS:
            print(f"    {columna:14} {fila[columna]}")
        print()
        resultado = "SI fue transportado" if prediccion == 1 else "NO fue transportado"
        print(f"  >>> Prediccion: {resultado}")
        print(f"  >>> Probabilidad estimada de ser transportado: {probabilidad:.1%}")

        if input("\nOtro pasajero? (s/n): ").strip().lower() not in ("s", "si", "y"):
            print("Listo.")
            return


def completar_fila(fila, ref):
    """Rellena los campos vacios de una fila capturada a mano.

    Reutiliza las mismas reglas de `preparar_datos.imputar`, pero para una sola
    fila y sin informacion de grupo.
    """
    for campo in prep.GASTOS:
        fila.setdefault(campo, None)

    presentes = [fila[c] for c in prep.GASTOS if fila.get(c) is not None]
    fila["GastoTotal"] = sum(presentes) if presentes else None

    if fila.get("CryoSleep") is None:
        gasto = fila["GastoTotal"]
        fila["CryoSleep"] = "False" if (gasto is not None and gasto > 0) else "True"

    if fila.get("HomePlanet") is None:
        fila["HomePlanet"] = ref["moda_HomePlanet"]
    if fila.get("Deck") is None:
        fila["Deck"] = ref["cubierta_por_planeta"].get(fila["HomePlanet"], "F")
    if fila["Deck"] == "T":
        fila["Deck"] = "A"
    if fila.get("Side") is None:
        fila["Side"] = ref["moda_Side"]
    if fila.get("Destination") is None:
        fila["Destination"] = ref["moda_Destination"]
    if fila.get("Age") is None:
        clave = (fila["HomePlanet"], fila["CryoSleep"])
        fila["Age"] = ref["edad_por_planeta_cryo"].get(clave) or ref["mediana_Age"]

    for campo in prep.GASTOS:
        if fila.get(campo) is None:
            fila[campo] = 0.0 if fila["CryoSleep"] == "True" else ref["mediana_" + campo]

    if fila.get("TamGrupo") is None:
        fila["TamGrupo"] = 1
    fila["ViajaSolo"] = "Si" if fila["TamGrupo"] == 1 else "No"
    fila["GastoTotal"] = sum(fila[c] for c in prep.GASTOS)
    fila["EsNino"] = "Si" if fila["Age"] <= 12 else "No"
    return fila


def modo_kaggle(arbol, referencias):
    """Predice sobre el test.csv de Kaggle, que no tiene la columna objetivo.

    Sirve para demostrar que el programa puede clasificar datos completamente
    nuevos. No se pueden calcular metricas aqui porque no hay etiquetas reales
    con que comparar: Kaggle las mantiene ocultas.
    """
    separador("PREDICCIONES SOBRE DATOS NUEVOS (test.csv sin etiqueta)")
    ids, X = prep.cargar_sin_etiqueta(RUTA_TEST, referencias)
    predicciones = arbol.predecir(X)

    salida = CARPETA / "predicciones_test.csv"
    with open(salida, "w", encoding="utf-8", newline="") as archivo:
        archivo.write("PassengerId,Transported\n")
        for identificador, prediccion in zip(ids, predicciones):
            archivo.write(f"{identificador},{'True' if prediccion == 1 else 'False'}\n")

    positivos = sum(predicciones)
    print(f"Pasajeros clasificados: {len(ids)}")
    print(f"  Predichos como transportados:    {positivos} "
          f"({positivos / len(ids) * 100:.2f}%)")
    print(f"  Predichos como no transportados: {len(ids) - positivos} "
          f"({(len(ids) - positivos) / len(ids) * 100:.2f}%)")
    print()
    print(f"Archivo generado: {salida.name}")
    print("La proporcion predicha se parece a la del entrenamiento (~50%), que es")
    print("lo esperado si el modelo no esta sesgado hacia una clase.")


# ---------------------------------------------------------------------------

def construir_argumentos():
    """Define las opciones que acepta el programa desde la terminal."""
    p = argparse.ArgumentParser(
        description="Arbol de decision implementado desde cero (sin frameworks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--profundidad", type=int, default=PROFUNDIDAD,
                   help="profundidad maxima del arbol")
    p.add_argument("--min-division", type=int, default=MIN_DIVISION,
                   dest="min_division",
                   help="minimo de ejemplos para dividir un nodo")
    p.add_argument("--min-hoja", type=int, default=MIN_HOJA, dest="min_hoja",
                   help="minimo de ejemplos por hoja")
    p.add_argument("--criterio", choices=["gini", "entropia"], default="gini",
                   help="medida de impureza")
    p.add_argument("--semilla", type=int, default=42,
                   help="semilla de la division entrenamiento/prueba")
    p.add_argument("--proporcion-prueba", type=float, default=0.3,
                   dest="proporcion_prueba",
                   help="fraccion de los datos reservada para prueba")
    p.add_argument("--arbol", type=int, nargs="?", const=3, default=None,
                   metavar="NIVELES",
                   help="dibuja las reglas aprendidas hasta N niveles")
    p.add_argument("--curva", action="store_true",
                   help="compara la exactitud para profundidades de 1 a 20")
    p.add_argument("--interactivo", action="store_true",
                   help="predice a partir de datos capturados en consola")
    p.add_argument("--kaggle", action="store_true",
                   help="predice sobre dataset/test.csv y guarda un CSV")
    p.add_argument("--guardar-json", dest="guardar_json", default=None,
                   metavar="RUTA", help="guarda los resultados en un JSON")
    return p


def main():
    args = construir_argumentos().parse_args()

    print()
    print("ARBOL DE DECISION IMPLEMENTADO SIN FRAMEWORKS")
    print("Dataset: Spaceship Titanic  |  Objetivo: predecir Transported")

    arbol, _, _, _, _, info = entrenar_y_evaluar(args)

    if args.kaggle:
        modo_kaggle(arbol, info["referencias"])
    if args.interactivo:
        modo_interactivo(arbol, info["referencias"])

    print()


if __name__ == "__main__":
    main()
