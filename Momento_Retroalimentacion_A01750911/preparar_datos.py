"""
Carga y preparacion del dataset Spaceship Titanic.

Todo esta hecho con la libreria estandar de Python (csv, random, collections).
No se usa pandas ni numpy a proposito, porque la idea del entregable es que el
trabajo sea manual.

Un detalle importante: como el modelo que vamos a entrenar es un arbol de
decision, NO hace falta escalar las variables numericas ni convertir las
categoricas a one-hot. El arbol parte los datos con umbrales del tipo
"Age <= 27", y esa comparacion no cambia si multiplico o divido toda la columna
por una constante. Las categoricas las maneja directamente preguntando
"HomePlanet == Europa". Eso simplifica bastante el preprocesamiento respecto a
lo que necesitaria una regresion.
"""

import csv
import random
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuracion: nombres de columnas
# ---------------------------------------------------------------------------

# Las 5 columnas de consumo a bordo
GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Variable que queremos predecir
OBJETIVO = "Transported"

# Variables que se le pasan al arbol, separadas por tipo.
# El arbol necesita saber el tipo para elegir como partir: por umbral (<=) si es
# numerica, o por igualdad (==) si es categorica.
NUMERICAS = ["Age", "TamGrupo"] + GASTOS + ["GastoTotal"]
CATEGORICAS = ["HomePlanet", "CryoSleep", "Destination", "Deck", "Side",
               "ViajaSolo", "EsNino"]
COLUMNAS = NUMERICAS + CATEGORICAS


# ---------------------------------------------------------------------------
# Utilidades basicas (reemplazan lo que normalmente haria pandas/numpy)
# ---------------------------------------------------------------------------

def a_numero(texto):
    """Convierte un texto a float. Devuelve None si esta vacio o no es numero."""
    if texto is None or texto == "":
        return None
    try:
        return float(texto)
    except ValueError:
        return None


def mediana(valores):
    """Mediana de una lista de numeros, ignorando los None."""
    limpios = sorted(v for v in valores if v is not None)
    if not limpios:
        return None
    n = len(limpios)
    medio = n // 2
    if n % 2 == 1:
        return limpios[medio]
    return (limpios[medio - 1] + limpios[medio]) / 2


def moda(valores):
    """Valor mas frecuente de una lista, ignorando los None."""
    limpios = [v for v in valores if v is not None]
    if not limpios:
        return None
    return Counter(limpios).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# 1. Lectura del archivo
# ---------------------------------------------------------------------------

def leer_csv(ruta):
    """Lee un CSV y devuelve una lista de diccionarios, uno por fila.

    Los campos vacios quedan como None en lugar de cadena vacia, para poder
    distinguir "falta el dato" de "el dato es cero".
    """
    filas = []
    with open(ruta, newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            filas.append({k: (v if v != "" else None) for k, v in fila.items()})
    return filas


# ---------------------------------------------------------------------------
# 2. Ingenieria de variables
# ---------------------------------------------------------------------------

def derivar_variables(filas):
    """Crea variables nuevas a partir de las columnas originales.

    Se aprovechan dos columnas que por si solas no sirven porque tienen
    demasiados valores distintos:

    - Cabin viene como "B/0/P" -> se parte en Deck (cubierta) y Side (babor o
      estribor). El numero de cabina en medio se ignora porque en el analisis
      previo no mostro relacion con el objetivo.
    - PassengerId viene como "0001_01", donde los primeros 4 digitos son el
      grupo de viaje -> de ahi salen Grupo, TamGrupo y ViajaSolo.

    Ademas se crea GastoTotal (suma de los 5 consumos) y EsNino (menor o igual
    a 12 anios), que en la exploracion resultaron informativas.
    """
    # Primero se parten Cabin y PassengerId fila por fila
    for fila in filas:
        cabina = fila.get("Cabin")
        if cabina:
            partes = cabina.split("/")
            fila["Deck"] = partes[0] if len(partes) > 0 else None
            fila["Side"] = partes[2] if len(partes) > 2 else None
        else:
            fila["Deck"] = None
            fila["Side"] = None

        fila["Grupo"] = fila["PassengerId"].split("_")[0]

        # Los gastos se convierten a numero de una vez
        for col in GASTOS:
            fila[col] = a_numero(fila.get(col))
        fila["Age"] = a_numero(fila.get("Age"))

        # GastoTotal queda en None solo si TODOS los gastos faltan; si alguno
        # existe se suma lo que haya (mismo criterio que min_count=1 de pandas)
        presentes = [fila[c] for c in GASTOS if fila[c] is not None]
        fila["GastoTotal"] = sum(presentes) if presentes else None

    # TamGrupo necesita contar cuantos pasajeros comparten cada grupo, asi que
    # se calcula despues de recorrer todas las filas
    conteo_grupos = Counter(fila["Grupo"] for fila in filas)
    for fila in filas:
        fila["TamGrupo"] = conteo_grupos[fila["Grupo"]]
        fila["ViajaSolo"] = "Si" if fila["TamGrupo"] == 1 else "No"

    return filas


# ---------------------------------------------------------------------------
# 3. Imputacion de valores faltantes
# ---------------------------------------------------------------------------

def calcular_referencias(filas):
    """Calcula las modas y medianas que se usaran para imputar.

    Se separa del paso de imputar por una razon importante: estos estadisticos
    se calculan SOLO con el conjunto de entrenamiento y despues se aplican tal
    cual al de prueba. Si se calcularan con todos los datos juntos, el modelo
    estaria usando informacion del conjunto de prueba para prepararse, y las
    metricas saldrian mejores de lo que realmente son (fuga de informacion).
    """
    ref = {}

    ref["moda_HomePlanet"] = moda([f.get("HomePlanet") for f in filas])
    ref["moda_Destination"] = moda([f.get("Destination") for f in filas])
    ref["moda_Side"] = moda([f.get("Side") for f in filas])
    ref["mediana_Age"] = mediana([f.get("Age") for f in filas])

    # Moda de HomePlanet dentro de cada cubierta: sirve porque el cruce esta muy
    # restringido (por ejemplo, en las cubiertas A/B/C casi no hay pasajeros de
    # la Tierra), asi que la cubierta dice mucho del planeta de origen.
    por_cubierta = {}
    for f in filas:
        if f.get("Deck") and f.get("HomePlanet"):
            por_cubierta.setdefault(f["Deck"], []).append(f["HomePlanet"])
    ref["planeta_por_cubierta"] = {k: moda(v) for k, v in por_cubierta.items()}

    # Y al reves: moda de la cubierta dentro de cada planeta
    por_planeta = {}
    for f in filas:
        if f.get("Deck") and f.get("HomePlanet"):
            por_planeta.setdefault(f["HomePlanet"], []).append(f["Deck"])
    ref["cubierta_por_planeta"] = {k: moda(v) for k, v in por_planeta.items()}

    # Mediana de edad por planeta y estado de criosueno
    edades = {}
    for f in filas:
        if f.get("Age") is not None and f.get("HomePlanet") and f.get("CryoSleep"):
            edades.setdefault((f["HomePlanet"], f["CryoSleep"]), []).append(f["Age"])
    ref["edad_por_planeta_cryo"] = {k: mediana(v) for k, v in edades.items()}

    # Mediana de cada gasto considerando solo a los pasajeros despiertos. A los
    # que estan en criosueno no tiene sentido imputarles consumo.
    for col in GASTOS:
        despiertos = [f[col] for f in filas
                      if f.get("CryoSleep") == "False" and f[col] is not None]
        ref["mediana_" + col] = mediana(despiertos) or 0.0

    return ref


def imputar(filas, ref):
    """Rellena los valores faltantes usando los estadisticos de `ref`.

    El orden va de la informacion mas confiable a la menos confiable: primero
    reglas logicas que son deducciones seguras, luego informacion del grupo de
    viaje, luego relaciones entre variables y al final modas y medianas.
    """
    # Mapas construidos con las propias filas: si un pasajero no tiene planeta
    # pero un companiero de su grupo si, se copia. En el analisis previo se
    # verifico que el planeta de origen nunca varia dentro de un grupo.
    planeta_de_grupo = {}
    cubierta_de_grupo = {}
    for f in filas:
        if f.get("HomePlanet"):
            planeta_de_grupo.setdefault(f["Grupo"], f["HomePlanet"])
        if f.get("Deck"):
            cubierta_de_grupo.setdefault(f["Grupo"], f["Deck"])

    for f in filas:
        # --- CryoSleep: aqui hay una regla logica, no hace falta estimar ------
        # Si el pasajero gasto dinero, es imposible que estuviera dormido.
        # Si no gasto nada, lo mas probable es que si (en el analisis previo,
        # el 85% de los que gastan 0 y tienen el dato estan en criosueno).
        if f.get("CryoSleep") is None:
            gasto = f.get("GastoTotal")
            f["CryoSleep"] = "False" if (gasto is not None and gasto > 0) else "True"

        # --- HomePlanet: grupo -> cubierta -> moda general -------------------
        if f.get("HomePlanet") is None:
            f["HomePlanet"] = (planeta_de_grupo.get(f["Grupo"])
                               or ref["planeta_por_cubierta"].get(f.get("Deck"))
                               or ref["moda_HomePlanet"])

        # --- Deck: grupo -> cubierta tipica del planeta ----------------------
        if f.get("Deck") is None:
            f["Deck"] = (cubierta_de_grupo.get(f["Grupo"])
                         or ref["cubierta_por_planeta"].get(f["HomePlanet"])
                         or "F")

        # La cubierta T tiene solo 5 pasajeros en todo el dataset. Se junta con
        # A porque ambas son cubiertas altas y de pasajeros de Europa; dejarla
        # sola solo agregaria ruido.
        if f["Deck"] == "T":
            f["Deck"] = "A"

        # --- Side y Destination: moda, no hay mejor informacion --------------
        if f.get("Side") is None:
            f["Side"] = ref["moda_Side"]
        if f.get("Destination") is None:
            f["Destination"] = ref["moda_Destination"]

        # --- Age: mediana del grupo planeta x criosueno ----------------------
        if f.get("Age") is None:
            clave = (f["HomePlanet"], f["CryoSleep"])
            f["Age"] = ref["edad_por_planeta_cryo"].get(clave) or ref["mediana_Age"]

        # --- Gastos: 0 si duerme, mediana de despiertos si no ----------------
        for col in GASTOS:
            if f[col] is None:
                if f["CryoSleep"] == "True":
                    f[col] = 0.0
                else:
                    f[col] = ref["mediana_" + col]

        # --- Variables que dependen de lo anterior: recalcular ---------------
        # Si no se recalculan, quedan con el valor viejo o en None.
        f["GastoTotal"] = sum(f[c] for c in GASTOS)
        f["EsNino"] = "Si" if f["Age"] <= 12 else "No"

    return filas


# ---------------------------------------------------------------------------
# 4. Division en entrenamiento y prueba
# ---------------------------------------------------------------------------

def dividir_estratificado(X, y, proporcion_prueba=0.3, semilla=42):
    """Divide los datos en entrenamiento y prueba manteniendo el balance de clases.

    Estratificado quiere decir que si en el total hay 50% de transportados, tanto
    el conjunto de entrenamiento como el de prueba van a tener cerca de 50%. Se
    logra revolviendo y partiendo cada clase por separado.

    La semilla fija hace que la division sea siempre la misma, para que los
    resultados del reporte se puedan reproducir.
    """
    azar = random.Random(semilla)

    # Se agrupan los indices por clase
    indices_por_clase = {}
    for i, etiqueta in enumerate(y):
        indices_por_clase.setdefault(etiqueta, []).append(i)

    idx_entrena, idx_prueba = [], []
    for etiqueta in sorted(indices_por_clase):
        indices = indices_por_clase[etiqueta][:]
        azar.shuffle(indices)
        corte = int(round(len(indices) * proporcion_prueba))
        idx_prueba.extend(indices[:corte])
        idx_entrena.extend(indices[corte:])

    # Se revuelve otra vez para que las clases no queden en bloques
    azar.shuffle(idx_entrena)
    azar.shuffle(idx_prueba)

    X_entrena = [X[i] for i in idx_entrena]
    y_entrena = [y[i] for i in idx_entrena]
    X_prueba = [X[i] for i in idx_prueba]
    y_prueba = [y[i] for i in idx_prueba]
    return X_entrena, y_entrena, X_prueba, y_prueba


# ---------------------------------------------------------------------------
# 5. Funcion principal de este modulo
# ---------------------------------------------------------------------------

def cargar_datos(ruta_csv, proporcion_prueba=0.3, semilla=42):
    """Deja los datos listos para entrenar el arbol.

    Devuelve (X_entrena, y_entrena, X_prueba, y_prueba, info).

    El orden de los pasos importa: primero se divide y despues se imputa, no al
    reves. Asi las medianas y modas salen unicamente del conjunto de
    entrenamiento y el de prueba se mantiene como datos verdaderamente nuevos.
    """
    filas = leer_csv(ruta_csv)
    filas = derivar_variables(filas)

    # La etiqueta viene como el texto "True"/"False" y se pasa a 1/0
    y = [1 if f[OBJETIVO] == "True" else 0 for f in filas]

    # Se divide ANTES de imputar
    filas_entrena, y_entrena, filas_prueba, y_prueba = dividir_estratificado(
        filas, y, proporcion_prueba, semilla)

    # Las referencias se calculan solo con entrenamiento y se aplican a ambos
    ref = calcular_referencias(filas_entrena)
    filas_entrena = imputar(filas_entrena, ref)
    filas_prueba = imputar(filas_prueba, ref)

    # Se conservan unicamente las columnas que entran al modelo
    X_entrena = [{c: f[c] for c in COLUMNAS} for f in filas_entrena]
    X_prueba = [{c: f[c] for c in COLUMNAS} for f in filas_prueba]

    info = {
        "total": len(filas),
        "n_entrena": len(X_entrena),
        "n_prueba": len(X_prueba),
        "positivos_entrena": sum(y_entrena),
        "positivos_prueba": sum(y_prueba),
        "referencias": ref,
    }
    return X_entrena, y_entrena, X_prueba, y_prueba, info


def cargar_sin_etiqueta(ruta_csv, ref):
    """Prepara un CSV que no tiene la columna objetivo (el test.csv de Kaggle).

    Se usan las referencias ya calculadas con el conjunto de entrenamiento.
    Devuelve (lista de PassengerId, lista de diccionarios de variables).
    """
    filas = derivar_variables(leer_csv(ruta_csv))
    filas = imputar(filas, ref)
    ids = [f["PassengerId"] for f in filas]
    X = [{c: f[c] for c in COLUMNAS} for f in filas]
    return ids, X


# Comprobacion rapida si se ejecuta este archivo directamente
if __name__ == "__main__":
    ruta = Path(__file__).parent / "dataset" / "train.csv"
    Xe, ye, Xp, yp, info = cargar_datos(ruta)

    print("Filas leidas del archivo:", info["total"])
    print(f"Entrenamiento: {info['n_entrena']} filas "
          f"({info['positivos_entrena'] / info['n_entrena'] * 100:.2f}% transportados)")
    print(f"Prueba:        {info['n_prueba']} filas "
          f"({info['positivos_prueba'] / info['n_prueba'] * 100:.2f}% transportados)")
    print("Variables por fila:", len(COLUMNAS))

    faltantes = sum(1 for fila in Xe + Xp for v in fila.values() if v is None)
    print("Valores faltantes despues de imputar:", faltantes)
    print()
    print("Ejemplo de una fila preparada:")
    for clave, valor in Xe[0].items():
        print(f"  {clave:14} {valor}")
