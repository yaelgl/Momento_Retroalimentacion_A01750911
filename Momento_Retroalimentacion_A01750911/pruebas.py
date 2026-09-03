"""
Pruebas de correctitud del arbol de decision.

La idea de este archivo es demostrar que la implementacion hace lo que dice
hacer. Medir 77% de exactitud en el dataset real no prueba que el algoritmo este
bien programado: un algoritmo con errores tambien puede dar un numero que parece
razonable.

Entonces aqui se prueba de otra forma: con casos donde la respuesta correcta se
conoce de antemano, ya sea porque se puede calcular a mano o porque los datos se
generaron a partir de una regla conocida. Si el arbol recupera esa regla, la
implementacion es correcta.

    python pruebas.py
"""

import argparse
import json
import random
from pathlib import Path

from arbol_decision import ArbolDecision, entropia, gini
from metricas import calcular_metricas, matriz_confusion
import preparar_datos as prep

# Contadores globales del reporte
PASARON = 0
FALLARON = 0


def verificar(descripcion, condicion, detalle=""):
    """Registra el resultado de una comprobacion y lo imprime."""
    global PASARON, FALLARON
    if condicion:
        PASARON += 1
        print(f"  [ok]    {descripcion}")
    else:
        FALLARON += 1
        print(f"  [FALLO] {descripcion}")
    if detalle:
        print(f"          {detalle}")


def casi_igual(a, b, tolerancia=1e-9):
    """Compara dos numeros con tolerancia, para evitar problemas de redondeo."""
    return abs(a - b) < tolerancia


def titulo(texto):
    print()
    print("-" * 70)
    print(texto)
    print("-" * 70)


# ===========================================================================
# 1. Funciones de impureza
# ===========================================================================

def probar_impureza():
    """El Gini y la entropia se comparan contra valores calculados a mano."""
    titulo("1. FUNCIONES DE IMPUREZA (contra valores calculados a mano)")

    # Mitad y mitad: 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = 0.5, el maximo posible
    g = gini({0: 50, 1: 50}, 100)
    verificar("Gini de 50/50 es 0.5 (impureza maxima)", casi_igual(g, 0.5),
              f"obtenido: {g}")

    # Una sola clase: 1 - 1^2 = 0
    g = gini({0: 100}, 100)
    verificar("Gini de un nodo puro es 0", casi_igual(g, 0.0), f"obtenido: {g}")

    # 75/25: 1 - (0.75^2 + 0.25^2) = 1 - (0.5625 + 0.0625) = 0.375
    g = gini({0: 75, 1: 25}, 100)
    verificar("Gini de 75/25 es 0.375", casi_igual(g, 0.375), f"obtenido: {g}")

    # Entropia de 50/50 = -(0.5*log2(0.5))*2 = 1 bit
    e = entropia({0: 50, 1: 50}, 100)
    verificar("Entropia de 50/50 es 1.0 bit", casi_igual(e, 1.0), f"obtenido: {e}")

    e = entropia({0: 100}, 100)
    verificar("Entropia de un nodo puro es 0", casi_igual(e, 0.0), f"obtenido: {e}")

    # Un nodo mas puro debe tener impureza menor que uno menos puro
    verificar("Gini 90/10 es menor que Gini 60/40",
              gini({0: 90, 1: 10}, 100) < gini({0: 60, 1: 40}, 100))

    verificar("Gini de un nodo vacio es 0 (no truena)",
              casi_igual(gini({}, 0), 0.0))


# ===========================================================================
# 2. Matriz de confusion y metricas
# ===========================================================================

def probar_metricas():
    """Se usa un ejemplo chico donde cada metrica se puede verificar a mano."""
    titulo("2. MATRIZ DE CONFUSION Y METRICAS (ejemplo verificable a mano)")

    # 10 casos armados a proposito:
    #   reales:    1 1 1 1 0 0 0 0 0 0    (4 positivos, 6 negativos)
    #   predichos: 1 1 1 0 0 0 0 0 1 1
    # Comparando uno por uno:
    #   VP = 3 (los tres primeros)
    #   FN = 1 (el cuarto: era 1 y dijo 0)
    #   VN = 4 (posiciones 5 a 8)
    #   FP = 2 (las dos ultimas: eran 0 y dijo 1)
    y_real = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]

    mc = matriz_confusion(y_real, y_pred)
    verificar("VP = 3", mc["VP"] == 3, f"obtenido: {mc['VP']}")
    verificar("FN = 1", mc["FN"] == 1, f"obtenido: {mc['FN']}")
    verificar("VN = 4", mc["VN"] == 4, f"obtenido: {mc['VN']}")
    verificar("FP = 2", mc["FP"] == 2, f"obtenido: {mc['FP']}")
    verificar("Los cuatro conteos suman el total de casos",
              mc["VP"] + mc["VN"] + mc["FP"] + mc["FN"] == len(y_real))

    m = calcular_metricas(y_real, y_pred)
    # exactitud = (VP + VN) / total = (3 + 4) / 10 = 0.7
    verificar("Exactitud = 7/10 = 0.70", casi_igual(m["exactitud"], 0.7),
              f"obtenido: {m['exactitud']}")
    # precision = VP / (VP + FP) = 3 / 5 = 0.6
    verificar("Precision = 3/5 = 0.60", casi_igual(m["precision"], 0.6),
              f"obtenido: {m['precision']}")
    # sensibilidad = VP / (VP + FN) = 3 / 4 = 0.75
    verificar("Sensibilidad = 3/4 = 0.75", casi_igual(m["sensibilidad"], 0.75),
              f"obtenido: {m['sensibilidad']}")
    # especificidad = VN / (VN + FP) = 4 / 6 = 0.6667
    verificar("Especificidad = 4/6 = 0.6667",
              casi_igual(m["especificidad"], 4 / 6), f"obtenido: {m['especificidad']}")
    # F1 = 2 * 0.6 * 0.75 / (0.6 + 0.75) = 0.9 / 1.35 = 0.6667
    verificar("F1 = 0.9/1.35 = 0.6667", casi_igual(m["f1"], 0.9 / 1.35),
              f"obtenido: {m['f1']}")

    # Prediccion perfecta: todo debe valer 1
    m = calcular_metricas([1, 0, 1, 0], [1, 0, 1, 0])
    verificar("Con prediccion perfecta todas las metricas valen 1",
              all(casi_igual(m[k], 1.0) for k in
                  ["exactitud", "precision", "sensibilidad", "especificidad", "f1"]))

    # Caso limite: el modelo nunca predice la clase positiva. La precision seria
    # 0/0 y debe devolver 0 en lugar de tronar con ZeroDivisionError.
    m = calcular_metricas([1, 1, 0, 0], [0, 0, 0, 0])
    verificar("Si el modelo nunca predice positivo, la precision es 0 y no truena",
              casi_igual(m["precision"], 0.0) and casi_igual(m["exactitud"], 0.5))


# ===========================================================================
# 3. El arbol recupera reglas conocidas
# ===========================================================================

def probar_regla_simple():
    """Datos generados con la regla y = 1 si x > 5. El arbol debe encontrarla."""
    titulo("3. APRENDIZAJE DE UNA REGLA CONOCIDA: y = 1 si x > 5")

    X = [{"x": float(i), "cat": "a"} for i in range(1, 11)]
    y = [1 if i > 5 else 0 for i in range(1, 11)]

    arbol = ArbolDecision(["x"], ["cat"], max_profundidad=3,
                          min_muestras_division=2, min_muestras_hoja=1)
    arbol.entrenar(X, y)

    exactitud = calcular_metricas(y, arbol.predecir(X))["exactitud"]
    verificar("Clasifica perfectamente los datos (exactitud = 1.0)",
              casi_igual(exactitud, 1.0), f"obtenido: {exactitud}")
    verificar("Divide por la variable 'x' y no por la categorica irrelevante",
              arbol.raiz.columna == "x", f"uso: {arbol.raiz.columna}")

    # El umbral debe ser el punto medio entre 5 y 6
    verificar("El umbral es 5.5, el punto medio entre 5 y 6",
              casi_igual(arbol.raiz.corte, 5.5), f"obtenido: {arbol.raiz.corte}")
    verificar("Con una sola pregunta le basta (2 hojas)",
              arbol.contar_nodos()[1] == 2, f"hojas: {arbol.contar_nodos()[1]}")

    # Generaliza a valores que no estaban en el entrenamiento
    verificar("Predice bien un valor nuevo por debajo del umbral (x=2.5 -> 0)",
              arbol.predecir_uno({"x": 2.5, "cat": "a"}) == 0)
    verificar("Predice bien un valor nuevo por encima del umbral (x=8.7 -> 1)",
              arbol.predecir_uno({"x": 8.7, "cat": "a"}) == 1)


def probar_regla_and():
    """Regla con dos condiciones: y = 1 solo si a > 0 Y b > 0."""
    titulo("4. REGLA CON DOS CONDICIONES: y = 1 si (a > 0 Y b > 0)")

    X, y = [], []
    for a in range(-5, 6):
        for b in range(-5, 6):
            X.append({"a": float(a), "b": float(b)})
            y.append(1 if (a > 0 and b > 0) else 0)

    arbol = ArbolDecision(["a", "b"], [], max_profundidad=4,
                          min_muestras_division=2, min_muestras_hoja=1)
    arbol.entrenar(X, y)

    exactitud = calcular_metricas(y, arbol.predecir(X))["exactitud"]
    verificar("Clasifica perfectamente los 121 casos",
              casi_igual(exactitud, 1.0), f"obtenido: {exactitud}")
    verificar("Usa las dos variables (ambas tienen importancia > 0)",
              arbol.importancias["a"] > 0 and arbol.importancias["b"] > 0)
    verificar("Predice 1 para (a=3, b=4)",
              arbol.predecir_uno({"a": 3.0, "b": 4.0}) == 1)
    verificar("Predice 0 para (a=3, b=-4), donde falla una condicion",
              arbol.predecir_uno({"a": 3.0, "b": -4.0}) == 0)


def probar_xor():
    """XOR: y = 1 si a y b son distintos.

    Es una prueba interesante porque ninguna de las dos variables por si sola
    dice nada del resultado: hay que combinarlas. Una regresion logistica no
    puede resolver este caso, un arbol si, porque cada rama puede hacer una
    pregunta diferente segun el camino recorrido.
    """
    titulo("5. XOR: y = 1 si a y b son distintos (requiere combinar variables)")

    X, y = [], []
    for a in (0, 1):
        for b in (0, 1):
            for _ in range(25):                    # 25 copias de cada combinacion
                X.append({"a": float(a), "b": float(b)})
                y.append(1 if a != b else 0)

    arbol = ArbolDecision(["a", "b"], [], max_profundidad=3,
                          min_muestras_division=2, min_muestras_hoja=1)
    arbol.entrenar(X, y)

    exactitud = calcular_metricas(y, arbol.predecir(X))["exactitud"]
    verificar("Resuelve el XOR perfectamente (exactitud = 1.0)",
              casi_igual(exactitud, 1.0), f"obtenido: {exactitud}")
    verificar("Necesita al menos 2 niveles de profundidad",
              arbol.profundidad_alcanzada() >= 2,
              f"profundidad: {arbol.profundidad_alcanzada()}")
    verificar("Predice 1 para (0,1)", arbol.predecir_uno({"a": 0.0, "b": 1.0}) == 1)
    verificar("Predice 1 para (1,0)", arbol.predecir_uno({"a": 1.0, "b": 0.0}) == 1)
    verificar("Predice 0 para (0,0)", arbol.predecir_uno({"a": 0.0, "b": 0.0}) == 0)
    verificar("Predice 0 para (1,1)", arbol.predecir_uno({"a": 1.0, "b": 1.0}) == 0)


def probar_categorica():
    """Regla sobre una variable categorica: y = 1 solo si color es 'rojo'."""
    titulo("6. REGLA CATEGORICA: y = 1 si color == 'rojo'")

    colores = ["rojo", "verde", "azul", "amarillo"]
    X, y = [], []
    for color in colores:
        for _ in range(30):
            X.append({"color": color, "ruido": 1.0})
            y.append(1 if color == "rojo" else 0)

    arbol = ArbolDecision(["ruido"], ["color"], max_profundidad=3,
                          min_muestras_division=2, min_muestras_hoja=1)
    arbol.entrenar(X, y)

    exactitud = calcular_metricas(y, arbol.predecir(X))["exactitud"]
    verificar("Clasifica perfectamente (exactitud = 1.0)",
              casi_igual(exactitud, 1.0), f"obtenido: {exactitud}")
    verificar("La primera pregunta es sobre 'color'",
              arbol.raiz.columna == "color", f"uso: {arbol.raiz.columna}")
    verificar("Y separa justamente la categoria 'rojo'",
              arbol.raiz.corte == "rojo", f"corte: {arbol.raiz.corte}")
    verificar("Con una sola pregunta le basta (2 hojas)",
              arbol.contar_nodos()[1] == 2)

    # Categoria que no aparecio al entrenar: debe caer del lado "cualquier otra"
    verificar("Una categoria nueva ('morado') no truena y se predice como 0",
              arbol.predecir_uno({"color": "morado", "ruido": 1.0}) == 0)


def probar_variable_irrelevante():
    """El arbol debe ignorar variables que son puro ruido aleatorio."""
    titulo("7. VARIABLES IRRELEVANTES: el arbol debe ignorar el ruido")

    azar = random.Random(7)
    X, y = [], []
    for i in range(400):
        senal = azar.choice([0.0, 1.0])
        X.append({"senal": senal,
                  "ruido1": azar.random(),
                  "ruido2": azar.random()})
        y.append(int(senal))

    arbol = ArbolDecision(["senal", "ruido1", "ruido2"], [],
                          max_profundidad=5, min_muestras_division=10,
                          min_muestras_hoja=5)
    arbol.entrenar(X, y)

    exactitud = calcular_metricas(y, arbol.predecir(X))["exactitud"]
    verificar("Clasifica perfectamente usando solo la senal",
              casi_igual(exactitud, 1.0), f"obtenido: {exactitud}")
    verificar("Toda la importancia se la lleva 'senal'",
              casi_igual(arbol.importancias["senal"], 1.0),
              f"senal={arbol.importancias['senal']:.4f} "
              f"ruido1={arbol.importancias['ruido1']:.4f} "
              f"ruido2={arbol.importancias['ruido2']:.4f}")
    verificar("El arbol para en cuanto los nodos quedan puros (2 hojas)",
              arbol.contar_nodos()[1] == 2, f"hojas: {arbol.contar_nodos()[1]}")


# ===========================================================================
# 4. Los limites de crecimiento se respetan
# ===========================================================================

def recolectar_hojas(nodo, hojas=None):
    """Devuelve la lista de nodos hoja del arbol."""
    if hojas is None:
        hojas = []
    if nodo.es_hoja:
        hojas.append(nodo)
    else:
        recolectar_hojas(nodo.izquierda, hojas)
        recolectar_hojas(nodo.derecha, hojas)
    return hojas


def probar_limites():
    """Los hiperparametros de paro deben cumplirse siempre."""
    titulo("8. LOS CRITERIOS DE PARO SE RESPETAN")

    azar = random.Random(1)
    X = [{"x1": azar.random(), "x2": azar.random(), "c": azar.choice("abc")}
         for _ in range(600)]
    y = [azar.choice([0, 1]) for _ in range(600)]

    # Profundidad maxima
    for limite in (1, 3, 5):
        arbol = ArbolDecision(["x1", "x2"], ["c"], max_profundidad=limite,
                              min_muestras_division=2, min_muestras_hoja=1)
        arbol.entrenar(X, y)
        alcanzada = arbol.profundidad_alcanzada()
        verificar(f"Con max_profundidad={limite} el arbol no pasa de ese nivel",
                  alcanzada <= limite, f"profundidad alcanzada: {alcanzada}")

    # Minimo de ejemplos por hoja
    for minimo in (20, 50, 100):
        arbol = ArbolDecision(["x1", "x2"], ["c"], max_profundidad=15,
                              min_muestras_division=2, min_muestras_hoja=minimo)
        arbol.entrenar(X, y)
        hojas = recolectar_hojas(arbol.raiz)
        mas_chica = min(h.n for h in hojas)
        verificar(f"Con min_muestras_hoja={minimo} ninguna hoja queda mas chica",
                  mas_chica >= minimo,
                  f"hoja mas chica: {mas_chica} ({len(hojas)} hojas)")

    # Un nodo puro no se divide: si todas las etiquetas son iguales no hay nada
    # que ganar y el arbol debe quedar en una sola hoja
    arbol = ArbolDecision(["x1"], [], max_profundidad=10,
                          min_muestras_division=2, min_muestras_hoja=1)
    arbol.entrenar([{"x1": float(i)} for i in range(50)], [1] * 50)
    verificar("Si todas las etiquetas son iguales, el arbol es una sola hoja",
              arbol.raiz.es_hoja and arbol.contar_nodos() == (1, 1),
              f"nodos: {arbol.contar_nodos()}")
    verificar("Y ese arbol siempre predice esa clase",
              arbol.predecir_uno({"x1": 999.0}) == 1)

    # La suma de los ejemplos de las hojas debe ser el total de entrenamiento:
    # ningun ejemplo se pierde ni se duplica al bajar por el arbol
    arbol = ArbolDecision(["x1", "x2"], ["c"], max_profundidad=6,
                          min_muestras_division=10, min_muestras_hoja=5)
    arbol.entrenar(X, y)
    suma = sum(h.n for h in recolectar_hojas(arbol.raiz))
    verificar("Los ejemplos de todas las hojas suman el total de entrenamiento",
              suma == len(X), f"suma: {suma}, esperado: {len(X)}")


# ===========================================================================
# 5. Comportamiento general del modelo
# ===========================================================================

def probar_consistencia():
    """Determinismo, rango de probabilidades y coherencia entre metodos."""
    titulo("9. CONSISTENCIA Y DETERMINISMO")

    azar = random.Random(3)
    X = [{"x": azar.random(), "c": azar.choice("ab")} for _ in range(300)]
    y = [1 if fila["x"] > 0.5 else 0 for fila in X]

    a1 = ArbolDecision(["x"], ["c"], max_profundidad=4).entrenar(X, y)
    a2 = ArbolDecision(["x"], ["c"], max_profundidad=4).entrenar(X, y)

    verificar("Entrenar dos veces con los mismos datos da el mismo arbol",
              a1.predecir(X) == a2.predecir(X))
    verificar("Predecir la misma fila dos veces da el mismo resultado",
              a1.predecir_uno(X[0]) == a1.predecir_uno(X[0]))

    probabilidades = a1.predecir_probabilidad(X)
    verificar("Todas las probabilidades caen entre 0 y 1",
              all(0.0 <= p <= 1.0 for p in probabilidades))
    verificar("La clase predicha coincide con la probabilidad (>= 0.5 -> clase 1)",
              all((p >= 0.5) == (c == 1)
                  for p, c in zip(probabilidades, a1.predecir(X))))
    verificar("Hay una prediccion por cada fila de entrada",
              len(a1.predecir(X)) == len(X))

    # El criterio de entropia debe dar resultados parecidos al de Gini: miden lo
    # mismo de formas distintas
    a_gini = ArbolDecision(["x"], ["c"], max_profundidad=4, criterio="gini").entrenar(X, y)
    a_ent = ArbolDecision(["x"], ["c"], max_profundidad=4, criterio="entropia").entrenar(X, y)
    ex_g = calcular_metricas(y, a_gini.predecir(X))["exactitud"]
    ex_e = calcular_metricas(y, a_ent.predecir(X))["exactitud"]
    verificar("Gini y entropia dan exactitudes similares (difieren menos de 0.05)",
              abs(ex_g - ex_e) < 0.05, f"gini={ex_g:.4f}  entropia={ex_e:.4f}")


# ===========================================================================
# 6. Preparacion de los datos reales
# ===========================================================================

def probar_preparacion_datos():
    """La division de los datos reales debe ser estratificada y sin fugas."""
    titulo("10. PREPARACION DEL DATASET REAL")

    from pathlib import Path
    ruta = Path(__file__).parent / "dataset" / "train.csv"
    Xe, ye, Xp, yp, info = prep.cargar_datos(ruta, proporcion_prueba=0.3, semilla=42)

    verificar("Entrenamiento y prueba juntos suman todas las filas del archivo",
              len(Xe) + len(Xp) == info["total"],
              f"{len(Xe)} + {len(Xp)} = {len(Xe) + len(Xp)}, total {info['total']}")

    esperado = int(round(info["total"] * 0.3))
    verificar("El conjunto de prueba tiene cerca del 30% de los datos",
              abs(len(Xp) - esperado) <= 2, f"prueba: {len(Xp)}, esperado ~{esperado}")

    pct_e = sum(ye) / len(ye)
    pct_p = sum(yp) / len(yp)
    verificar("La proporcion de clases es practicamente igual en ambos conjuntos",
              abs(pct_e - pct_p) < 0.01,
              f"entrenamiento {pct_e:.4f}  prueba {pct_p:.4f}")

    faltantes = sum(1 for fila in Xe + Xp for v in fila.values() if v is None)
    verificar("No quedan valores faltantes despues de imputar",
              faltantes == 0, f"faltantes: {faltantes}")

    verificar("Todas las filas tienen las 15 variables esperadas",
              all(len(fila) == len(prep.COLUMNAS) for fila in Xe + Xp))

    verificar("Las etiquetas son solo 0 y 1",
              set(ye) | set(yp) == {0, 1})

    # Reproducibilidad: la misma semilla debe dar exactamente la misma division
    Xe2, ye2, _, _, _ = prep.cargar_datos(ruta, proporcion_prueba=0.3, semilla=42)
    verificar("Con la misma semilla la division es identica (reproducible)",
              ye == ye2 and Xe[0] == Xe2[0])

    # Y una semilla distinta debe dar una division distinta
    _, ye3, _, _, _ = prep.cargar_datos(ruta, proporcion_prueba=0.3, semilla=99)
    verificar("Con otra semilla la division cambia", ye != ye3)


def probar_modelo_real():
    """El modelo entrenado con los datos reales debe superar al azar con claridad."""
    titulo("11. DESEMPENIO EN EL DATASET REAL")

    from pathlib import Path
    ruta = Path(__file__).parent / "dataset" / "train.csv"
    Xe, ye, Xp, yp, _ = prep.cargar_datos(ruta, proporcion_prueba=0.3, semilla=42)

    arbol = ArbolDecision(prep.NUMERICAS, prep.CATEGORICAS, max_profundidad=9,
                          min_muestras_division=80, min_muestras_hoja=40)
    arbol.entrenar(Xe, ye)

    m_p = calcular_metricas(yp, arbol.predecir(Xp))
    m_e = calcular_metricas(ye, arbol.predecir(Xe))

    verificar("La exactitud en prueba supera el 70%",
              m_p["exactitud"] > 0.70, f"obtenido: {m_p['exactitud']:.4f}")
    verificar("Y supera por mucho al clasificador trivial (50.36%)",
              m_p["exactitud"] > 0.60, f"obtenido: {m_p['exactitud']:.4f}")
    verificar("La exactitud en entrenamiento es mayor que en prueba (esperado)",
              m_e["exactitud"] >= m_p["exactitud"],
              f"entrenamiento {m_e['exactitud']:.4f}  prueba {m_p['exactitud']:.4f}")
    verificar("La brecha entre ambos es menor a 0.10 (no hay sobreajuste severo)",
              m_e["exactitud"] - m_p["exactitud"] < 0.10,
              f"brecha: {m_e['exactitud'] - m_p['exactitud']:+.4f}")
    verificar("El modelo funciona parecido en las dos clases",
              abs(m_p["sensibilidad"] - m_p["especificidad"]) < 0.15,
              f"sensibilidad {m_p['sensibilidad']:.4f}  "
              f"especificidad {m_p['especificidad']:.4f}")
    verificar("Predice ambas clases y no se queda con una sola",
              0 < sum(arbol.predecir(Xp)) < len(Xp))


# ===========================================================================

def main():
    analizador = argparse.ArgumentParser(
        description="Pruebas de correctitud del arbol de decision.")
    analizador.add_argument("--guardar-json", dest="guardar_json",
                            nargs="?", const="pruebas.json", default=None,
                            metavar="RUTA",
                            help="guarda el conteo de pruebas en un JSON")
    args = analizador.parse_args()

    print()
    print("=" * 70)
    print("PRUEBAS DE CORRECTITUD DEL ARBOL DE DECISION")
    print("=" * 70)
    print("Cada prueba compara el resultado del programa contra una respuesta")
    print("que se conoce de antemano, calculada a mano o derivada de una regla")
    print("con la que se generaron los datos a proposito.")

    probar_impureza()
    probar_metricas()
    probar_regla_simple()
    probar_regla_and()
    probar_xor()
    probar_categorica()
    probar_variable_irrelevante()
    probar_limites()
    probar_consistencia()
    probar_preparacion_datos()
    probar_modelo_real()

    print()
    print("=" * 70)
    print(f"RESULTADO: {PASARON} pruebas pasaron, {FALLARON} fallaron")
    print("=" * 70)
    if FALLARON == 0:
        print("Todas las pruebas pasaron.")
    else:
        print("Hay pruebas fallando, revisar la implementacion.")

    # El conteo se puede exportar para que el reporte lo cite sin escribirlo a mano
    if args.guardar_json:
        ruta = Path(args.guardar_json)
        ruta.write_text(json.dumps({"pasaron": PASARON, "fallaron": FALLARON},
                                   indent=2), encoding="utf-8")
        print(f"Conteo guardado en {ruta}")

    print()
    return 0 if FALLARON == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
