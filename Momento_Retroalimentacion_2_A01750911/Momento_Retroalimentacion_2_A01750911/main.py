"""
Programa principal: entrena un Random Forest con scikit-learn y lo evalua.

Se ejecuta desde la terminal, sin notebook ni IDE:

    python main.py                        entrena y evalua con los parametros elegidos
    python main.py --buscar               repite la busqueda de hiperparametros
    python main.py --importancias         calcula la importancia por permutacion
    python main.py --curva-aprendizaje    exactitud segun cuantos datos se usan
    python main.py --interactivo          captura un pasajero y predice
    python main.py --kaggle               predice sobre test.csv y guarda un CSV
    python main.py --guardar-json r.json  guarda los resultados para el reporte
    python main.py --todo                 corre todo lo anterior de una vez
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import evaluacion as ev
import preparar_datos as prep
from modelo import (MEJORES, PLIEGUES, buscar_hiperparametros,
                    construir_modelo, resumen_busqueda)

CARPETA = Path(__file__).parent
RUTA_TRAIN = CARPETA / "dataset" / "train.csv"
RUTA_TEST = CARPETA / "dataset" / "test.csv"

PROPORCION_PRUEBA = 0.3


def separador(titulo):
    print()
    print("=" * 72)
    print(titulo)
    print("=" * 72)


# ---------------------------------------------------------------------------

def ejecutar(args):
    resultados = {}

    # ------------------------------------------------------------------ datos
    separador("1. DATOS")
    X, y = prep.cargar_entrenamiento(RUTA_TRAIN)

    # La particion se hace ANTES de cualquier transformacion. El preprocesamiento
    # vive dentro del Pipeline, asi que se ajusta solo con lo que le toca.
    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, stratify=y,
        random_state=prep.SEMILLA)

    print(f"Archivo: {RUTA_TRAIN.name}  ({len(X)} pasajeros con etiqueta)")
    print()
    print(f"  Entrenamiento: {len(X_ent):5d} filas   "
          f"{y_ent.sum():4d} transportados ({y_ent.mean() * 100:.2f}%)")
    print(f"  Prueba:        {len(X_pru):5d} filas   "
          f"{y_pru.sum():4d} transportados ({y_pru.mean() * 100:.2f}%)")
    print()
    print(f"Particion {int((1 - PROPORCION_PRUEBA) * 100)}/"
          f"{int(PROPORCION_PRUEBA * 100)} con train_test_split, estratificada "
          f"(stratify=y) y semilla {prep.SEMILLA}.")
    print("El conjunto de prueba se mide una sola vez, al final.")

    resultados["datos"] = {
        "total": int(len(X)),
        "n_entrena": int(len(X_ent)),
        "n_prueba": int(len(X_pru)),
        "positivos_entrena": int(y_ent.sum()),
        "positivos_prueba": int(y_pru.sum()),
        "proporcion_prueba": PROPORCION_PRUEBA,
        "semilla": prep.SEMILLA,
    }

    # --------------------------------------------------------- hiperparametros
    if args.buscar:
        separador("2. BUSQUEDA DE HIPERPARAMETROS (GridSearchCV)")
        print(f"Validacion cruzada estratificada de {PLIEGUES} pliegues.")
        print("Solo se usa el conjunto de entrenamiento: la prueba no participa.")
        print()
        inicio = time.perf_counter()
        busqueda = buscar_hiperparametros(X_ent, y_ent, verbose=1)
        print(f"\nTardo {time.perf_counter() - inicio:.1f} segundos")
        print()
        print("Mejor combinacion:")
        for clave, valor in busqueda.best_params_.items():
            print(f"  {clave.replace('clasificador__', ''):20} {valor}")
        print(f"  exactitud en validacion cruzada: {busqueda.best_score_:.4f}")
        print()
        print(resumen_busqueda(busqueda, top=10).to_string(index=False))
        parametros = busqueda.best_params_
        resultados["busqueda"] = {
            "mejores": {k.replace("clasificador__", ""): v
                        for k, v in busqueda.best_params_.items()},
            "mejor_puntaje_cv": float(busqueda.best_score_),
            "combinaciones": int(len(busqueda.cv_results_["params"])),
            "top": resumen_busqueda(busqueda, top=8).to_dict("records"),
        }
    else:
        parametros = MEJORES

    # -------------------------------------------------------------- entrenar
    separador("3. ENTRENAMIENTO DEL MODELO")
    modelo = construir_modelo(**parametros)

    bosque = modelo.named_steps["clasificador"]
    print("Algoritmo: RandomForestClassifier de scikit-learn")
    print()
    print("Configuracion:")
    for clave in ["n_estimators", "max_depth", "min_samples_leaf",
                  "max_features", "criterion", "bootstrap", "random_state"]:
        print(f"  {clave:20} {getattr(bosque, clave)}")
    print()
    print("Pasos del Pipeline:")
    for nombre, paso in modelo.steps:
        print(f"  {nombre:14} {type(paso).__name__}")

    inicio = time.perf_counter()
    modelo.fit(X_ent, y_ent)
    tardo = time.perf_counter() - inicio
    print()
    print(f"Entrenado en {tardo:.2f} segundos")

    # Cuantas columnas genero el preprocesamiento
    nombres_variables = modelo.named_steps["preprocesar"].get_feature_names_out()
    print(f"Variables despues del preprocesamiento: {len(nombres_variables)}")
    print(f"Arboles en el bosque: {len(bosque.estimators_)}")
    profundidades = [est.get_depth() for est in bosque.estimators_]
    print(f"Profundidad de los arboles: min {min(profundidades)}, "
          f"max {max(profundidades)}, promedio {sum(profundidades) / len(profundidades):.1f}")

    resultados["modelo"] = {
        "algoritmo": "RandomForestClassifier",
        "parametros": {k.replace("clasificador__", ""): v
                       for k, v in parametros.items()},
        "n_variables": int(len(nombres_variables)),
        "n_arboles": int(len(bosque.estimators_)),
        "profundidad_promedio": float(sum(profundidades) / len(profundidades)),
        "segundos_entrenamiento": round(tardo, 3),
        "pasos_pipeline": [type(p).__name__ for _, p in modelo.steps],
    }

    # ------------------------------------------------------ validacion cruzada
    separador("4. VALIDACION CRUZADA (sobre el entrenamiento)")
    cv = ev.validacion_cruzada(modelo, X_ent, y_ent)
    print(f"Exactitud en cada uno de los {PLIEGUES} pliegues:")
    for i, p in enumerate(cv["puntajes"], 1):
        print(f"  pliegue {i}: {p:.4f}")
    print(f"\nPromedio: {cv['media']:.4f}   desviacion estandar: {cv['desviacion']:.4f}")
    print()
    print("La desviacion baja indica que el resultado no depende de como se")
    print("partieron los datos, o sea que el modelo es estable.")
    resultados["validacion_cruzada"] = cv

    # ------------------------------------------------------------- evaluacion
    separador("5. MATRIZ DE CONFUSION (conjunto de prueba)")
    pred_pru = modelo.predict(X_pru)
    prob_pru = modelo.predict_proba(X_pru)[:, 1]
    m_pru = ev.calcular_metricas(y_pru, pred_pru, prob_pru)
    print(ev.texto_matriz_confusion(m_pru["matriz"]))

    separador("6. METRICAS")
    pred_ent = modelo.predict(X_ent)
    prob_ent = modelo.predict_proba(X_ent)[:, 1]
    m_ent = ev.calcular_metricas(y_ent, pred_ent, prob_ent)

    print("Conjunto de PRUEBA (datos que el modelo nunca vio):")
    print(ev.texto_metricas(m_pru))
    print()
    print("Conjunto de ENTRENAMIENTO (referencia para revisar sobreajuste):")
    print(ev.texto_metricas(m_ent))
    print()
    brecha = m_ent["exactitud"] - m_pru["exactitud"]
    print(f"Diferencia entrenamiento - prueba: {brecha:+.4f}")
    print()
    print("Reporte de clasificacion de sklearn (para verificar los numeros):")
    print(ev.reporte_sklearn(y_pru, pred_pru))

    resultados["prueba"] = m_pru
    resultados["entrenamiento"] = m_ent
    resultados["curva_roc"] = ev.curva_roc(y_pru, prob_pru)

    # -------------------------------------------- comparacion contra referencia
    separador("7. COMPARACION CONTRA REFERENCIAS")
    clase_mayoritaria = int(y_ent.mean() >= 0.5)
    trivial = accuracy_score(y_pru, [clase_mayoritaria] * len(y_pru))
    print(f"Modelo trivial (siempre predice la clase {clase_mayoritaria}): "
          f"{trivial:.4f}")
    print(f"Random Forest de este programa:                     "
          f"{m_pru['exactitud']:.4f}")
    print(f"Arbol de decision a mano (Parte I del modulo):      0.7748")
    print()
    print(f"El bosque supera al modelo trivial por "
          f"{(m_pru['exactitud'] - trivial) * 100:.2f} puntos porcentuales,")
    print(f"y al arbol programado a mano por "
          f"{(m_pru['exactitud'] - 0.7748) * 100:.2f} puntos.")
    resultados["trivial"] = {"exactitud": float(trivial),
                             "clase": clase_mayoritaria}
    resultados["arbol_parte1"] = 0.7748

    # ------------------------------------------------------------- opcionales
    if args.importancias or args.todo:
        separador("8. IMPORTANCIA DE VARIABLES (por permutacion)")
        print("Se revuelve cada columna y se mide cuanto empeora la exactitud.")
        print("Calculado sobre el conjunto de prueba, con 10 repeticiones.")
        print()
        imps = ev.importancia_por_permutacion(modelo, X_pru, y_pru)
        for fila in imps:
            if fila["importancia"] > 0.0005:
                barra = "#" * int(round(fila["importancia"] * 300))
                print(f"  {fila['variable']:14} {fila['importancia'] * 100:5.2f}%  {barra}")
        resultados["importancias"] = imps

    if args.curva_aprendizaje or args.todo:
        separador("9. CURVA DE APRENDIZAJE")
        curva = ev.curva_aprendizaje(modelo, X_ent, y_ent)
        print(f"{'muestras':>10} {'entrena':>10} {'validacion':>12}")
        print("-" * 34)
        for n, e, v in zip(curva["tamanios"], curva["entrenamiento"],
                           curva["validacion"]):
            print(f"{int(n):10d} {e:10.4f} {v:12.4f}")
        resultados["curva_aprendizaje"] = curva

    if args.guardar_json:
        Path(args.guardar_json).write_text(
            json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print(f"Resultados guardados en {args.guardar_json}")

    return modelo


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------

OPCIONES = {
    "HomePlanet": ["Earth", "Europa", "Mars"],
    "CryoSleep": ["True", "False"],
    "Destination": ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],
    "Deck": ["A", "B", "C", "D", "E", "F", "G"],
    "Side": ["P", "S"],
}


class Salir(Exception):
    """Se lanza cuando el usuario escribe 'salir'."""


def pedir_opcion(texto, opciones):
    """Pide un valor de una lista cerrada. Insiste hasta que sea valido."""
    validas = {o.lower(): o for o in opciones}
    while True:
        respuesta = input(f"{texto}: ").strip()
        if respuesta.lower() == "salir":
            raise Salir
        if respuesta == "":
            return None                       # el Pipeline lo imputa
        if respuesta.lower() in validas:
            return validas[respuesta.lower()]
        print(f"    No valido. Opciones: {', '.join(opciones)}")


def pedir_numero(texto, minimo=None, maximo=None):
    """Pide un numero y verifica el rango."""
    while True:
        respuesta = input(f"{texto}: ").strip()
        if respuesta.lower() == "salir":
            raise Salir
        if respuesta == "":
            return None
        try:
            valor = float(respuesta)
        except ValueError:
            print("    Eso no es un numero.")
            continue
        if minimo is not None and valor < minimo:
            print(f"    Debe ser mayor o igual que {minimo:g}.")
            continue
        if maximo is not None and valor > maximo:
            print(f"    Debe ser menor o igual que {maximo:g}.")
            continue
        return valor


def modo_interactivo(modelo):
    """Captura los datos de un pasajero y predice.

    Aqui se ve una ventaja practica de haber metido todo en un Pipeline: se le
    puede pasar una fila cruda, con nulos incluidos, y el propio Pipeline la
    imputa y la codifica antes de clasificarla. No hay que repetir a mano el
    preprocesamiento del entrenamiento.
    """
    separador("PREDICCION INTERACTIVA")
    print("Escribe los datos del pasajero. Enter deja el campo vacio y el")
    print("Pipeline lo imputa. Escribe 'salir' para terminar.")

    while True:
        print()
        print("-" * 72)
        try:
            fila = {
                "PassengerId": "9999_01",
                "HomePlanet": pedir_opcion("Planeta (Earth / Europa / Mars)",
                                           OPCIONES["HomePlanet"]),
                "CryoSleep": pedir_opcion("Criosuenio (True / False)",
                                          OPCIONES["CryoSleep"]),
                "Destination": pedir_opcion(
                    "Destino (TRAPPIST-1e / 55 Cancri e / PSO J318.5-22)",
                    OPCIONES["Destination"]),
                "Age": pedir_numero("Edad", minimo=0, maximo=120),
                "VIP": None,
                "RoomService": pedir_numero("Gasto en servicio a la habitacion",
                                            minimo=0),
                "FoodCourt": pedir_numero("Gasto en zona de comida", minimo=0),
                "ShoppingMall": pedir_numero("Gasto en centro comercial", minimo=0),
                "Spa": pedir_numero("Gasto en spa", minimo=0),
                "VRDeck": pedir_numero("Gasto en realidad virtual", minimo=0),
                "Name": None,
            }
            cubierta = pedir_opcion("Cubierta (A-G)", OPCIONES["Deck"])
            lado = pedir_opcion("Lado (P = babor / S = estribor)", OPCIONES["Side"])
        except Salir:
            print("Listo.")
            return

        # Cabin se reconstruye porque el transformador espera el formato original
        fila["Cabin"] = (f"{cubierta or 'F'}/0/{lado or 'S'}"
                         if (cubierta or lado) else None)

        entrada = pd.DataFrame([fila])
        prediccion = int(modelo.predict(entrada)[0])
        probabilidad = float(modelo.predict_proba(entrada)[0][1])

        print()
        resultado = "SI fue transportado" if prediccion == 1 else "NO fue transportado"
        print(f"  >>> Prediccion: {resultado}")
        print(f"  >>> Probabilidad estimada de ser transportado: {probabilidad:.1%}")
        print(f"      (votaron a favor {probabilidad * 100:.0f} de cada 100 arboles)")

        if input("\nOtro pasajero? (s/n): ").strip().lower() not in ("s", "si", "y"):
            print("Listo.")
            return


def modo_kaggle(modelo):
    """Predice sobre el test.csv de Kaggle, que no tiene etiquetas.

    Sirve para demostrar que el programa clasifica datos completamente nuevos. No
    se pueden calcular metricas porque no hay respuestas con que comparar: Kaggle
    las mantiene ocultas.
    """
    separador("PREDICCIONES SOBRE DATOS NUEVOS (test.csv sin etiqueta)")
    ids, X_nuevo = prep.cargar_sin_etiqueta(RUTA_TEST)
    predicciones = modelo.predict(X_nuevo)
    probabilidades = modelo.predict_proba(X_nuevo)[:, 1]

    salida = CARPETA / "predicciones_test.csv"
    pd.DataFrame({
        "PassengerId": ids,
        "Transported": [bool(p) for p in predicciones],
        "Probabilidad": probabilidades.round(4),
    }).to_csv(salida, index=False)

    positivos = int(predicciones.sum())
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
    p = argparse.ArgumentParser(
        description="Random Forest con scikit-learn sobre Spaceship Titanic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--buscar", action="store_true",
                   help="repite la busqueda de hiperparametros con GridSearchCV")
    p.add_argument("--importancias", action="store_true",
                   help="calcula la importancia de variables por permutacion")
    p.add_argument("--curva-aprendizaje", action="store_true",
                   dest="curva_aprendizaje",
                   help="exactitud segun el tamanio del entrenamiento")
    p.add_argument("--interactivo", action="store_true",
                   help="predice a partir de datos capturados en consola")
    p.add_argument("--kaggle", action="store_true",
                   help="predice sobre dataset/test.csv y guarda un CSV")
    p.add_argument("--todo", action="store_true",
                   help="incluye importancias y curva de aprendizaje")
    p.add_argument("--guardar-json", dest="guardar_json", default=None,
                   metavar="RUTA", help="guarda los resultados en un JSON")
    return p


def main():
    args = construir_argumentos().parse_args()

    print()
    print("RANDOM FOREST CON SCIKIT-LEARN")
    print("Dataset: Spaceship Titanic  |  Objetivo: predecir Transported")

    modelo = ejecutar(args)

    if args.kaggle:
        modo_kaggle(modelo)
    if args.interactivo:
        modo_interactivo(modelo)

    print()


if __name__ == "__main__":
    main()
