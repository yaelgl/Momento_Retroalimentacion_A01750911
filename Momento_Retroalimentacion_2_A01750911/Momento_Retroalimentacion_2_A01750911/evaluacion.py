"""
Evaluacion del modelo: matriz de confusion, metricas y curvas.

Aqui si se usan las funciones de sklearn.metrics, a diferencia de la Parte I del
modulo donde las formulas se programaron a mano.

QUE METRICA SE ELIGIO Y POR QUE
===============================

Metrica principal: EXACTITUD (accuracy).

Es la proporcion de predicciones correctas sobre el total. Se eligio esa y no otra
por tres razones concretas de este problema:

1. Las clases estan balanceadas. En el dataset el 50.36% de los pasajeros fueron
   transportados y el 49.64% no. Ese es el punto clave, porque la critica habitual
   contra la exactitud es que enganie en datos desbalanceados: si el 95% de los
   casos fueran de una clase, predecir siempre esa clase daria 95% de exactitud
   sin haber aprendido nada. Aqui ese problema no existe, y de hecho el modelo
   trivial que siempre predice la clase mas frecuente sirve como piso de
   comparacion (~50%).

2. Los dos tipos de error cuestan lo mismo. No hay ninguna razon en este problema
   para que equivocarse diciendo "si fue transportado" sea peor que equivocarse al
   reves. Cuando los costos son asimetricos (por ejemplo en diagnostico medico,
   donde un falso negativo puede ser grave) conviene priorizar sensibilidad o
   precision; aqui no aplica.

3. Es directamente interpretable y comparable. "El modelo acierta 8 de cada 10"
   se entiende sin explicacion, y permite comparar de frente contra el arbol
   programado a mano en la Parte I del modulo.

Por que NO se eligio F1 como principal: el F1 combina precision y sensibilidad,
que son dos metricas centradas en la clase positiva, y por construccion ignora los
verdaderos negativos. En este problema acertar quien NO fue transportado importa
exactamente igual que acertar quien si, asi que usar F1 como criterio principal
dejaria fuera la mitad del problema. Se reporta de todos modos, pero como
secundaria.

Metricas secundarias que se reportan y para que sirven:

- Precision, sensibilidad y especificidad: sirven para verificar que el modelo no
  este inclinado hacia una clase. Reportar sensibilidad y especificidad juntas es
  lo que permite ver eso; si una fuera mucho mas alta que la otra, la exactitud
  global podria estar escondiendo que el modelo falla sistematicamente en un lado.
- ROC-AUC: mide la capacidad de ordenar los casos por probabilidad, sin depender
  del umbral de 0.5. Se incluye porque el bosque no solo predice una clase, tambien
  entrega una probabilidad, y el AUC aprovecha esa informacion que la exactitud
  descarta.
- F1: resumen de precision y sensibilidad, para completar el panorama.
"""

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, learning_curve

from modelo import construir_validacion_cruzada

# Nombre de la metrica principal, para que quede en un solo lugar
METRICA_PRINCIPAL = "exactitud"


def calcular_metricas(y_real, y_predicho, probabilidades=None):
    """Calcula todas las metricas de un conjunto de predicciones.

    `probabilidades` es opcional: si se pasa, se agrega el ROC-AUC, que necesita
    las probabilidades y no solo la clase predicha.
    """
    mc = confusion_matrix(y_real, y_predicho)
    vn, fp, fn, vp = mc.ravel()

    resultado = {
        "matriz": {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp)},
        "total": int(len(y_real)),
        "exactitud": accuracy_score(y_real, y_predicho),
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "sensibilidad": recall_score(y_real, y_predicho, zero_division=0),
        # La especificidad no viene en sklearn como funcion propia, pero es la
        # sensibilidad de la clase negativa: se obtiene invirtiendo las etiquetas
        "especificidad": recall_score(y_real, y_predicho, pos_label=0,
                                      zero_division=0),
        "f1": f1_score(y_real, y_predicho, zero_division=0),
    }
    if probabilidades is not None:
        resultado["roc_auc"] = roc_auc_score(y_real, probabilidades)
    return resultado


def texto_matriz_confusion(mc, nombres=("No transportado", "Si transportado")):
    """Formatea la matriz de confusion como tabla para la consola."""
    vn, fp, fn, vp = mc["VN"], mc["FP"], mc["FN"], mc["VP"]
    return "\n".join([
        " " * 22 + "PREDICHO",
        " " * 20 + f"{nombres[0]:>18} {nombres[1]:>18}",
        " " * 18 + "+" + "-" * 38 + "+",
        f"REAL {nombres[0]:<13}|{vn:>18} {fp:>18} |  total {vn + fp}",
        f"     {nombres[1]:<13}|{fn:>18} {vp:>18} |  total {fn + vp}",
        " " * 18 + "+" + "-" * 38 + "+",
        " " * 20 + f"{vn + fn:>18} {fp + vp:>18}",
        "",
        f"  Verdaderos negativos (VN): {vn:5d}   acerto que no fue transportado",
        f"  Falsos positivos     (FP): {fp:5d}   dijo que si, pero no fue",
        f"  Falsos negativos     (FN): {fn:5d}   dijo que no, pero si fue",
        f"  Verdaderos positivos (VP): {vp:5d}   acerto que si fue transportado",
    ])


def texto_metricas(m):
    """Formatea las metricas para la consola, marcando la principal."""
    lineas = [
        f"  Exactitud (accuracy).....: {m['exactitud']:.4f}  "
        f"({m['exactitud'] * 100:.2f}%)   <-- metrica principal",
        f"  Precision................: {m['precision']:.4f}",
        f"  Sensibilidad (recall)....: {m['sensibilidad']:.4f}",
        f"  Especificidad............: {m['especificidad']:.4f}",
        f"  Puntaje F1...............: {m['f1']:.4f}",
    ]
    if "roc_auc" in m:
        lineas.append(f"  ROC-AUC..................: {m['roc_auc']:.4f}")
    return "\n".join(lineas)


def validacion_cruzada(modelo, X, y):
    """Evalua el modelo con validacion cruzada estratificada.

    Es una medida mas estable que una sola particion: entrena y evalua varias
    veces con reparticiones distintas, y la desviacion estandar entre pliegues
    dice que tanto depende el resultado de como se partieron los datos.
    """
    puntajes = cross_val_score(modelo, X, y, cv=construir_validacion_cruzada(),
                               scoring="accuracy", n_jobs=-1)
    return {
        "puntajes": puntajes.tolist(),
        "media": float(puntajes.mean()),
        "desviacion": float(puntajes.std()),
    }


def importancia_por_permutacion(modelo, X, y, repeticiones=10, semilla=42):
    """Importancia de variables medida por permutacion.

    Como funciona: se revuelven al azar los valores de una columna y se mide
    cuanto empeora la exactitud. Si al desordenarla el modelo empeora mucho, esa
    variable era importante.

    Se prefiere sobre la importancia por impureza que trae el bosque
    (feature_importances_) porque esa ultima tiende a inflar las variables
    numericas continuas y las categoricas con muchos niveles, simplemente porque
    ofrecen mas puntos de corte posibles. La permutacion mide el efecto real sobre
    el desempenio.

    Se calcula sobre el conjunto de PRUEBA, no el de entrenamiento: interesa saber
    que variables ayudan a generalizar, no cuales uso el modelo para memorizar.
    """
    resultado = permutation_importance(
        modelo, X, y, n_repeats=repeticiones, random_state=semilla,
        scoring="accuracy", n_jobs=-1)

    pares = sorted(zip(X.columns, resultado.importances_mean,
                       resultado.importances_std),
                   key=lambda t: t[1], reverse=True)
    return [{"variable": v, "importancia": float(m), "desviacion": float(s)}
            for v, m, s in pares]


def curva_roc(y_real, probabilidades):
    """Puntos de la curva ROC, para graficarla."""
    fpr, tpr, _ = roc_curve(y_real, probabilidades)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(y_real, probabilidades))}


def curva_aprendizaje(modelo, X, y, fracciones=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """Exactitud segun cuantos datos de entrenamiento se usan.

    Sirve para responder si conviene juntar mas datos: si la curva de validacion
    todavia va subiendo al usar el 100%, mas datos ayudarian; si ya se aplano, el
    limite esta en el modelo y no en la cantidad de datos.
    """
    tamanios, puntajes_ent, puntajes_val = learning_curve(
        modelo, X, y, train_sizes=list(fracciones),
        cv=construir_validacion_cruzada(), scoring="accuracy", n_jobs=-1)
    return {
        "tamanios": tamanios.tolist(),
        "entrenamiento": puntajes_ent.mean(axis=1).tolist(),
        "validacion": puntajes_val.mean(axis=1).tolist(),
    }


def reporte_sklearn(y_real, y_predicho):
    """Reporte de clasificacion que trae sklearn, como referencia cruzada.

    Se incluye para confirmar que las metricas calculadas arriba coinciden con las
    que reporta la libreria por su cuenta.
    """
    return classification_report(
        y_real, y_predicho, target_names=["No transportado", "Si transportado"],
        digits=4)
