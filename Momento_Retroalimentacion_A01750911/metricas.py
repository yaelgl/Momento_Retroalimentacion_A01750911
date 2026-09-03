"""
Matriz de confusion y metricas de evaluacion, calculadas a mano.

Todas las formulas estan escritas explicitamente en lugar de usar
sklearn.metrics, porque el entregable pide que no se importen algoritmos ya
implementados.

Convencion usada en todo el archivo: la clase positiva es 1 (el pasajero SI fue
transportado) y la negativa es 0.

    VP (verdadero positivo)  el modelo dijo 1 y era 1
    VN (verdadero negativo)  el modelo dijo 0 y era 0
    FP (falso positivo)      el modelo dijo 1 pero era 0
    FN (falso negativo)      el modelo dijo 0 pero era 1
"""


def matriz_confusion(y_real, y_predicho):
    """Devuelve un diccionario con los cuatro conteos de la matriz de confusion.

    La matriz se lee asi:

                        Predicho: 0    Predicho: 1
        Real: 0             VN             FP
        Real: 1             FN             VP
    """
    if len(y_real) != len(y_predicho):
        raise ValueError("las listas deben tener el mismo largo")

    vp = vn = fp = fn = 0
    for real, predicho in zip(y_real, y_predicho):
        if real == 1 and predicho == 1:
            vp += 1
        elif real == 0 and predicho == 0:
            vn += 1
        elif real == 0 and predicho == 1:
            fp += 1
        else:
            fn += 1
    return {"VP": vp, "VN": vn, "FP": fp, "FN": fn}


def _division_segura(numerador, denominador):
    """Divide evitando el error de division entre cero.

    Puede pasar, por ejemplo, si el modelo nunca predice la clase positiva: la
    precision quedaria como 0/0. En ese caso se devuelve 0.
    """
    return numerador / denominador if denominador else 0.0


def calcular_metricas(y_real, y_predicho):
    """Calcula todas las metricas a partir de las etiquetas reales y predichas.

    Que significa cada una:

    exactitud (accuracy)
        Proporcion de aciertos sobre el total. Es la metrica mas directa y aqui
        es apropiada porque las clases estan balanceadas (~50/50). En un problema
        desbalanceado seria enganiosa: con 95% de una clase, predecir siempre esa
        clase daria 95% de exactitud sin haber aprendido nada.

    precision
        De todos los que el modelo dijo que fueron transportados, cuantos si lo
        fueron. Responde "cuando el modelo dice si, que tan confiable es".

    sensibilidad (recall)
        De todos los que realmente fueron transportados, cuantos detecto el
        modelo. Responde "que tanto de lo que hay que encontrar encuentra".

    especificidad
        Lo mismo que la sensibilidad pero para la clase negativa. Se incluye
        porque en un problema balanceado interesa que el modelo funcione igual de
        bien en las dos clases, no solo en la positiva.

    F1
        Media armonica de precision y sensibilidad. Se usa la armonica y no la
        aritmetica porque castiga los desequilibrios: si una de las dos es muy
        baja, el F1 tambien lo es.
    """
    mc = matriz_confusion(y_real, y_predicho)
    vp, vn, fp, fn = mc["VP"], mc["VN"], mc["FP"], mc["FN"]
    total = vp + vn + fp + fn

    exactitud = _division_segura(vp + vn, total)
    precision = _division_segura(vp, vp + fp)
    sensibilidad = _division_segura(vp, vp + fn)
    especificidad = _division_segura(vn, vn + fp)
    f1 = _division_segura(2 * precision * sensibilidad, precision + sensibilidad)

    return {
        "matriz": mc,
        "total": total,
        "exactitud": exactitud,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "f1": f1,
        # Promedio de sensibilidad y especificidad. Coincide con la exactitud
        # cuando las clases estan balanceadas, y sirve para confirmar que el
        # modelo no esta favoreciendo a una clase.
        "exactitud_balanceada": (sensibilidad + especificidad) / 2,
    }


def texto_matriz_confusion(mc, nombres=("No transportado", "Si transportado")):
    """Formatea la matriz de confusion como una tabla legible en consola."""
    vp, vn, fp, fn = mc["VP"], mc["VN"], mc["FP"], mc["FN"]
    ancho = max(len(n) for n in nombres) + 2

    lineas = [
        " " * (ancho + 12) + "PREDICHO",
        " " * (ancho + 2) + f"{nombres[0]:>18} {nombres[1]:>18}",
        " " * ancho + "+" + "-" * 38 + "+",
        f"{'REAL ' + nombres[0]:<{ancho}}|{vn:>18} {fp:>18} |   total {vn + fp}",
        f"{'     ' + nombres[1]:<{ancho}}|{fn:>18} {vp:>18} |   total {fn + vp}",
        " " * ancho + "+" + "-" * 38 + "+",
        " " * (ancho + 2) + f"{vn + fn:>18} {fp + vp:>18}",
        "",
        f"  Verdaderos negativos (VN): {vn:5d}   acerto que no fue transportado",
        f"  Falsos positivos     (FP): {fp:5d}   dijo que si, pero no fue",
        f"  Falsos negativos     (FN): {fn:5d}   dijo que no, pero si fue",
        f"  Verdaderos positivos (VP): {vp:5d}   acerto que si fue transportado",
    ]
    return "\n".join(lineas)


def texto_metricas(m):
    """Formatea las metricas como una lista legible en consola."""
    return "\n".join([
        f"  Exactitud (accuracy).....: {m['exactitud']:.4f}  ({m['exactitud'] * 100:.2f}%)",
        f"  Precision................: {m['precision']:.4f}",
        f"  Sensibilidad (recall)....: {m['sensibilidad']:.4f}",
        f"  Especificidad............: {m['especificidad']:.4f}",
        f"  Puntaje F1...............: {m['f1']:.4f}",
        f"  Exactitud balanceada.....: {m['exactitud_balanceada']:.4f}",
    ])
