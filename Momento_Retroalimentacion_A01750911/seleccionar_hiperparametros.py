"""
Seleccion de hiperparametros con un conjunto de validacion.

Por que existe este archivo aparte
----------------------------------
Es tentador probar varias profundidades, ver cual da la mejor exactitud en el
conjunto de prueba y quedarse con esa. El problema es que asi el conjunto de
prueba deja de ser una medida honesta: se estaria eligiendo el modelo que mejor
le queda a esos datos en particular, y la exactitud reportada saldria optimista.

La forma correcta es partir el conjunto de ENTRENAMIENTO otra vez:

    train.csv (8693)
      |
      +-- entrenamiento (70%)  ->  +-- sub-entrenamiento (80%)  <- se ajusta el arbol
      |                            +-- validacion       (20%)  <- se elige la profundidad
      |
      +-- prueba (30%)  <- no se toca hasta el final, se mide UNA sola vez

Se elige la combinacion con mejor exactitud en validacion, se vuelve a entrenar
con todo el conjunto de entrenamiento usando esa combinacion, y solo entonces se
mide en prueba.

    python seleccionar_hiperparametros.py
"""

from pathlib import Path

import preparar_datos as prep
from arbol_decision import ArbolDecision
from metricas import calcular_metricas

CARPETA = Path(__file__).parent
RUTA_TRAIN = CARPETA / "dataset" / "train.csv"

# Valores a probar. Se mantiene una rejilla chica a proposito: con pocas
# combinaciones el riesgo de elegir una buena por casualidad es menor.
PROFUNDIDADES = list(range(1, 16))
MINIMOS_HOJA = [5, 10, 20, 40]


def exactitud(arbol, X, y):
    """Atajo para calcular solo la exactitud."""
    return calcular_metricas(y, arbol.predecir(X))["exactitud"]


def main():
    print()
    print("SELECCION DE HIPERPARAMETROS CON CONJUNTO DE VALIDACION")
    print("=" * 70)

    # Division principal: la misma que usa main.py, con la misma semilla
    X_entrena, y_entrena, X_prueba, y_prueba, _ = prep.cargar_datos(
        RUTA_TRAIN, proporcion_prueba=0.3, semilla=42)

    # Segunda division, solo dentro del conjunto de entrenamiento
    X_sub, y_sub, X_val, y_val = prep.dividir_estratificado(
        X_entrena, y_entrena, proporcion_prueba=0.2, semilla=42)

    print(f"Sub-entrenamiento: {len(X_sub)} filas")
    print(f"Validacion:        {len(X_val)} filas")
    print(f"Prueba (reservada):{len(X_prueba):5d} filas  <- no se usa en esta busqueda")
    print()
    print(f"Combinaciones a probar: {len(PROFUNDIDADES) * len(MINIMOS_HOJA)}")
    print()
    print(f"{'prof':>5} {'min_hoja':>9} {'sub-entr':>10} {'validacion':>11}")
    print("-" * 40)

    resultados = []
    for min_hoja in MINIMOS_HOJA:
        for profundidad in PROFUNDIDADES:
            arbol = ArbolDecision(
                prep.NUMERICAS, prep.CATEGORICAS,
                max_profundidad=profundidad,
                min_muestras_division=max(2 * min_hoja, 4),
                min_muestras_hoja=min_hoja)
            arbol.entrenar(X_sub, y_sub)

            ex_sub = exactitud(arbol, X_sub, y_sub)
            ex_val = exactitud(arbol, X_val, y_val)
            resultados.append({"profundidad": profundidad, "min_hoja": min_hoja,
                               "sub_entrena": ex_sub, "validacion": ex_val})
            print(f"{profundidad:5d} {min_hoja:9d} {ex_sub:10.4f} {ex_val:11.4f}")
        print("-" * 40)

    # Si dos combinaciones empatan en validacion se prefiere la del arbol mas
    # simple (menor profundidad y hojas mas grandes). Entre dos modelos que
    # rinden igual, el mas simple suele generalizar mejor a datos nuevos.
    mejor = max(resultados,
                key=lambda r: (r["validacion"], -r["profundidad"], r["min_hoja"]))
    print()
    print("MEJOR COMBINACION SEGUN VALIDACION")
    print(f"  profundidad maxima  : {mejor['profundidad']}")
    print(f"  minimo por hoja     : {mejor['min_hoja']}")
    print(f"  minimo para dividir : {max(2 * mejor['min_hoja'], 4)}")
    print(f"  exactitud validacion: {mejor['validacion']:.4f}")

    # Reentrenar con TODO el conjunto de entrenamiento y medir en prueba una vez
    print()
    print("VERIFICACION FINAL (una sola medicion en el conjunto de prueba)")
    final = ArbolDecision(
        prep.NUMERICAS, prep.CATEGORICAS,
        max_profundidad=mejor["profundidad"],
        min_muestras_division=max(2 * mejor["min_hoja"], 4),
        min_muestras_hoja=mejor["min_hoja"])
    final.entrenar(X_entrena, y_entrena)

    ex_entrena = exactitud(final, X_entrena, y_entrena)
    ex_prueba = exactitud(final, X_prueba, y_prueba)
    print(f"  exactitud entrenamiento: {ex_entrena:.4f}")
    print(f"  exactitud prueba       : {ex_prueba:.4f}")
    print(f"  brecha                 : {ex_entrena - ex_prueba:+.4f}")
    print()
    print("Estos son los valores que quedan como predeterminados en main.py.")
    print()


if __name__ == "__main__":
    main()
