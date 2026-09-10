"""
Definicion del modelo y busqueda de hiperparametros con scikit-learn.

El algoritmo elegido es Random Forest. Es la contraparte natural de la Parte I del
modulo, donde se programo un arbol de decision a mano: aqui se usa el framework y
en lugar de un arbol se entrena un bosque de muchos arboles que votan.

Por que un bosque funciona mejor que un arbol solo
--------------------------------------------------
Un arbol de decision es inestable: si se cambia un poco el conjunto de
entrenamiento, puede elegir divisiones distintas y quedar un arbol bastante
diferente. Tiene varianza alta.

El bosque ataca eso con dos fuentes de aleatoriedad:

1. Bagging. Cada arbol se entrena con una muestra distinta de los datos, sacada
   con reemplazo del conjunto original.
2. Submuestreo de variables. En cada division, cada arbol solo puede elegir entre
   un subconjunto aleatorio de las variables (parametro max_features).

Los arboles quedan distintos entre si y sus errores tienden a cancelarse al
promediar los votos. Ninguno de los dos mecanismos reduce el sesgo, pero juntos
reducen mucho la varianza, y por eso el bosque generaliza mejor.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preparar_datos import (DerivarVariables, SEMILLA, construir_preprocesador)

# Numero de pliegues de la validacion cruzada
PLIEGUES = 5

# Hiperparametros que quedaron despues de la busqueda. Se dejan como constantes
# para que main.py pueda entrenar rapido sin repetir el GridSearchCV completo.
MEJORES = {
    "clasificador__n_estimators": 200,
    "clasificador__max_depth": 14,
    "clasificador__min_samples_leaf": 5,
    "clasificador__max_features": "sqrt",
}


def construir_modelo(**parametros):
    """Arma el Pipeline completo: derivar variables, preprocesar y clasificar.

    Meter los tres pasos en un solo Pipeline tiene dos ventajas concretas:

    - No hay fuga de informacion. Cuando GridSearchCV hace validacion cruzada,
      vuelve a ajustar el imputador y el codificador dentro de cada pliegue,
      usando solo los datos de entrenamiento de ese pliegue.
    - El objeto resultante recibe datos crudos y devuelve predicciones, asi que
      predecir sobre datos nuevos es una sola llamada y no hay riesgo de aplicar
      las transformaciones en distinto orden.
    """
    modelo = Pipeline([
        ("derivar", DerivarVariables()),
        ("preprocesar", construir_preprocesador()),
        ("clasificador", RandomForestClassifier(
            # Cuantos arboles tiene el bosque. Mas arboles siempre es igual o
            # mejor en desempenio (no sobreajusta por subirlo), pero el tiempo de
            # entrenamiento crece linealmente y la mejora se aplana rapido. En la
            # busqueda, 400 arboles empataron con 200: se quedo el mas chico.
            n_estimators=200,

            # Profundidad maxima de cada arbol. Es el freno principal contra el
            # sobreajuste, igual que en la Parte I.
            max_depth=14,

            # Minimo de muestras que debe quedar en cada hoja. Subirlo suaviza el
            # modelo porque evita hojas construidas sobre 1 o 2 ejemplos.
            min_samples_leaf=5,

            # Cuantas variables considera en cada division. "sqrt" toma la raiz
            # cuadrada del total (aqui unas 6 de 35). Este parametro es el que
            # hace que los arboles sean distintos entre si: si se pusiera None,
            # todos verian todas las variables y quedarian muy parecidos, con lo
            # que se perderia la ventaja de promediar.
            max_features="sqrt",

            # Fija la aleatoriedad del bagging y del submuestreo de variables,
            # para que el resultado sea reproducible.
            random_state=SEMILLA,

            # Usa todos los nucleos disponibles. Los arboles son independientes
            # entre si, asi que el entrenamiento se paraleliza sin problema.
            n_jobs=-1,
        )),
    ])

    if parametros:
        modelo.set_params(**parametros)
    return modelo


def construir_validacion_cruzada():
    """Estrategia de validacion cruzada estratificada.

    Estratificada quiere decir que cada pliegue conserva la proporcion de clases
    del total. Con clases balanceadas casi al 50% no cambia mucho, pero es la
    opcion correcta por costumbre: garantiza que ningun pliegue quede sesgado.
    """
    return StratifiedKFold(n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA)


def buscar_hiperparametros(X, y, verbose=1):
    """Busca la mejor combinacion de hiperparametros con GridSearchCV.

    GridSearchCV prueba todas las combinaciones de la rejilla y evalua cada una
    con validacion cruzada, o sea entrenando y midiendo PLIEGUES veces por
    combinacion. Se queda con la que da mejor promedio.

    Punto importante del procedimiento: la busqueda recibe unicamente el conjunto
    de ENTRENAMIENTO. El conjunto de prueba no participa, y se mide una sola vez
    al final con los parametros ya fijados. Si se eligieran los parametros
    mirando la prueba, esa medicion dejaria de ser honesta porque estaria
    escogiendo el modelo que mejor le queda justamente a esos datos.

    La metrica que optimiza es la exactitud, por la razon que se explica en
    evaluacion.py: las clases estan balanceadas y los dos tipos de error cuestan
    lo mismo en este problema.
    """
    rejilla = {
        "clasificador__n_estimators": [200, 400],
        "clasificador__max_depth": [10, 14, 18, None],
        "clasificador__min_samples_leaf": [1, 3, 5],
        "clasificador__max_features": ["sqrt", 0.4],
    }

    busqueda = GridSearchCV(
        estimator=construir_modelo(),
        param_grid=rejilla,
        scoring="accuracy",
        cv=construir_validacion_cruzada(),
        n_jobs=-1,
        verbose=verbose,
        return_train_score=True,   # sirve para ver la brecha entrenamiento/validacion
        refit=True,                # deja el mejor modelo ya reentrenado con todo
    )
    busqueda.fit(X, y)
    return busqueda


def resumen_busqueda(busqueda, top=10):
    """Devuelve las mejores combinaciones de la busqueda como DataFrame."""
    import pandas as pd

    columnas = ["param_clasificador__n_estimators",
                "param_clasificador__max_depth",
                "param_clasificador__min_samples_leaf",
                "param_clasificador__max_features",
                "mean_train_score", "mean_test_score", "std_test_score",
                "rank_test_score"]
    tabla = pd.DataFrame(busqueda.cv_results_)[columnas]
    tabla = tabla.sort_values("rank_test_score").head(top)
    tabla.columns = ["arboles", "profundidad", "min_hoja", "max_variables",
                     "entrenamiento", "validacion", "desv_est", "lugar"]
    return tabla.reset_index(drop=True)
