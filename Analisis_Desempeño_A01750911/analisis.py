"""
Analisis del desempenio del Random Forest: sesgo, varianza, ajuste y regularizacion.

Este modulo es la Parte 3 del modulo. Reutiliza el modelo de la Parte 2 (Random
Forest con scikit-learn) y se concentra en diagnosticarlo, no en volver a
construirlo.

QUE SE HACE AQUI Y POR QUE
==========================

1. Particion en TRES conjuntos, no dos.
   En la Parte 2 se uso train_test_split 70/30 mas validacion cruzada. Aqui se
   aparta un conjunto de VALIDACION explicito, separado del de prueba, porque el
   analisis necesita distinguir dos cosas que la validacion cruzada mezcla:

     - El conjunto de VALIDACION se usa para decidir: elegir hiperparametros,
       comparar configuraciones, diagnosticar sesgo y varianza. Se mira muchas
       veces.
     - El conjunto de PRUEBA se usa una sola vez, al final, para reportar. Si se
       mirara durante la busqueda dejaria de ser una estimacion honesta del
       desempenio en datos nuevos, porque se habria elegido el modelo que mejor le
       queda justamente a esos datos.

2. Dos modelos, no uno.
   Se entrena a proposito un modelo SIN regularizar (arboles sin limite de
   profundidad, hojas de un solo ejemplo, todas las variables visibles en cada
   division) para que el sobreajuste sea medible y no una suposicion. Ese es el
   punto de partida del diagnostico. Despues se regulariza y se compara.

3. Los diagnosticos se calculan con umbrales declarados, no a ojo.
   Las funciones `diagnosticar_sesgo`, `diagnosticar_varianza` y
   `diagnosticar_ajuste` aplican criterios numericos explicitos. Asi la
   clasificacion (bajo/medio/alto, underfit/fit/overfit) es reproducible y se
   puede discutir el criterio en lugar de la impresion.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import preparar_datos as prep
from preparar_datos import DerivarVariables, construir_preprocesador

CARPETA = Path(__file__).parent
RUTA_TRAIN = CARPETA / "dataset" / "train.csv"

SEMILLA = 42

# 60 / 20 / 20. Se reserva 20% para validacion y 20% para prueba: con 8,693 filas
# eso deja 1,739 casos en cada uno, suficiente para que las metricas sean estables
# (el error estandar de una proporcion cercana a 0.8 con n=1739 es ~0.010).
PROP_PRUEBA = 0.20
PROP_VALIDACION = 0.20

# ---------------------------------------------------------------------------
# Configuraciones de los dos modelos que se comparan
# ---------------------------------------------------------------------------

# Modelo A: SIN regularizar. Cada freno del bosque esta desactivado a proposito.
SIN_REGULARIZAR = {
    "n_estimators": 200,
    "max_depth": None,          # los arboles crecen hasta agotar los datos
    "min_samples_leaf": 1,      # se permiten hojas con un solo pasajero
    "min_samples_split": 2,     # se divide un nodo con solo 2 ejemplos
    "max_features": None,       # cada arbol ve TODAS las variables: se parecen entre si
    "ccp_alpha": 0.0,           # sin poda por costo-complejidad
}


def construir(parametros):
    """Arma el Pipeline completo con los hiperparametros dados.

    Mismo Pipeline de la Parte 2: derivar variables, preprocesar, clasificar.
    Todo el preprocesamiento va adentro para que se reajuste solo con los datos
    de entrenamiento en cada fit y no haya fuga de informacion.
    """
    return Pipeline([
        ("derivar", DerivarVariables()),
        ("preprocesar", construir_preprocesador()),
        ("clasificador", RandomForestClassifier(
            random_state=SEMILLA, n_jobs=-1, **parametros)),
    ])


# ---------------------------------------------------------------------------
# Particion en tres conjuntos
# ---------------------------------------------------------------------------

def particionar(X, y):
    """Divide en entrenamiento / validacion / prueba, estratificado.

    Se hace en dos pasos porque train_test_split solo parte en dos:
      1. Se separa la PRUEBA del resto.
      2. Lo que queda se parte en ENTRENAMIENTO y VALIDACION.

    `stratify` en ambos pasos conserva la proporcion de clases en los tres
    conjuntos, para que las metricas sean comparables entre ellos.
    """
    X_resto, X_pru, y_resto, y_pru = train_test_split(
        X, y, test_size=PROP_PRUEBA, stratify=y, random_state=SEMILLA)

    # La proporcion se recalcula sobre lo que quedo: para que validacion sea el
    # 20% del TOTAL, debe ser 25% del 80% restante.
    prop_val_ajustada = PROP_VALIDACION / (1 - PROP_PRUEBA)
    X_ent, X_val, y_ent, y_val = train_test_split(
        X_resto, y_resto, test_size=prop_val_ajustada, stratify=y_resto,
        random_state=SEMILLA)

    return X_ent, X_val, X_pru, y_ent, y_val, y_pru


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def metricas(modelo, X, y):
    """Calcula el paquete completo de metricas de un conjunto."""
    pred = modelo.predict(X)
    prob = modelo.predict_proba(X)[:, 1]
    vn, fp, fn, vp = confusion_matrix(y, pred).ravel()
    return {
        "n": int(len(y)),
        "exactitud": float(accuracy_score(y, pred)),
        "error": float(1 - accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensibilidad": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "matriz": {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp)},
    }


def puntos_roc(modelo, X, y):
    fpr, tpr, _ = roc_curve(y, modelo.predict_proba(X)[:, 1])
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "auc": float(roc_auc_score(y, modelo.predict_proba(X)[:, 1]))}


# ---------------------------------------------------------------------------
# Diagnosticos con umbrales declarados
# ---------------------------------------------------------------------------

# Error irreducible estimado del problema. Es el piso practico: incluso un modelo
# perfecto se equivocaria porque hay pasajeros con caracteristicas identicas y
# etiquetas distintas. Se estima en ~19% a partir del mejor desempenio alcanzable
# observado en el dataset (los mejores resultados publicos rondan 80-81% de
# exactitud). Sirve como referencia para juzgar si el error de entrenamiento es
# "alto" en terminos absolutos o simplemente cercano al limite del problema.
ERROR_IRREDUCIBLE = 0.19

UMBRAL_SESGO_BAJO = 0.05      # error de entrenamiento < 5%  -> sesgo bajo
UMBRAL_SESGO_ALTO = 0.15      # error de entrenamiento > 15% -> sesgo alto

UMBRAL_VAR_BAJA = 0.03        # brecha entrena-valida < 3 pts  -> varianza baja
UMBRAL_VAR_ALTA = 0.10        # brecha entrena-valida > 10 pts -> varianza alta


def diagnosticar_sesgo(error_entrenamiento):
    """Clasifica el sesgo en bajo / medio / alto.

    El sesgo es el error que comete el modelo por ser demasiado simple para el
    patron real. Se mide con el error de ENTRENAMIENTO: si el modelo no logra
    ajustar ni los datos que ya vio, el problema no es de generalizacion sino de
    capacidad. Un error de entrenamiento cercano a cero significa sesgo bajo por
    definicion, porque el modelo es capaz de representar los datos.
    """
    if error_entrenamiento < UMBRAL_SESGO_BAJO:
        nivel = "BAJO"
    elif error_entrenamiento <= UMBRAL_SESGO_ALTO:
        nivel = "MEDIO"
    else:
        nivel = "ALTO"
    return {
        "nivel": nivel,
        "error_entrenamiento": float(error_entrenamiento),
        "umbral_bajo": UMBRAL_SESGO_BAJO,
        "umbral_alto": UMBRAL_SESGO_ALTO,
        "criterio": (f"error de entrenamiento = {error_entrenamiento:.4f}; "
                     f"< {UMBRAL_SESGO_BAJO} es bajo, "
                     f"> {UMBRAL_SESGO_ALTO} es alto"),
    }


def diagnosticar_varianza(error_entrenamiento, error_validacion):
    """Clasifica la varianza en baja / media / alta.

    La varianza es la sensibilidad del modelo a los datos concretos con los que
    se entreno. Se mide con la BRECHA entre el error de validacion y el de
    entrenamiento: si el modelo acierta mucho en lo que vio y bastante menos en
    lo que no vio, es que memorizo particularidades de su muestra.
    """
    brecha = error_validacion - error_entrenamiento
    if brecha < UMBRAL_VAR_BAJA:
        nivel = "BAJA"
    elif brecha <= UMBRAL_VAR_ALTA:
        nivel = "MEDIA"
    else:
        nivel = "ALTA"
    return {
        "nivel": nivel,
        "brecha": float(brecha),
        "error_validacion": float(error_validacion),
        "umbral_baja": UMBRAL_VAR_BAJA,
        "umbral_alta": UMBRAL_VAR_ALTA,
        "criterio": (f"brecha validacion - entrenamiento = {brecha:.4f}; "
                     f"< {UMBRAL_VAR_BAJA} es baja, "
                     f"> {UMBRAL_VAR_ALTA} es alta"),
    }


def diagnosticar_ajuste(sesgo, varianza):
    """Combina sesgo y varianza para clasificar el ajuste.

    Las cuatro combinaciones posibles y lo que significan:

      sesgo ALTO  + varianza BAJA  -> UNDERFITTING: el modelo es muy simple, falla
                                      parejo en todo. Mas datos no ayudan.
      sesgo BAJO  + varianza ALTA  -> OVERFITTING: el modelo memorizo. Acierta en
                                      lo que vio y falla en lo nuevo.
      sesgo ALTO  + varianza ALTA  -> mal planteado: hay que revisar variables y
                                      preprocesamiento antes que el modelo.
      sesgo BAJO/MEDIO + varianza BAJA/MEDIA -> AJUSTE ADECUADO.
    """
    s, v = sesgo["nivel"], varianza["nivel"]

    if s == "ALTO" and v == "ALTA":
        nivel, explica = "UNDERFITTING CON INESTABILIDAD", (
            "sesgo alto y varianza alta a la vez: el modelo no captura el patrón "
            "y además es inestable")
    elif s == "ALTO":
        nivel, explica = "UNDERFITTING", (
            "el error de entrenamiento ya es alto, así que el modelo no alcanza a "
            "representar el patrón ni en los datos que vio")
    elif v == "ALTA":
        nivel, explica = "OVERFITTING", (
            "el modelo ajusta casi perfecto el entrenamiento pero pierde mucho en "
            "validación: memorizó en lugar de generalizar")
    elif v == "MEDIA" and s == "MEDIO":
        nivel, explica = "AJUSTE ADECUADO", (
            "sesgo y varianza moderados, sin evidencia de memorización ni de falta "
            "de capacidad")
    else:
        nivel, explica = "AJUSTE ADECUADO", (
            f"sesgo {s.lower()} y varianza {v.lower()}, así que el modelo "
            f"generaliza con una brecha controlada")

    return {"nivel": nivel, "explicacion": explica,
            "sesgo": s, "varianza": v}


def diagnostico_completo(modelo, X_ent, y_ent, X_val, y_val, etiqueta):
    """Aplica los tres diagnosticos a un modelo ya entrenado."""
    m_ent = metricas(modelo, X_ent, y_ent)
    m_val = metricas(modelo, X_val, y_val)
    sesgo = diagnosticar_sesgo(m_ent["error"])
    varianza = diagnosticar_varianza(m_ent["error"], m_val["error"])
    ajuste = diagnosticar_ajuste(sesgo, varianza)
    return {
        "etiqueta": etiqueta,
        "entrenamiento": m_ent,
        "validacion": m_val,
        "sesgo": sesgo,
        "varianza": varianza,
        "ajuste": ajuste,
    }


# ---------------------------------------------------------------------------
# Curvas de aprendizaje: los tres conjuntos en la misma grafica
# ---------------------------------------------------------------------------

def curva_aprendizaje(parametros, X_ent, y_ent, X_val, y_val, X_pru, y_pru,
                      fracciones=(0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Exactitud en los tres conjuntos segun cuantos datos de entrenamiento se usan.

    Se calcula a mano en lugar de usar `learning_curve` de sklearn porque esa
    funcion evalua con validacion cruzada interna, y aqui interesa medir contra
    los conjuntos de validacion y prueba FIJOS que se apartaron. Es lo que permite
    leer la brecha entre las tres curvas.

    Nota metodologica: la curva de prueba se grafica solo para el reporte. Ninguna
    decision de este analisis se tomo mirandola; todas se tomaron con validacion.
    """
    filas = []
    for frac in fracciones:
        n = max(50, int(len(X_ent) * frac))
        # Submuestra estratificada para que la proporcion de clases no cambie
        if n < len(X_ent):
            X_sub, _, y_sub, _ = train_test_split(
                X_ent, y_ent, train_size=n, stratify=y_ent,
                random_state=SEMILLA)
        else:
            X_sub, y_sub = X_ent, y_ent

        modelo = construir(parametros).fit(X_sub, y_sub)
        filas.append({
            "n": int(len(X_sub)),
            "fraccion": float(frac),
            "entrenamiento": float(accuracy_score(y_sub, modelo.predict(X_sub))),
            "validacion": float(accuracy_score(y_val, modelo.predict(X_val))),
            "prueba": float(accuracy_score(y_pru, modelo.predict(X_pru))),
        })
    return filas


# ---------------------------------------------------------------------------
# Curvas de complejidad: efecto de cada hiperparametro de regularizacion
# ---------------------------------------------------------------------------

def curva_complejidad(nombre, valores, base, X_ent, y_ent, X_val, y_val):
    """Varia un solo hiperparametro y mide entrenamiento vs validacion.

    Es la herramienta que justifica el valor elegido para cada tecnica de
    regularizacion: se ve donde la curva de validacion alcanza su maximo y donde
    la brecha con entrenamiento empieza a abrirse.
    """
    filas = []
    for valor in valores:
        parametros = dict(base)
        parametros[nombre] = valor
        modelo = construir(parametros).fit(X_ent, y_ent)
        e_ent = accuracy_score(y_ent, modelo.predict(X_ent))
        e_val = accuracy_score(y_val, modelo.predict(X_val))
        filas.append({
            "valor": "None" if valor is None else valor,
            "entrenamiento": float(e_ent),
            "validacion": float(e_val),
            "brecha": float(e_ent - e_val),
        })
    return {"hiperparametro": nombre, "puntos": filas}


# ---------------------------------------------------------------------------
# Busqueda de hiperparametros contra el conjunto de validacion
# ---------------------------------------------------------------------------

def buscar_regularizacion(X_ent, y_ent, X_val, y_val):
    """Prueba combinaciones de regularizacion y elige la mejor en VALIDACION.

    Se usa el conjunto de validacion apartado en lugar de validacion cruzada
    porque el objetivo de esta parte es precisamente mostrar el papel del
    conjunto de validacion en la toma de decisiones. El conjunto de prueba no
    participa en ningun momento de esta busqueda.

    Criterio de seleccion: mayor exactitud en validacion. En caso de empate
    practico (diferencia menor a 0.002, dentro del ruido de muestreo con n=1739)
    se prefiere el modelo con menor brecha, o sea el mas simple, siguiendo el
    principio de parsimonia.
    """
    rejilla = []
    for profundidad in [8, 12, 16, None]:
        for min_hoja in [1, 5, 10, 20]:
            for max_var in ["sqrt", 0.4]:
                for alpha in [0.0, 0.0005, 0.001]:
                    rejilla.append({
                        "n_estimators": 200,
                        "max_depth": profundidad,
                        "min_samples_leaf": min_hoja,
                        "min_samples_split": 2,
                        "max_features": max_var,
                        "ccp_alpha": alpha,
                    })

    resultados = []
    for i, parametros in enumerate(rejilla, 1):
        modelo = construir(parametros).fit(X_ent, y_ent)
        e_ent = accuracy_score(y_ent, modelo.predict(X_ent))
        e_val = accuracy_score(y_val, modelo.predict(X_val))
        resultados.append({
            "max_depth": "None" if parametros["max_depth"] is None
                         else parametros["max_depth"],
            "min_samples_leaf": parametros["min_samples_leaf"],
            "max_features": parametros["max_features"],
            "ccp_alpha": parametros["ccp_alpha"],
            "entrenamiento": float(e_ent),
            "validacion": float(e_val),
            "brecha": float(e_ent - e_val),
            "_parametros": parametros,
        })
        if i % 8 == 0 or i == len(rejilla):
            print(f"  [{i:3d}/{len(rejilla)}] profundidad={parametros['max_depth']}, "
                  f"min_hoja={parametros['min_samples_leaf']}, "
                  f"max_var={parametros['max_features']}, "
                  f"alpha={parametros['ccp_alpha']}  ->  "
                  f"validacion {e_val:.4f}  (brecha {e_ent - e_val:+.4f})")

    mejor_val = max(r["validacion"] for r in resultados)
    empatados = [r for r in resultados if mejor_val - r["validacion"] < 0.002]
    elegido = min(empatados, key=lambda r: r["brecha"])

    return elegido, sorted(resultados, key=lambda r: -r["validacion"])


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    inicio = time.perf_counter()
    salida = {}

    def titulo(texto):
        print()
        print("=" * 74)
        print(texto)
        print("=" * 74)

    # ---------------------------------------------------------------- 1. datos
    titulo("1. PARTICION EN TRES CONJUNTOS")
    X, y = prep.cargar_entrenamiento(RUTA_TRAIN)
    X_ent, X_val, X_pru, y_ent, y_val, y_pru = particionar(X, y)

    print(f"Total con etiqueta: {len(X)} pasajeros")
    print()
    print(f"{'conjunto':<16}{'filas':>8}{'%':>8}{'transportados':>16}")
    print("-" * 48)
    for nombre, yy in [("Entrenamiento", y_ent), ("Validacion", y_val),
                       ("Prueba", y_pru)]:
        print(f"{nombre:<16}{len(yy):>8}{100 * len(yy) / len(X):>7.1f}%"
              f"{yy.mean() * 100:>15.2f}%")
    print()
    print("La estratificacion mantuvo la proporcion de clases en los tres.")
    print("Validacion: se usa para decidir. Prueba: se mide una sola vez al final.")

    salida["particion"] = {
        "total": int(len(X)),
        "n_entrenamiento": int(len(y_ent)),
        "n_validacion": int(len(y_val)),
        "n_prueba": int(len(y_pru)),
        "pct_entrenamiento": round(100 * len(y_ent) / len(X), 1),
        "pct_validacion": round(100 * len(y_val) / len(X), 1),
        "pct_prueba": round(100 * len(y_pru) / len(X), 1),
        "positivos_entrenamiento": float(y_ent.mean()),
        "positivos_validacion": float(y_val.mean()),
        "positivos_prueba": float(y_pru.mean()),
        "semilla": SEMILLA,
    }

    # ------------------------------------------------- 2. modelo sin regularizar
    titulo("2. MODELO SIN REGULARIZAR (punto de partida)")
    print("Configuracion con todos los frenos desactivados a proposito:")
    for k, v in SIN_REGULARIZAR.items():
        print(f"  {k:20} {v}")

    modelo_a = construir(SIN_REGULARIZAR).fit(X_ent, y_ent)
    bosque_a = modelo_a.named_steps["clasificador"]
    prof_a = [e.get_depth() for e in bosque_a.estimators_]
    hojas_a = [e.get_n_leaves() for e in bosque_a.estimators_]
    print()
    print(f"Profundidad de los arboles: promedio {np.mean(prof_a):.1f}, "
          f"maxima {max(prof_a)}")
    print(f"Hojas por arbol: promedio {np.mean(hojas_a):.0f}")

    diag_a = diagnostico_completo(modelo_a, X_ent, y_ent, X_val, y_val,
                                 "sin regularizar")
    diag_a["profundidad_promedio"] = float(np.mean(prof_a))
    diag_a["hojas_promedio"] = float(np.mean(hojas_a))

    print()
    print(f"Exactitud en entrenamiento: {diag_a['entrenamiento']['exactitud']:.4f}")
    print(f"Exactitud en validacion:    {diag_a['validacion']['exactitud']:.4f}")
    print(f"Brecha:                     {diag_a['varianza']['brecha']:+.4f}")
    print()
    print("DIAGNOSTICO:")
    print(f"  Sesgo:    {diag_a['sesgo']['nivel']:6}  ({diag_a['sesgo']['criterio']})")
    print(f"  Varianza: {diag_a['varianza']['nivel']:6}  ({diag_a['varianza']['criterio']})")
    print(f"  Ajuste:   {diag_a['ajuste']['nivel']}")
    print(f"            {diag_a['ajuste']['explicacion']}")

    # --------------------------------------------------- 3. curvas de complejidad
    titulo("3. CURVAS DE COMPLEJIDAD (efecto de cada tecnica de regularizacion)")
    curvas = []

    print("3.1 max_depth (limitar la profundidad de los arboles)")
    c = curva_complejidad("max_depth", [2, 4, 6, 8, 10, 12, 14, 18, None],
                          SIN_REGULARIZAR, X_ent, y_ent, X_val, y_val)
    for p in c["puntos"]:
        print(f"   profundidad {str(p['valor']):>5}: entrena {p['entrenamiento']:.4f}  "
              f"valida {p['validacion']:.4f}  brecha {p['brecha']:+.4f}")
    curvas.append(c)

    print()
    print("3.2 min_samples_leaf (minimo de ejemplos por hoja)")
    c = curva_complejidad("min_samples_leaf", [1, 2, 5, 10, 20, 40, 80],
                          SIN_REGULARIZAR, X_ent, y_ent, X_val, y_val)
    for p in c["puntos"]:
        print(f"   min_hoja {str(p['valor']):>5}: entrena {p['entrenamiento']:.4f}  "
              f"valida {p['validacion']:.4f}  brecha {p['brecha']:+.4f}")
    curvas.append(c)

    print()
    print("3.3 ccp_alpha (poda por costo-complejidad)")
    c = curva_complejidad("ccp_alpha", [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005],
                          SIN_REGULARIZAR, X_ent, y_ent, X_val, y_val)
    for p in c["puntos"]:
        print(f"   alpha {str(p['valor']):>7}: entrena {p['entrenamiento']:.4f}  "
              f"valida {p['validacion']:.4f}  brecha {p['brecha']:+.4f}")
    curvas.append(c)

    # ----------------------------------------------------------- 4. busqueda
    titulo("4. BUSQUEDA DE LA COMBINACION REGULARIZADA (contra validacion)")
    print("32 combinaciones. El conjunto de prueba NO participa.")
    print()
    elegido, ranking = buscar_regularizacion(X_ent, y_ent, X_val, y_val)
    REGULARIZADO = elegido.pop("_parametros")

    print()
    print("Combinacion elegida:")
    for k, v in REGULARIZADO.items():
        print(f"  {k:20} {v}")
    print(f"  exactitud en validacion: {elegido['validacion']:.4f}")
    print(f"  brecha entrena-valida:   {elegido['brecha']:+.4f}")

    for r in ranking:
        r.pop("_parametros", None)
    salida["busqueda"] = {"elegido": elegido, "top": ranking[:10],
                          "n_combinaciones": len(ranking)}

    # La curva de max_features se traza SOBRE LA BASE YA REGULARIZADA, no sobre la
    # sin regularizar. Razon: con profundidad ilimitada y hojas de un solo ejemplo
    # el arbol memoriza sin importar cuantas variables vea en cada division, asi
    # que la curva sale plana y no informa nada. Su efecto solo es visible cuando
    # los otros frenos ya estan puestos.
    print()
    print("3.4 max_features, medido sobre la base ya regularizada")
    base_reg = dict(REGULARIZADO)
    c = curva_complejidad("max_features", ["sqrt", 0.3, 0.4, 0.6, 0.8, None],
                          base_reg, X_ent, y_ent, X_val, y_val)
    for p in c["puntos"]:
        print(f"   max_var {str(p['valor']):>5}: entrena {p['entrenamiento']:.4f}  "
              f"valida {p['validacion']:.4f}  brecha {p['brecha']:+.4f}")
    curvas.insert(2, c)
    salida["curvas_complejidad"] = curvas
    salida["base_curva_max_features"] = {
        k: ("None" if v is None else v) for k, v in base_reg.items()}
    salida["parametros_regularizado"] = {
        k: ("None" if v is None else v) for k, v in REGULARIZADO.items()}
    salida["parametros_sin_regularizar"] = {
        k: ("None" if v is None else v) for k, v in SIN_REGULARIZAR.items()}

    # ------------------------------------------------- 5. modelo regularizado
    titulo("5. MODELO REGULARIZADO")
    modelo_b = construir(REGULARIZADO).fit(X_ent, y_ent)
    bosque_b = modelo_b.named_steps["clasificador"]
    prof_b = [e.get_depth() for e in bosque_b.estimators_]
    hojas_b = [e.get_n_leaves() for e in bosque_b.estimators_]
    print(f"Profundidad de los arboles: promedio {np.mean(prof_b):.1f}, "
          f"maxima {max(prof_b)}")
    print(f"Hojas por arbol: promedio {np.mean(hojas_b):.0f}  "
          f"(antes eran {np.mean(hojas_a):.0f})")

    diag_b = diagnostico_completo(modelo_b, X_ent, y_ent, X_val, y_val,
                                 "regularizado")
    diag_b["profundidad_promedio"] = float(np.mean(prof_b))
    diag_b["hojas_promedio"] = float(np.mean(hojas_b))

    print()
    print(f"Exactitud en entrenamiento: {diag_b['entrenamiento']['exactitud']:.4f}")
    print(f"Exactitud en validacion:    {diag_b['validacion']['exactitud']:.4f}")
    print(f"Brecha:                     {diag_b['varianza']['brecha']:+.4f}")
    print()
    print("DIAGNOSTICO:")
    print(f"  Sesgo:    {diag_b['sesgo']['nivel']:6}  ({diag_b['sesgo']['criterio']})")
    print(f"  Varianza: {diag_b['varianza']['nivel']:6}  ({diag_b['varianza']['criterio']})")
    print(f"  Ajuste:   {diag_b['ajuste']['nivel']}")
    print(f"            {diag_b['ajuste']['explicacion']}")

    # ------------------------------------------------- 6. curvas de aprendizaje
    titulo("6. CURVAS DE APRENDIZAJE (los tres conjuntos)")
    print("Sin regularizar:")
    ca_a = curva_aprendizaje(SIN_REGULARIZAR, X_ent, y_ent, X_val, y_val,
                             X_pru, y_pru)
    print(f"{'n':>7}{'entrena':>10}{'valida':>10}{'prueba':>10}{'brecha':>10}")
    for f in ca_a:
        print(f"{f['n']:>7}{f['entrenamiento']:>10.4f}{f['validacion']:>10.4f}"
              f"{f['prueba']:>10.4f}{f['entrenamiento'] - f['validacion']:>+10.4f}")

    print()
    print("Regularizado:")
    ca_b = curva_aprendizaje(REGULARIZADO, X_ent, y_ent, X_val, y_val,
                             X_pru, y_pru)
    print(f"{'n':>7}{'entrena':>10}{'valida':>10}{'prueba':>10}{'brecha':>10}")
    for f in ca_b:
        print(f"{f['n']:>7}{f['entrenamiento']:>10.4f}{f['validacion']:>10.4f}"
              f"{f['prueba']:>10.4f}{f['entrenamiento'] - f['validacion']:>+10.4f}")

    salida["curva_aprendizaje_sin_regularizar"] = ca_a
    salida["curva_aprendizaje_regularizado"] = ca_b

    # ------------------------------------------ 7. evaluacion final en prueba
    titulo("7. EVALUACION FINAL EN EL CONJUNTO DE PRUEBA")
    print("Primera y unica vez que se mide la prueba para reportar.")
    print()
    m_pru_a = metricas(modelo_a, X_pru, y_pru)
    m_pru_b = metricas(modelo_b, X_pru, y_pru)

    print(f"{'metrica':<16}{'sin regularizar':>18}{'regularizado':>16}{'cambio':>12}")
    print("-" * 62)
    for clave in ["exactitud", "precision", "sensibilidad", "f1", "roc_auc"]:
        d = m_pru_b[clave] - m_pru_a[clave]
        print(f"{clave:<16}{m_pru_a[clave]:>18.4f}{m_pru_b[clave]:>16.4f}"
              f"{d:>+12.4f}")

    diag_a["prueba"] = m_pru_a
    diag_b["prueba"] = m_pru_b
    diag_a["roc"] = puntos_roc(modelo_a, X_pru, y_pru)
    diag_b["roc"] = puntos_roc(modelo_b, X_pru, y_pru)

    salida["sin_regularizar"] = diag_a
    salida["regularizado"] = diag_b

    # ------------------------------------------------------- 8. comparacion
    titulo("8. RESUMEN DE LA MEJORA")
    print(f"{'':<24}{'sin regular.':>14}{'regularizado':>14}{'cambio':>12}")
    print("-" * 64)
    filas_resumen = [
        ("Exactitud entrenamiento", diag_a["entrenamiento"]["exactitud"],
         diag_b["entrenamiento"]["exactitud"]),
        ("Exactitud validacion", diag_a["validacion"]["exactitud"],
         diag_b["validacion"]["exactitud"]),
        ("Exactitud prueba", m_pru_a["exactitud"], m_pru_b["exactitud"]),
        ("Brecha entrena-valida", diag_a["varianza"]["brecha"],
         diag_b["varianza"]["brecha"]),
        ("ROC-AUC prueba", m_pru_a["roc_auc"], m_pru_b["roc_auc"]),
        ("Hojas por arbol", diag_a["hojas_promedio"], diag_b["hojas_promedio"]),
    ]
    for nombre, a, b in filas_resumen:
        print(f"{nombre:<24}{a:>14.4f}{b:>14.4f}{b - a:>+12.4f}")

    print()
    print(f"Sesgo:    {diag_a['sesgo']['nivel']:6} -> {diag_b['sesgo']['nivel']}")
    print(f"Varianza: {diag_a['varianza']['nivel']:6} -> {diag_b['varianza']['nivel']}")
    print(f"Ajuste:   {diag_a['ajuste']['nivel']} -> {diag_b['ajuste']['nivel']}")

    salida["error_irreducible_estimado"] = ERROR_IRREDUCIBLE
    salida["segundos"] = round(time.perf_counter() - inicio, 1)

    ruta_json = CARPETA / "resultados_analisis.json"
    ruta_json.write_text(json.dumps(salida, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print()
    print(f"Resultados guardados en {ruta_json.name}")
    print(f"Tiempo total: {salida['segundos']} segundos")


if __name__ == "__main__":
    main()
