"""
Carga del dataset y transformadores de scikit-learn para el preprocesamiento.

A diferencia de la Parte I del modulo (donde todo se programo a mano), aqui se
aprovecha el framework. La idea es que TODO el preprocesamiento viva dentro de un
Pipeline de scikit-learn, no en pasos suertos antes de entrenar.

Por que eso importa: si uno imputa y codifica "a mano" sobre el dataset completo
antes de partirlo, los estadisticos del relleno se calculan mezclando
entrenamiento y prueba, y el modelo termina usando informacion que no deberia
conocer. Eso es fuga de informacion y hace que las metricas salgan mejores de lo
que son. Metiendo el preprocesamiento en un Pipeline, scikit-learn se encarga de
ajustarlo SOLO con los datos de entrenamiento en cada llamada a fit, y eso vale
tambien dentro de cada pliegue de la validacion cruzada.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
OBJETIVO = "Transported"
SEMILLA = 42

# Columnas que entran al modelo, ya contando las derivadas que crea el
# transformador de abajo. Se separan por tipo porque cada grupo recibe un
# tratamiento distinto en el ColumnTransformer.
NUMERICAS = ["Age", "TamGrupo", "GastoTotal"] + GASTOS
CATEGORICAS = ["HomePlanet", "CryoSleep", "Destination", "Deck", "Side",
               "ViajaSolo", "EsNino"]


# ---------------------------------------------------------------------------
# Transformador propio para la ingenieria de variables
# ---------------------------------------------------------------------------

class DerivarVariables(BaseEstimator, TransformerMixin):
    """Crea variables nuevas a partir de Cabin, PassengerId, Age y los gastos.

    Se implementa como un transformador de scikit-learn (heredando de
    BaseEstimator y TransformerMixin) en lugar de como una funcion suelta. Asi se
    puede meter dentro de un Pipeline y se comporta como cualquier otro paso del
    framework: tiene fit y transform, y GridSearchCV lo respeta al hacer
    validacion cruzada.

    Variables que genera:
        Deck, Side       partiendo Cabin, que viene con formato "B/0/P"
        TamGrupo         cuantos pasajeros comparten el grupo de PassengerId
        ViajaSolo        si TamGrupo es 1
        GastoTotal       suma de los cinco consumos a bordo
        EsNino           si tiene 12 anios o menos
    """

    def fit(self, X, y=None):
        # No aprende nada de los datos: solo descompone columnas. Aun asi hace
        # falta definir fit para cumplir con la interfaz de scikit-learn.
        return self

    def transform(self, X):
        X = X.copy()

        # Cabin -> Deck y Side. El numero de cabina de en medio se ignora porque
        # en el analisis exploratorio no mostro relacion con el objetivo.
        partes = X["Cabin"].str.split("/", expand=True)
        X["Deck"] = partes[0]
        X["Side"] = partes[2]

        # La cubierta T tiene solo 5 pasajeros en todo el dataset: se junta con A
        # para no dejar una categoria casi vacia.
        X["Deck"] = X["Deck"].replace({"T": "A"})

        # PassengerId tiene formato "0001_01": los primeros digitos son el grupo
        grupo = X["PassengerId"].str.split("_").str[0]
        X["TamGrupo"] = grupo.map(grupo.value_counts())
        X["ViajaSolo"] = (X["TamGrupo"] == 1).map({True: "Si", False: "No"})

        # min_count=1 deja NaN solo si TODOS los gastos faltan
        X["GastoTotal"] = X[GASTOS].sum(axis=1, min_count=1)

        # EsNino se deja como categorica (con su propia categoria para los nulos)
        # para no tener que inventar una edad antes de imputar
        X["EsNino"] = np.where(X["Age"].isna(), None,
                               np.where(X["Age"] <= 12, "Si", "No"))

        # CryoSleep es booleana con nulos; como texto el imputador de categoricas
        # la maneja sin problema
        X["CryoSleep"] = X["CryoSleep"].astype("object").astype(str)
        X.loc[X["CryoSleep"].isin(["nan", "None", "<NA>"]), "CryoSleep"] = None

        return X[NUMERICAS + CATEGORICAS]


# ---------------------------------------------------------------------------
# Preprocesamiento con ColumnTransformer
# ---------------------------------------------------------------------------

def construir_preprocesador():
    """Arma el ColumnTransformer que prepara las columnas para el modelo.

    Numericas:
        SimpleImputer con la mediana. Se usa mediana y no media porque los gastos
        estan muy sesgados a la derecha (mas del 60% son cero y hay valores de
        miles), y la media quedaria arrastrada por esa cola.

    Categoricas:
        SimpleImputer con la constante "Desconocido" y despues OneHotEncoder.
        Se rellena con una categoria propia en lugar de la moda porque en este
        dataset la ausencia puede ser informativa, y asi el modelo puede
        aprender de ella en vez de que se disfrace como el valor mas comun.

        handle_unknown="ignore" es importante: si al predecir aparece una
        categoria que no estaba al entrenar, en lugar de tronar la codifica como
        ceros en todas las columnas de esa variable.

    No se escala nada. Los arboles y los bosques parten los datos con umbrales
    del tipo "Age <= 27", y esa comparacion no cambia si la columna se multiplica
    por una constante, asi que escalar no aportaria nada.
    """
    tuberia_numerica = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
    ])

    tuberia_categorica = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", tuberia_numerica, NUMERICAS),
            ("cat", tuberia_categorica, CATEGORICAS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def cargar_entrenamiento(ruta_csv):
    """Lee train.csv y devuelve (X, y) sin transformar todavia.

    El preprocesamiento NO se aplica aqui a proposito: se aplica dentro del
    Pipeline, cuando ya se hizo la particion. Devolver los datos crudos es lo que
    permite que no haya fuga de informacion.
    """
    datos = pd.read_csv(ruta_csv)
    y = datos[OBJETIVO].astype(int)
    X = datos.drop(columns=[OBJETIVO])
    return X, y


def cargar_sin_etiqueta(ruta_csv):
    """Lee el test.csv de Kaggle, que no trae la columna objetivo.

    Devuelve (identificadores, X) para poder armar un archivo de predicciones.
    """
    datos = pd.read_csv(ruta_csv)
    return datos["PassengerId"].tolist(), datos


# Revision rapida si se ejecuta este archivo directamente
if __name__ == "__main__":
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    ruta = Path(__file__).parent / "dataset" / "train.csv"
    X, y = cargar_entrenamiento(ruta)
    print(f"Filas leidas: {len(X)}   columnas originales: {X.shape[1]}")
    print(f"Balance del objetivo: {y.mean() * 100:.2f}% transportados")
    print()

    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEMILLA)
    print(f"Entrenamiento: {len(X_ent)}   Prueba: {len(X_pru)}")

    # Se ajusta con entrenamiento y se aplica a prueba, como debe ser
    preparador = Pipeline([("derivar", DerivarVariables()),
                           ("preprocesar", construir_preprocesador())])
    Z_ent = preparador.fit_transform(X_ent)
    Z_pru = preparador.transform(X_pru)

    nombres = preparador.named_steps["preprocesar"].get_feature_names_out()
    print(f"\nColumnas despues del preprocesamiento: {Z_ent.shape[1]}")
    print(f"Nulos restantes: {int(np.isnan(Z_ent).sum())}")
    print()
    print("Nombres generados:")
    for i, nombre in enumerate(nombres):
        print(f"  {i:2d}. {nombre}")
