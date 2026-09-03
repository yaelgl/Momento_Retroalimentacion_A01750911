"""
Arbol de decision para clasificacion, implementado desde cero.

Sigue la idea del algoritmo CART (Classification And Regression Trees). No usa
scikit-learn ni ninguna otra libreria de aprendizaje automatico: solo `math` y
`collections` de la libreria estandar.

Como funciona, en resumen
-------------------------
La pregunta que responde el arbol es "que division de los datos deja los grupos
resultantes lo mas puros posible". Puro significa que casi todos los ejemplos de
un grupo pertenecen a la misma clase.

1. Se mide la impureza del nodo actual con el indice de Gini:

       Gini = 1 - sum(p_k^2)

   donde p_k es la proporcion de la clase k en el nodo. Vale 0 cuando todos los
   ejemplos son de la misma clase (nodo puro) y 0.5 cuando estan mitad y mitad
   en un problema de dos clases (lo peor posible).

2. Se prueban todas las divisiones posibles. Para una variable numerica se
   prueban umbrales del tipo "Age <= 27"; para una categorica, preguntas del
   tipo "HomePlanet == Europa".

3. Para cada division se calcula la ganancia:

       ganancia = Gini(padre) - [ (n_izq/n) * Gini(izq) + (n_der/n) * Gini(der) ]

   Es decir, cuanta impureza se elimina al partir. Se elige la division con la
   ganancia mas alta.

4. Se repite el proceso de forma recursiva en cada rama hasta que se cumple
   alguna condicion de paro. Ese nodo se vuelve una hoja y predice la clase
   mayoritaria de los ejemplos que le llegaron.

Las condiciones de paro son las que evitan el sobreajuste: sin ellas el arbol
crece hasta memorizar cada ejemplo del entrenamiento y no generaliza.
"""

import math
from collections import Counter


# ---------------------------------------------------------------------------
# Medidas de impureza
# ---------------------------------------------------------------------------

def gini(conteos, total):
    """Indice de Gini a partir de un diccionario {clase: cantidad}.

    Ejemplo: 50 de una clase y 50 de otra -> 1 - (0.5^2 + 0.5^2) = 0.5 (maxima
    impureza). 100 de una sola clase -> 1 - 1^2 = 0 (nodo puro).
    """
    if total == 0:
        return 0.0
    suma = 0.0
    for cantidad in conteos.values():
        p = cantidad / total
        suma += p * p
    return 1.0 - suma


def entropia(conteos, total):
    """Entropia de Shannon. Alternativa al Gini, se incluye para comparar.

    Mide lo mismo (que tan mezcladas estan las clases) pero en bits. Da arboles
    muy parecidos; el Gini es un poco mas rapido porque no calcula logaritmos.
    """
    if total == 0:
        return 0.0
    suma = 0.0
    for cantidad in conteos.values():
        if cantidad > 0:
            p = cantidad / total
            suma -= p * math.log2(p)
    return suma


# ---------------------------------------------------------------------------
# Estructura del arbol
# ---------------------------------------------------------------------------

class Nodo:
    """Un nodo del arbol. Puede ser una hoja o una pregunta.

    Si es hoja: guarda la clase que predice y con que probabilidad.
    Si es pregunta: guarda la variable por la que divide, el valor de corte y
    sus dos hijos.
    """

    def __init__(self, conteos, profundidad):
        self.conteos = conteos              # {clase: cantidad} de este nodo
        self.n = sum(conteos.values())      # ejemplos que llegaron aqui
        self.profundidad = profundidad

        # La clase mayoritaria es lo que predeciria este nodo si fuera hoja
        self.clase = max(conteos, key=lambda c: (conteos[c], -c)) if conteos else 0
        self.probabilidad = conteos.get(1, 0) / self.n if self.n else 0.0

        # Se llenan solo si el nodo termina siendo una pregunta
        self.es_hoja = True
        self.columna = None
        self.tipo = None          # "numerica" o "categorica"
        self.corte = None         # umbral si es numerica, categoria si no
        self.izquierda = None
        self.derecha = None
        self.ganancia = 0.0

    def describir_pregunta(self):
        """Texto de la condicion, para poder imprimir el arbol."""
        if self.tipo == "numerica":
            return f"{self.columna} <= {self.corte:g}"
        return f"{self.columna} == {self.corte}"


# ---------------------------------------------------------------------------
# El clasificador
# ---------------------------------------------------------------------------

class ArbolDecision:
    """Clasificador de arbol de decision binario.

    Parametros
    ----------
    max_profundidad : int
        Cuantos niveles de preguntas puede tener el arbol como maximo. Es el
        control principal contra el sobreajuste: mientras mas profundo, mas se
        ajusta al entrenamiento y peor generaliza.
    min_muestras_division : int
        Un nodo con menos ejemplos que esto ya no se divide. Evita partir grupos
        tan chicos que la division sea puro ruido.
    min_muestras_hoja : int
        Cada hoja debe quedar con al menos estos ejemplos. Una division que deje
        una hoja con 2 ejemplos se descarta aunque tenga buena ganancia.
    min_ganancia : float
        Ganancia minima de impureza para aceptar una division. Si la mejor
        division apenas mejora, no vale la pena y el nodo se vuelve hoja.
    criterio : str
        "gini" (predeterminado) o "entropia".
    """

    def __init__(self, columnas_numericas, columnas_categoricas,
                 max_profundidad=8, min_muestras_division=20,
                 min_muestras_hoja=10, min_ganancia=0.0, criterio="gini"):
        self.numericas = list(columnas_numericas)
        self.categoricas = list(columnas_categoricas)
        self.columnas = self.numericas + self.categoricas
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.min_muestras_hoja = min_muestras_hoja
        self.min_ganancia = min_ganancia

        if criterio not in ("gini", "entropia"):
            raise ValueError("criterio debe ser 'gini' o 'entropia'")
        self.criterio = criterio
        self._impureza = gini if criterio == "gini" else entropia

        self.raiz = None
        self.importancias = {}

    # -- Entrenamiento ------------------------------------------------------

    def entrenar(self, X, y):
        """Construye el arbol a partir de los datos de entrenamiento.

        X es una lista de diccionarios {nombre_columna: valor} y y una lista de
        etiquetas 0/1 del mismo largo.
        """
        if len(X) != len(y):
            raise ValueError("X y y deben tener el mismo numero de filas")
        if not X:
            raise ValueError("no hay datos para entrenar")

        self._X = X
        self._y = y
        self.importancias = {col: 0.0 for col in self.columnas}

        # Se trabaja con indices en lugar de copiar sublistas de datos: es mas
        # rapido y usa menos memoria
        indices = list(range(len(X)))
        self.raiz = self._construir(indices, profundidad=0)

        # La importancia se normaliza para que sume 1 y se lea como porcentaje
        total = sum(self.importancias.values())
        if total > 0:
            self.importancias = {k: v / total for k, v in self.importancias.items()}

        # Ya no se necesita guardar referencia a los datos de entrenamiento
        del self._X
        del self._y
        return self

    def _construir(self, indices, profundidad):
        """Crea un nodo de forma recursiva."""
        conteos = Counter(self._y[i] for i in indices)
        nodo = Nodo(conteos, profundidad)

        # ¿Se debe dejar de dividir? Tres razones para parar aqui mismo
        if profundidad >= self.max_profundidad:
            return nodo
        if nodo.n < self.min_muestras_division:
            return nodo
        if len(conteos) == 1:
            return nodo              # el nodo ya es puro, no hay nada que ganar

        # Buscar la mejor division entre todas las variables
        mejor = self._mejor_division(indices, conteos, nodo.n)
        if mejor is None:
            return nodo              # no existe ninguna division posible

        # Se acepta la division si la ganancia alcanza el minimo pedido. Con el
        # valor por omision (0) se aceptan divisiones de ganancia CERO, y eso es
        # a proposito.
        #
        # El caso clasico es una relacion tipo XOR: y = 1 si a y b son distintos.
        # Ahi ninguna de las dos variables por separado reduce la impureza, asi
        # que la ganancia en la raiz es exactamente 0. Si se exigiera ganancia
        # positiva el arbol se detendria de inmediato y nunca aprenderia la
        # regla, aunque con dos niveles la resuelve sin problema. Al permitir la
        # division de ganancia cero, el siguiente nivel si encuentra cortes
        # utiles. Es el mismo criterio que usa scikit-learn.
        #
        # La tolerancia evita que una ganancia de -1e-17 por redondeo de punto
        # flotante se interprete como negativa.
        if mejor["ganancia"] < self.min_ganancia - 1e-12:
            return nodo              # ninguna division vale la pena

        # Convertir el nodo en pregunta y seguir con las dos ramas
        nodo.es_hoja = False
        nodo.columna = mejor["columna"]
        nodo.tipo = mejor["tipo"]
        nodo.corte = mejor["corte"]
        nodo.ganancia = mejor["ganancia"]

        # La importancia de una variable es cuanta impureza total elimina, pesada
        # por cuantos ejemplos pasan por ese nodo. Una division muy buena pero en
        # un nodo de 20 ejemplos importa menos que una decente en la raiz.
        self.importancias[mejor["columna"]] += mejor["ganancia"] * nodo.n

        nodo.izquierda = self._construir(mejor["izquierda"], profundidad + 1)
        nodo.derecha = self._construir(mejor["derecha"], profundidad + 1)
        return nodo

    def _mejor_division(self, indices, conteos_padre, n_padre):
        """Prueba todas las divisiones posibles y devuelve la de mayor ganancia."""
        impureza_padre = self._impureza(conteos_padre, n_padre)
        mejor = None

        for columna in self.numericas:
            candidata = self._mejor_division_numerica(
                indices, columna, conteos_padre, n_padre, impureza_padre)
            if candidata and (mejor is None or candidata["ganancia"] > mejor["ganancia"]):
                mejor = candidata

        for columna in self.categoricas:
            candidata = self._mejor_division_categorica(
                indices, columna, conteos_padre, n_padre, impureza_padre)
            if candidata and (mejor is None or candidata["ganancia"] > mejor["ganancia"]):
                mejor = candidata

        return mejor

    def _mejor_division_numerica(self, indices, columna, conteos_padre,
                                 n_padre, impureza_padre):
        """Mejor umbral para una variable numerica.

        Se ordenan los ejemplos por el valor de la variable y se recorre una sola
        vez, moviendo ejemplos del lado derecho al izquierdo. Asi se evaluan
        todos los umbrales posibles sin recalcular los conteos desde cero cada
        vez, que seria mucho mas lento.

        Los umbrales candidatos son los puntos medios entre valores consecutivos
        distintos. Usar el punto medio y no el valor exacto hace que el arbol
        generalice un poco mejor a datos nuevos.
        """
        pares = sorted((self._X[i][columna], self._y[i]) for i in indices)

        # Empezamos con todo del lado derecho
        izq = Counter()
        der = Counter(conteos_padre)
        n_izq, n_der = 0, n_padre
        mejor = None

        for k in range(len(pares) - 1):
            valor, etiqueta = pares[k]
            izq[etiqueta] += 1
            der[etiqueta] -= 1
            n_izq += 1
            n_der -= 1

            # Solo tiene sentido cortar donde el valor cambia: si el siguiente
            # valor es igual, no se puede separar entre ellos
            if pares[k + 1][0] == valor:
                continue
            if n_izq < self.min_muestras_hoja or n_der < self.min_muestras_hoja:
                continue

            ganancia = impureza_padre - (
                (n_izq / n_padre) * self._impureza(izq, n_izq)
                + (n_der / n_padre) * self._impureza(der, n_der))

            if mejor is None or ganancia > mejor["ganancia"]:
                mejor = {"ganancia": ganancia,
                         "corte": (valor + pares[k + 1][0]) / 2.0}

        if mejor is None:
            return None

        # Se reconstruyen las listas de indices de cada rama
        umbral = mejor["corte"]
        izquierda = [i for i in indices if self._X[i][columna] <= umbral]
        derecha = [i for i in indices if self._X[i][columna] > umbral]
        return {"columna": columna, "tipo": "numerica", "corte": umbral,
                "ganancia": mejor["ganancia"],
                "izquierda": izquierda, "derecha": derecha}

    def _mejor_division_categorica(self, indices, columna, conteos_padre,
                                   n_padre, impureza_padre):
        """Mejor division para una variable categorica.

        Se usan divisiones binarias del tipo "es igual a esta categoria" contra
        "es cualquier otra". La alternativa seria probar todos los subconjuntos
        posibles de categorias, pero eso crece como 2^k y aqui no hace falta:
        con divisiones binarias sucesivas el arbol puede reconstruir cualquier
        agrupacion en niveles siguientes.
        """
        # Conteo de clases por categoria, en una sola pasada
        por_categoria = {}
        for i in indices:
            valor = self._X[i][columna]
            if valor not in por_categoria:
                por_categoria[valor] = Counter()
            por_categoria[valor][self._y[i]] += 1

        if len(por_categoria) < 2:
            return None              # una sola categoria: no hay como dividir

        mejor = None
        for categoria, conteos_cat in por_categoria.items():
            n_izq = sum(conteos_cat.values())
            n_der = n_padre - n_izq
            if n_izq < self.min_muestras_hoja or n_der < self.min_muestras_hoja:
                continue

            # El lado derecho es el complemento: padre menos la categoria
            conteos_der = Counter(conteos_padre)
            conteos_der.subtract(conteos_cat)

            ganancia = impureza_padre - (
                (n_izq / n_padre) * self._impureza(conteos_cat, n_izq)
                + (n_der / n_padre) * self._impureza(conteos_der, n_der))

            if mejor is None or ganancia > mejor["ganancia"]:
                mejor = {"ganancia": ganancia, "corte": categoria}

        if mejor is None:
            return None

        categoria = mejor["corte"]
        izquierda = [i for i in indices if self._X[i][columna] == categoria]
        derecha = [i for i in indices if self._X[i][columna] != categoria]
        return {"columna": columna, "tipo": "categorica", "corte": categoria,
                "ganancia": mejor["ganancia"],
                "izquierda": izquierda, "derecha": derecha}

    # -- Prediccion ---------------------------------------------------------

    def _recorrer(self, fila):
        """Baja por el arbol siguiendo las respuestas hasta llegar a una hoja."""
        nodo = self.raiz
        while not nodo.es_hoja:
            valor = fila.get(nodo.columna)

            if valor is None:
                # No deberia pasar porque los datos se imputan antes, pero por
                # seguridad se manda por la rama con mas ejemplos de entrenamiento
                nodo = (nodo.izquierda if nodo.izquierda.n >= nodo.derecha.n
                        else nodo.derecha)
                continue

            if nodo.tipo == "numerica":
                nodo = nodo.izquierda if valor <= nodo.corte else nodo.derecha
            else:
                # Si llega una categoria que no se vio al entrenar, cae en el
                # lado del "cualquier otra", que es el comportamiento correcto
                nodo = nodo.izquierda if valor == nodo.corte else nodo.derecha
        return nodo

    def predecir_uno(self, fila):
        """Clase predicha (0 o 1) para una sola fila."""
        return self._recorrer(fila).clase

    def predecir(self, X):
        """Clases predichas para una lista de filas."""
        return [self.predecir_uno(fila) for fila in X]

    def predecir_probabilidad(self, X):
        """Probabilidad estimada de la clase 1 para cada fila.

        Es la proporcion de ejemplos de clase 1 que cayeron en esa hoja durante
        el entrenamiento.
        """
        return [self._recorrer(fila).probabilidad for fila in X]

    # -- Informacion sobre el arbol -----------------------------------------

    def profundidad_alcanzada(self):
        """Nivel del nodo mas profundo. Puede ser menor que max_profundidad."""
        def bajar(nodo):
            if nodo.es_hoja:
                return nodo.profundidad
            return max(bajar(nodo.izquierda), bajar(nodo.derecha))
        return bajar(self.raiz)

    def contar_nodos(self):
        """Devuelve (nodos_totales, hojas)."""
        def contar(nodo):
            if nodo.es_hoja:
                return 1, 1
            n_izq, h_izq = contar(nodo.izquierda)
            n_der, h_der = contar(nodo.derecha)
            return n_izq + n_der + 1, h_izq + h_der
        return contar(self.raiz)

    def importancia_ordenada(self):
        """Variables ordenadas de mas a menos importante."""
        return sorted(self.importancias.items(), key=lambda par: par[1], reverse=True)

    def a_texto(self, max_profundidad=None, nombres_clase=("No", "Si")):
        """Dibuja el arbol como texto indentado.

        Sirve para revisar que las reglas aprendidas tengan sentido, que es una
        de las ventajas de los arboles sobre otros modelos: se pueden leer.
        """
        lineas = []

        def escribir(nodo, prefijo, etiqueta_rama):
            if max_profundidad is not None and nodo.profundidad > max_profundidad:
                return
            if nodo.es_hoja:
                lineas.append(f"{prefijo}{etiqueta_rama}[hoja] predice "
                              f"{nombres_clase[nodo.clase]} "
                              f"(n={nodo.n}, P(Si)={nodo.probabilidad:.2f})")
                return

            # Si se corta el dibujo por profundidad, se avisa que hay mas abajo
            if max_profundidad is not None and nodo.profundidad == max_profundidad:
                lineas.append(f"{prefijo}{etiqueta_rama}{nodo.describir_pregunta()} "
                              f"(n={nodo.n}) [...]")
                return

            lineas.append(f"{prefijo}{etiqueta_rama}{nodo.describir_pregunta()} "
                          f"(n={nodo.n}, ganancia={nodo.ganancia:.4f})")
            sangria = prefijo + ("    " if etiqueta_rama else "")
            escribir(nodo.izquierda, sangria, "|-- si:  ")
            escribir(nodo.derecha, sangria, "|-- no:  ")

        escribir(self.raiz, "", "")
        return "\n".join(lineas)
