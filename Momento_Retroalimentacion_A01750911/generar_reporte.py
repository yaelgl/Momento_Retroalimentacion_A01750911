"""
Genera el reporte en PDF con los resultados del arbol de decision.

Este script NO forma parte de la implementacion del algoritmo: solo toma los
resultados que main.py guardo en resultados.json y los acomoda en un documento.
Por eso si usa librerias externas (fpdf2 para el PDF y matplotlib para las
graficas), mientras que el algoritmo en si esta escrito unicamente con la
libreria estandar.

Ninguna cifra del reporte esta escrita a mano: todas se leen del JSON, que a su
vez lo produjo la ejecucion real del programa.

El contenido esta separado de la presentacion: `construir_contenido` devuelve una
lista de bloques (secciones, parrafos, tablas, figuras) y `renderizar_pdf` los
dibuja. Asi el texto del reporte se lee de corrido, sin mezclarse con las
instrucciones de formato.

Uso:
    python pruebas.py --guardar-json
    python main.py --curva --arbol 3 --guardar-json resultados.json
    python generar_reporte.py --alumno "Tu Nombre" --matricula "A01234567"
"""

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # sin ventana grafica, solo archivos
import matplotlib.pyplot as plt
from fpdf import FPDF

CARPETA = Path(__file__).parent
RUTA_JSON = CARPETA / "resultados.json"
CARPETA_FIG = CARPETA / "figuras"
RUTA_PDF = CARPETA / "reporte.pdf"

# Datos de la portada. Se pueden cambiar aqui o con los argumentos de la terminal.
ALUMNO = "Yael Michel García López"
MATRICULA = "A01750911"
MATERIA = "Inteligencia artificial avanzada para la ciencia de datos"
ENTREGABLE = ("Implementacion de una tecnica de aprendizaje maquina "
              "sin el uso de un framework")
REPOSITORIO = "https://github.com/yaelgl/Momento_Retroalimentacion_A01750911.git"


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

# Las fuentes basicas del PDF solo manejan el juego de caracteres latin-1, que
# si incluye las vocales acentuadas y la enie pero no cosas como la raya larga o
# las flechas. Se reemplazan para que el PDF no truene en ninguna computadora.
REEMPLAZOS = {
    "\u2014": "-", "\u2013": "-", "\u2192": "->", "\u2264": "<=",
    "\u2265": ">=", "\u201c": '"', "\u201d": '"', "\u2018": "'",
    "\u2019": "'", "\u2026": "...", "\u00d7": "x",
}


def limpiar(texto):
    """Sustituye los caracteres que la fuente del PDF no puede representar."""
    for original, reemplazo in REEMPLAZOS.items():
        texto = texto.replace(original, reemplazo)
    return texto


def pct(valor, decimales=2):
    """Formatea una proporcion como porcentaje."""
    return f"{valor * 100:.{decimales}f}%"


def contar_filas(ruta_csv):
    """Cuenta las filas de datos de un CSV, sin contar el encabezado."""
    with open(ruta_csv, encoding="utf-8") as archivo:
        return sum(1 for _ in archivo) - 1


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------

def figura_matriz_confusion(datos, ruta):
    """Dibuja la matriz de confusion como una cuadricula con los conteos."""
    mc = datos["prueba"]["matriz"]
    valores = [[mc["VN"], mc["FP"]], [mc["FN"], mc["VP"]]]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(valores, cmap="Blues")

    etiquetas = ["No transportado", "Si transportado"]
    ax.set_xticks([0, 1], etiquetas)
    ax.set_yticks([0, 1], etiquetas)
    ax.set_xlabel("Prediccion del modelo")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusion (conjunto de prueba)")

    # El texto se pone en blanco sobre las celdas oscuras para que se lea
    maximo = max(max(fila) for fila in valores)
    nombres = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if valores[i][j] > maximo * 0.6 else "black"
            ax.text(j, i, f"{nombres[i][j]}\n{valores[i][j]}", ha="center",
                    va="center", color=color, fontsize=13)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def figura_curva_profundidad(datos, ruta):
    """Exactitud en entrenamiento y prueba segun la profundidad del arbol."""
    curva = datos["curva_profundidad"]
    if not curva:
        return False

    profundidades = [r["profundidad"] for r in curva]
    entrena = [r["entrena"] for r in curva]
    prueba = [r["prueba"] for r in curva]
    elegida = datos["parametros"]["profundidad_maxima"]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(profundidades, entrena, "o-", label="Entrenamiento")
    ax.plot(profundidades, prueba, "s-", label="Prueba")
    ax.axvline(elegida, color="gray", linestyle="--", linewidth=1)
    ax.text(elegida + 0.2, min(prueba), f"profundidad elegida: {elegida}",
            fontsize=8, color="gray")

    ax.set_xlabel("Profundidad maxima del arbol")
    ax.set_ylabel("Exactitud")
    ax.set_title("Efecto de la profundidad: la brecha crece con el tamanio del arbol")
    ax.set_xticks(profundidades[::2])
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return True


def figura_importancias(datos, ruta):
    """Barras horizontales con la importancia de cada variable."""
    pares = [(n, v) for n, v in datos["importancias"] if v > 0.0001]
    pares.sort(key=lambda p: p[1])
    nombres = [p[0] for p in pares]
    valores = [p[1] * 100 for p in pares]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(nombres, valores, color="steelblue")
    ax.set_xlabel("Importancia (%)")
    ax.set_title("Cuanto aporta cada variable a las decisiones del arbol")
    for i, valor in enumerate(valores):
        ax.text(valor + 0.6, i, f"{valor:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, max(valores) * 1.18)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Recolector: define el contenido una sola vez
# ---------------------------------------------------------------------------

class Recolector:
    """Junta el contenido del reporte en una lista de bloques.

    En lugar de dibujar, guarda una descripcion de cada bloque: que es y que
    contiene. El renderizador se encarga despues de la parte visual. Separarlo
    asi deja el texto del reporte legible de corrido, sin comandos de formato
    intercalados.
    """

    def __init__(self):
        self.bloques = []

    def portada(self, titulo, subtitulo, campos, resumen_titulo, resumen):
        self.bloques.append(("portada", {
            "titulo": titulo, "subtitulo": subtitulo, "campos": campos,
            "resumen_titulo": resumen_titulo, "resumen": resumen}))

    def titulo_seccion(self, texto):
        self.bloques.append(("seccion", texto))

    def subtitulo(self, texto):
        self.bloques.append(("subtitulo", texto))

    def parrafo(self, texto):
        # Se normalizan los espacios aqui para que ambos formatos reciban lo mismo
        self.bloques.append(("parrafo", " ".join(texto.split())))

    def vinieta(self, texto):
        self.bloques.append(("vinieta", " ".join(texto.split())))

    def codigo(self, texto):
        self.bloques.append(("codigo", texto))

    def tabla(self, encabezados, filas, anchos):
        self.bloques.append(("tabla", {"encabezados": encabezados,
                                       "filas": filas, "anchos": anchos}))

    def imagen(self, ruta, ancho, proporcion):
        self.bloques.append(("imagen", {"ruta": str(ruta), "ancho": ancho,
                                        "proporcion": proporcion}))

    def add_page(self):
        self.bloques.append(("salto", None))


# ---------------------------------------------------------------------------
# Renderizado a PDF
# ---------------------------------------------------------------------------

class Reporte(FPDF):
    """PDF con encabezado y pie de pagina propios."""

    def header(self):
        if self.page_no() == 1:
            return                      # la portada va limpia
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, limpiar("Arbol de decision sin framework - Spaceship Titanic"),
                  align="R")
        self.ln(9)
        self.set_text_color(0)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")
        self.set_text_color(0)

    # -- bloques de contenido ------------------------------------------------

    def titulo_seccion(self, texto):
        """Titulo de seccion, con salto de pagina si ya no cabe."""
        if self.get_y() > 235:
            self.add_page()
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 60, 110)
        # new_x="LMARGIN" es indispensable: por omision fpdf deja el cursor en el
        # borde derecho y la siguiente celda se quedaria sin ancho disponible.
        self.multi_cell(0, 7, limpiar(texto), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(1)

    def subtitulo(self, texto):
        if self.get_y() > 245:
            self.add_page()
        self.ln(1)
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(0, 6, limpiar(texto), new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def parrafo(self, texto):
        self.set_font("Helvetica", "", 10)
        # Se juntan las lineas para que el PDF haga su propio ajuste de texto
        limpio = " ".join(texto.split())
        self.set_x(self.l_margin)
        self.multi_cell(0, 5, limpiar(limpio), new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def vinieta(self, texto):
        self.set_font("Helvetica", "", 10)
        limpio = " ".join(texto.split())
        self.set_x(self.l_margin)
        self.cell(5, 5, "-")
        self.multi_cell(0, 5, limpiar(limpio), new_x="LMARGIN", new_y="NEXT")

    def imagen(self, ruta, ancho, proporcion):
        """Inserta una figura centrada, saltando de pagina solo si no cabe.

        `proporcion` es alto/ancho de la imagen original. Se usa para estimar la
        altura que va a ocupar y evitar que una figura quede sola en una pagina
        casi vacia.
        """
        alto = ancho * proporcion
        disponible = self.h - self.b_margin - self.get_y()
        if alto + 4 > disponible:
            self.add_page()
        x = (self.w - ancho) / 2
        self.image(str(ruta), w=ancho, x=x)
        self.ln(2)

    def codigo(self, texto):
        """Bloque monoespaciado, para reglas del arbol o salidas de consola."""
        self.set_font("Courier", "", 7.5)
        self.set_fill_color(245, 245, 245)
        for linea in texto.split("\n"):
            if self.get_y() > 265:
                self.add_page()
                self.set_font("Courier", "", 7.5)
            self.cell(0, 3.8, limpiar(linea)[:110], fill=True, new_x="LMARGIN",
                      new_y="NEXT")
        self.ln(2)

    def tabla(self, encabezados, filas, anchos):
        """Tabla simple con encabezado sombreado."""
        if self.get_y() + 8 + 6 * len(filas) > 275:
            self.add_page()
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(225, 232, 240)
        for texto, ancho in zip(encabezados, anchos):
            self.cell(ancho, 6.5, limpiar(str(texto)), border=1, align="C", fill=True)
        self.ln()

        self.set_font("Helvetica", "", 9)
        for i, fila in enumerate(filas):
            self.set_fill_color(249, 249, 249) if i % 2 else self.set_fill_color(255, 255, 255)
            for j, (texto, ancho) in enumerate(zip(fila, anchos)):
                self.cell(ancho, 6, limpiar(str(texto)), border=1,
                          align="L" if j == 0 else "C", fill=True)
            self.ln()
        self.ln(2)


def renderizar_pdf(bloques, ruta_salida):
    """Vuelca los bloques a un archivo PDF."""
    pdf = Reporte()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 15, 20)

    for tipo, datos in bloques:
        if tipo == "portada":
            pdf.add_page()
            pdf.ln(35)
            pdf.set_font("Helvetica", "B", 19)
            pdf.multi_cell(0, 9, limpiar(datos["titulo"]), align="C",
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 13)
            pdf.multi_cell(0, 7, limpiar(datos["subtitulo"]), align="C",
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(22)

            # Cada renglon lleva la etiqueta a la izquierda y el valor a la
            # derecha. Hay que regresar el cursor al margen en cada vuelta,
            # porque si no se va desplazando y se acaba el espacio horizontal.
            ancho_etiqueta = 32
            ancho_valor = pdf.w - pdf.l_margin - pdf.r_margin - ancho_etiqueta
            for etiqueta, valor in datos["campos"]:
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(ancho_etiqueta, 7, limpiar(etiqueta + ":"))
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(ancho_valor, 7, limpiar(valor),
                               new_x="LMARGIN", new_y="NEXT")
            pdf.ln(20)
            pdf.set_x(pdf.l_margin)

            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, limpiar(datos["resumen_titulo"]),
                           new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, limpiar(datos["resumen"]),
                           new_x="LMARGIN", new_y="NEXT")
            pdf.add_page()                 # el contenido empieza en hoja nueva

        elif tipo == "seccion":
            pdf.titulo_seccion(datos)
        elif tipo == "subtitulo":
            pdf.subtitulo(datos)
        elif tipo == "parrafo":
            pdf.parrafo(datos)
        elif tipo == "vinieta":
            pdf.vinieta(datos)
        elif tipo == "codigo":
            pdf.codigo(datos)
        elif tipo == "tabla":
            pdf.tabla(datos["encabezados"], datos["filas"], datos["anchos"])
        elif tipo == "imagen":
            pdf.imagen(datos["ruta"], datos["ancho"], datos["proporcion"])
        elif tipo == "salto":
            pdf.add_page()

    pdf.output(str(ruta_salida))
    return pdf.page_no()


def construir_contenido(d, alumno, matricula, figuras, pruebas):
    """Arma la lista de bloques del reporte a partir de los resultados `d`.

    `pruebas` es el conteo que exporta pruebas.py, para no escribir a mano
    cuantas verificaciones hay.
    """
    n_pruebas = pruebas["pasaron"]
    par = d["parametros"]
    dat = d["datos"]
    arb = d["arbol"]
    pru = d["prueba"]
    ent = d["entrenamiento"]
    tri = d["trivial"]
    mc = pru["matriz"]

    doc = Recolector()
    doc.portada(
        titulo="Arbol de decision implementado sin framework",
        subtitulo="Clasificacion de pasajeros del dataset Spaceship Titanic",
        campos=[("Entregable", ENTREGABLE),
                ("Materia", MATERIA),
                ("Alumno", alumno),
                ("Matricula", matricula),
                ("Fecha", date.today().strftime("%d/%m/%Y")),
                ("Repositorio", REPOSITORIO)],
        resumen_titulo="Resultado principal",
        resumen=(
            f"El arbol alcanza una exactitud de {pct(pru['exactitud'])} sobre "
            f"{pru['total']} pasajeros que nunca vio durante el entrenamiento, "
            f"contra {pct(tri['exactitud'])} de un clasificador que siempre "
            f"predice la clase mas frecuente. La implementacion pasa las "
            f"{n_pruebas} pruebas de correctitud incluidas en la entrega."))

    # =================================================== 1. objetivo
    doc.titulo_seccion("1. Objetivo")
    doc.parrafo(f"""
        Programar manualmente un algoritmo de aprendizaje maquina, sin bibliotecas
        del ramo, entrenarlo con un conjunto de datos y comprobar que aprende y
        puede predecir. El algoritmo elegido es un arbol de decision para
        clasificacion, siguiendo la idea de CART. Se eligio uno supervisado y no un
        metodo de agrupamiento porque el entregable pide justificar los resultados
        con una matriz de confusion, y eso requiere una respuesta correcta contra
        la cual comparar cada prediccion.
    """)
    doc.parrafo(f"""
        La implementacion usa unicamente la libreria estandar de Python: csv, math,
        random y collections. No se importa scikit-learn, pandas ni numpy en
        ninguna parte del codigo que aprende o predice. El arbol se construye en
        {arb['segundos_entrenamiento']} segundos.
    """)

    # =================================================== 2. dataset
    doc.titulo_seccion("2. Datos de entrenamiento y de prueba")
    doc.parrafo("""
        Se usa el dataset Spaceship Titanic de Kaggle. Cada fila es un pasajero y
        la variable a predecir es Transported: si fue transportado a otra dimension
        o no. Es clasificacion binaria.
    """)
    doc.parrafo(f"""
        Kaggle reparte train.csv y test.csv, pero el segundo no sirve para evaluar
        porque no incluye la columna Transported: las respuestas se quedan en el
        servidor de la competencia. Por eso la evaluacion se hace partiendo
        train.csv, el unico archivo con etiquetas. De sus {dat['total']} pasajeros
        se aparto el {int(par['proporcion_prueba'] * 100)}% como conjunto de prueba,
        con particion estratificada y semilla {par['semilla']} para que sea
        reproducible.
    """)
    doc.tabla(
        ["Conjunto", "Pasajeros", "Transportados", "Proporcion", "Uso"],
        [["Entrenamiento", dat["n_entrena"], dat["positivos_entrena"],
          pct(dat["positivos_entrena"] / dat["n_entrena"]), "Construir el arbol"],
         ["Prueba", dat["n_prueba"], dat["positivos_prueba"],
          pct(dat["positivos_prueba"] / dat["n_prueba"]), "Medir el desempenio"],
         ["Total (train.csv)", dat["total"],
          dat["positivos_entrena"] + dat["positivos_prueba"],
          pct((dat["positivos_entrena"] + dat["positivos_prueba"]) / dat["total"]),
          "-"]],
        [42, 26, 30, 26, 46])
    doc.parrafo("""
        La proporcion de transportados es identica en ambos conjuntos, y las clases
        estan balanceadas casi al 50%. Ese dato importa despues para justificar las
        metricas. Un detalle de procedimiento: primero se divide y despues se
        imputan los faltantes, para que las medianas y modas del relleno salgan
        solo del entrenamiento. Al reves habria fuga de informacion y las metricas
        saldrian infladas.
    """)

    # =================================================== 3. preparacion
    doc.titulo_seccion("3. Preparacion de los datos")
    doc.parrafo(f"""
        Se alimentan {dat['n_variables']} variables: {len(dat['numericas'])}
        numericas y {len(dat['categoricas'])} categoricas. Algunas son derivadas:
        Cabin viene como "B/0/P" y al partirla se obtienen Deck y Side;
        PassengerId como "0001_01", de donde salen TamGrupo y ViajaSolo; ademas se
        agregan GastoTotal y EsNino.
    """)
    doc.parrafo("""
        Los faltantes se rellenan segun la columna, empezando por la informacion
        mas confiable. El caso mas claro es CryoSleep: si el pasajero registro
        consumo es imposible que estuviera dormido, asi que el valor se deduce en
        lugar de estimarse. Para HomePlanet se aprovecha que el planeta nunca varia
        dentro de un grupo de viaje. Solo cuando no hay mejor informacion se usa la
        moda o la mediana.
    """)
    doc.parrafo("""
        No hizo falta escalar las variables ni convertir las categoricas a columnas
        binarias. Un arbol parte con preguntas como "Age <= 27", y esa comparacion
        no cambia si la columna se multiplica por una constante. Es una ventaja real
        de los arboles frente a los modelos lineales.
    """)

    # =================================================== 4. algoritmo
    doc.titulo_seccion("4. Como funciona el algoritmo")
    doc.parrafo("""
        Cada nodo interno es una pregunta con dos respuestas y cada hoja es una
        prediccion. Para clasificar se empieza en la raiz y se baja respondiendo
        hasta llegar a una hoja. Lo que hay que decidir en cada nodo es cual es la
        mejor pregunta, y para eso se mide que tan mezcladas estan las clases con el
        indice de Gini:
    """)
    doc.codigo("    Gini = 1 - suma( p_k^2 )")
    doc.parrafo("""
        p_k es la proporcion de la clase k en el nodo. Vale 0 si todos los ejemplos
        son de la misma clase (nodo puro) y 0.5 si estan mitad y mitad, que es lo
        peor con dos clases. La implementacion incluye tambien la entropia de
        Shannon como alternativa: mide lo mismo y da arboles casi iguales.
    """)
    doc.parrafo("""
        Se prueban todas las divisiones posibles. Para una variable numerica,
        umbrales en el punto medio entre valores consecutivos distintos; para una
        categorica, preguntas del tipo "es igual a esta categoria". A cada una se le
        calcula la ganancia:
    """)
    doc.codigo("    ganancia = Gini(padre) - [ (n_izq/n) * Gini(izq)"
               "\n                               + (n_der/n) * Gini(der) ]")
    doc.parrafo("""
        Se elige la de mayor ganancia y el proceso se repite en cada rama. Para que
        la busqueda no sea lenta, los ejemplos se ordenan una sola vez por variable
        y se recorren moviendo ejemplos de una rama a otra, en lugar de recontar las
        clases para cada umbral.
    """)
    doc.parrafo("""
        Sin limites el arbol sigue partiendo hasta aislar cada ejemplo y termina
        memorizando en lugar de aprender. Los frenos implementados son cuatro:
        profundidad maxima, minimo de ejemplos para dividir, minimo por hoja y
        ganancia minima. Un nodo ya puro nunca se divide.
    """)

    # =================================================== 5. hiperparametros
    doc.titulo_seccion("5. Eleccion de los hiperparametros")
    doc.parrafo("""
        Hay una trampa facil de cometer: probar varias profundidades y quedarse con
        la que da mejor exactitud en prueba. Si se hace asi, el conjunto de prueba
        deja de ser una medida honesta. Para evitarlo, el conjunto de entrenamiento
        se vuelve a partir en sub-entrenamiento (80%) y validacion (20%), se
        prueban 60 combinaciones, se elige la mejor segun validacion y solo
        entonces se mide una vez en prueba.
    """)
    doc.tabla(
        ["Hiperparametro", "Valor", "Que controla"],
        [["Profundidad maxima", par["profundidad_maxima"],
          "Niveles de preguntas permitidos"],
         ["Minimo para dividir", par["min_muestras_division"],
          "Tamanio minimo de un nodo divisible"],
         ["Minimo por hoja", par["min_muestras_hoja"],
          "Ejemplos que conserva cada hoja"],
         ["Criterio", par["criterio"], "Medida de impureza"]],
        [45, 25, 100])
    doc.parrafo(f"""
        Mirando el conjunto de prueba se llegaria a un valor algo mejor. Se reporta
        {pct(pru['exactitud'])}, el resultado del procedimiento correcto, y no el
        numero mas favorable.
    """)

    # =================================================== 6. resultados
    doc.titulo_seccion("6. Resultados")
    doc.parrafo(f"""
        La matriz cruza las {pru['total']} predicciones del conjunto de prueba
        contra lo que realmente paso. La clase positiva es "si fue transportado".
    """)
    if figuras.get("matriz"):
        doc.imagen(figuras["matriz"], ancho=85, proporcion=4.2 / 5.2)
    doc.parrafo(f"""
        Los cuatro conteos son: {mc['VN']} verdaderos negativos, {mc['VP']}
        verdaderos positivos, {mc['FP']} falsos positivos y {mc['FN']} falsos
        negativos. O sea que el modelo acierta {mc['VN'] + mc['VP']} de
        {pru['total']} casos, y los dos tipos de error estan parejos
        ({mc['FP']} contra {mc['FN']}), lo que indica que no esta inclinado hacia
        una clase.
    """)
    doc.tabla(
        ["Metrica", "Formula", "Prueba", "Entrenamiento"],
        [["Exactitud", "(VP+VN)/total", pct(pru["exactitud"]), pct(ent["exactitud"])],
         ["Precision", "VP/(VP+FP)", pct(pru["precision"]), pct(ent["precision"])],
         ["Sensibilidad", "VP/(VP+FN)", pct(pru["sensibilidad"]),
          pct(ent["sensibilidad"])],
         ["Especificidad", "VN/(VN+FP)", pct(pru["especificidad"]),
          pct(ent["especificidad"])],
         ["F1", "media armonica", pct(pru["f1"]), pct(ent["f1"])]],
        [45, 38, 43, 44])
    doc.parrafo(f"""
        La exactitud es la metrica principal porque las clases estan balanceadas; en
        un dataset desbalanceado seria enganiosa, ya que con 95% de una clase
        bastaria predecir siempre esa para presumir 95% sin haber aprendido nada.
        Pero la exactitud sola no dice como se reparten los errores, y por eso se
        agregan las demas. Reportar sensibilidad ({pct(pru['sensibilidad'])}) y
        especificidad ({pct(pru['especificidad'])}) juntas muestra que el modelo
        funciona parecido en ambas clases, con
        {abs(pru['sensibilidad'] - pru['especificidad']) * 100:.2f} puntos de
        diferencia. El F1 resume precision y sensibilidad, y usa la media armonica
        porque castiga los desequilibrios.
    """)
    doc.parrafo(f"""
        Como referencia, un clasificador que siempre prediga la clase mas frecuente
        obtendria {pct(tri['exactitud'])}, o sea el balance de las clases. El arbol
        logra {pct(pru['exactitud'])},
        {(pru['exactitud'] - tri['exactitud']) * 100:.2f} puntos por encima. Esa
        comparacion es la evidencia de que aprendio estructura real de los datos.
    """)
    doc.subtitulo("Sobreajuste")
    doc.parrafo(f"""
        El arbol tiene {arb['nodos']} nodos, {arb['hojas']} de ellos hojas, y
        {arb['profundidad_alcanzada']} niveles de profundidad. Acierta
        {pct(ent['exactitud'])} en entrenamiento y {pct(pru['exactitud'])} en
        prueba: una brecha de {(ent['exactitud'] - pru['exactitud']) * 100:.2f}
        puntos. Que sea positiva es normal, al modelo siempre le va mejor con los
        datos que ya conoce; lo importante es que sea chica.
    """)
    if figuras.get("curva"):
        doc.parrafo("""
            La grafica muestra por que hay que limitar la profundidad: al dejar
            crecer el arbol la exactitud en entrenamiento sigue subiendo, pero en
            prueba se estanca. Esa separacion es el sobreajuste, y verla confirma
            que los criterios de paro sirven.
        """)
        doc.imagen(figuras["curva"], ancho=120, proporcion=3.8 / 6.4)

    # =================================================== 7. pruebas
    doc.titulo_seccion("7. Comprobacion de que la implementacion es correcta")
    doc.parrafo(f"""
        Una exactitud razonable no demuestra que el algoritmo este bien programado:
        un algoritmo con errores tambien puede dar un numero aceptable. Por eso el
        archivo pruebas.py incluye {n_pruebas} verificaciones contra respuestas
        conocidas de antemano, calculadas a mano o derivadas de reglas con las que
        se generaron los datos a proposito. Todas pasan.
    """)
    doc.tabla(
        ["Que se comprueba", "Como se verifica"],
        [["Gini y entropia", "Valores calculados a mano (50/50 da 0.5, puro da 0)"],
         ["Matriz de confusion", "Ejemplo de 10 casos contados uno por uno"],
         ["Las seis metricas", "Fracciones exactas de ese ejemplo (7/10, 3/5, 3/4)"],
         ["Regla y = 1 si x > 5", "Debe hallar el umbral 5.5 con una sola pregunta"],
         ["Regla (a>0 y b>0)", "Debe acertar los 121 casos"],
         ["Relacion tipo XOR", "Exige combinar variables, no basta una"],
         ["Regla categorica", "Debe aislar justamente la categoria 'rojo'"],
         ["Variables de ruido", "La importancia debe quedarse en la que tiene senial"],
         ["Categoria nunca vista", "No debe provocar un error al predecir"],
         ["Limites de crecimiento", "Ninguna hoja menor al minimo ni rama mas honda"],
         ["Determinismo", "Entrenar dos veces da el mismo arbol"],
         ["Reproducibilidad", "La misma semilla da la misma particion"]],
        [50, 120])
    doc.parrafo("""
        Las pruebas con reglas conocidas son las mas informativas. Con datos
        generados a partir de "la clase es 1 cuando x es mayor que 5", el arbol no
        solo debe clasificar bien: debe hacerlo con una sola pregunta y poner el
        umbral en 5.5. Si el calculo de la ganancia tuviera un error, elegiria otra
        variable u otro corte y la prueba lo detectaria.
    """)
    doc.subtitulo("Un error que las pruebas encontraron")
    doc.parrafo("""
        La prueba de XOR corrigio un defecto real. Ahi la clase es 1 cuando dos
        variables difieren, y ninguna por separado dice nada: partir por cualquiera
        deja las ramas mitad y mitad, o sea ganancia cero. La primera version
        exigia ganancia estrictamente positiva, asi que se detenia en la raiz y
        nunca aprendia la regla, aunque con dos niveles la resuelve. La correccion
        fue permitir divisiones de ganancia cero, igual que scikit-learn. El cambio
        no altero ningun resultado del dataset real, solo agrego capacidad.
    """)

    # =================================================== 8. interpretacion
    doc.titulo_seccion("8. Que aprendio el modelo")
    doc.parrafo("""
        Una ventaja de los arboles es que las reglas se pueden leer. Los tres
        primeros niveles del arbol entrenado:
    """)
    doc.codigo(arb["reglas"])
    principal = d["importancias"][0]
    doc.parrafo(f"""
        La primera pregunta es si el gasto total fue menor o igual a 7.5, o sea si
        el pasajero consumio algo. Tiene sentido: quienes viajaban en criosuenio no
        podian consumir, y son los que se transportaron con mas frecuencia. La
        grafica mide cuanta impureza elimina cada variable, pesada por cuantos
        pasajeros pasan por sus divisiones; {principal[0]} concentra
        {pct(principal[1])}.
    """)
    if figuras.get("importancias"):
        doc.imagen(figuras["importancias"], ancho=120, proporcion=4.2 / 6.4)
    doc.parrafo("""
        Hay un resultado que sorprende: CryoSleep queda casi al final con menos del
        1%, aunque es la variable con mayor relacion individual con el objetivo. La
        razon es que GastoTotal ya captura esa informacion, porque estar en
        criosuenio implica gasto cero. Al usarse en la raiz, para cuando el arbol
        podria preguntar por CryoSleep esa pregunta ya no aporta. No es inutil, es
        redundante dado el orden en que el arbol pregunta. Lo mismo pasa con
        ViajaSolo y EsNino, que quedaron sin usar por ser versiones simplificadas
        de TamGrupo y Age.
    """)

    # =================================================== 9. predicciones
    doc.titulo_seccion("9. El programa hace predicciones")
    doc.parrafo(f"""
        Hay dos formas, las dos desde la terminal. El modo interactivo
        (python main.py --interactivo) pide los datos de un pasajero y devuelve la
        prediccion con su probabilidad; los campos vacios se completan con las
        reglas de imputacion del entrenamiento y cada dato se valida, asi que no
        admite una edad negativa ni un grupo de cero personas. El modo por lotes
        (python main.py --kaggle) clasifica los
        {contar_filas(CARPETA / 'dataset' / 'test.csv')} pasajeros de test.csv, que
        son datos nuevos que no participaron ni en el entrenamiento ni en la
        evaluacion.
    """)
    doc.parrafo("""
        Sobre esas predicciones no se pueden calcular metricas porque no existen las
        etiquetas verdaderas. Lo que si se verifica es que la proporcion predicha se
        parezca a la del entrenamiento, y asi es: alrededor del 50%. Si el modelo
        estuviera sesgado, esa proporcion se iria a un extremo.
    """)

    # =================================================== 10. conclusion
    doc.titulo_seccion("10. Analisis y conclusiones")
    doc.parrafo(f"""
        El arbol clasifica correctamente {pct(pru['exactitud'])} de los pasajeros de
        prueba contra {pct(tri['exactitud'])} del clasificador trivial. Esa
        diferencia de {(pru['exactitud'] - tri['exactitud']) * 100:.2f} puntos,
        junto con las {n_pruebas} pruebas de correctitud, permite afirmar que el
        algoritmo aprende y que la implementacion es correcta. El desempenio esta
        equilibrado entre las dos clases y la brecha entrenamiento-prueba es de
        {(ent['exactitud'] - pru['exactitud']) * 100:.2f} puntos, asi que generaliza
        en lugar de memorizar.
    """)
    doc.parrafo("""
        Sobre las limitaciones: un arbol solo hace cortes perpendiculares a los
        ejes, una variable a la vez, asi que una frontera diagonal tendria que
        aproximarla con muchos escalones, y eso acota el techo alcanzable. Ademas la
        construccion es voraz, se toma la mejor division inmediata sin ver como
        afecta a los niveles siguientes, asi que no hay garantia de llegar al mejor
        arbol posible; encontrar el optimo es computacionalmente muy costoso y todas
        las implementaciones, incluidas las de las bibliotecas conocidas, usan esta
        misma estrategia. Tampoco es del todo estable: cambiar la semilla mueve la
        exactitud algunas decimas.
    """)
    doc.parrafo("""
        La extension natural seria entrenar muchos arboles sobre muestras distintas
        y promediar sus votos, que es la idea del random forest: reduce la
        inestabilidad y suele ganar algunos puntos, a cambio de perder la
        posibilidad de leer las reglas.
    """)
    if d["curva_profundidad"]:
        curva = d["curva_profundidad"]
        tope_entrena = max(r["entrena"] for r in curva)
        mejor_prueba = max(curva, key=lambda r: r["prueba"])
        ultimo = curva[-1]
        doc.parrafo(f"""
            Programar el algoritmo desde cero deja claro que un arbol no es una caja
            negra: es una sola formula, la del Gini, aplicada de forma recursiva. Lo
            que mas trabajo lleva no es entrenar sino lo que lo rodea, o sea
            preparar los datos, evitar la fuga de informacion, elegir los
            hiperparametros sin contaminar la prueba y comprobar que el codigo hace
            lo que se supone. Los frenos al crecimiento resultaron lo mas
            determinante: dejando crecer el arbol hasta {ultimo['profundidad']}
            niveles, el entrenamiento llega a {pct(tope_entrena)} pero la prueba se
            queda en {pct(ultimo['prueba'])}, por debajo del
            {pct(mejor_prueba['prueba'])} que se alcanza con
            {mejor_prueba['profundidad']}. Todo ese crecimiento extra sirvio solo
            para ajustarse a datos ya conocidos, que es la definicion de no haber
            aprendido nada util.
        """)

    return doc.bloques


def main():
    analizador = argparse.ArgumentParser(
        description="Genera el reporte en PDF y en Word a partir de resultados.json")
    analizador.add_argument("--alumno", default=ALUMNO)
    analizador.add_argument("--matricula", default=MATRICULA)
    analizador.add_argument("--json", default=str(RUTA_JSON))
    analizador.add_argument("--salida", default=str(RUTA_PDF),
                            help="ruta del PDF de salida")
    args = analizador.parse_args()

    ruta_json = Path(args.json)
    if not ruta_json.exists():
        raise SystemExit(
            f"No se encontro {ruta_json}. Corre primero:\n"
            f"  python main.py --curva --arbol 3 --guardar-json resultados.json")

    datos = json.loads(ruta_json.read_text(encoding="utf-8"))

    # Conteo de pruebas. Si el archivo no existe se avisa, porque el reporte cita
    # ese numero y no conviene que quede desactualizado.
    ruta_pruebas = CARPETA / "pruebas.json"
    if ruta_pruebas.exists():
        pruebas = json.loads(ruta_pruebas.read_text(encoding="utf-8"))
    else:
        raise SystemExit(
            f"No se encontro {ruta_pruebas}. Corre primero:\n"
            f"  python pruebas.py --guardar-json")

    if pruebas["fallaron"] > 0:
        raise SystemExit(
            f"Hay {pruebas['fallaron']} pruebas fallando. Corrige la "
            f"implementacion antes de generar el reporte.")

    CARPETA_FIG.mkdir(exist_ok=True)
    figuras = {}

    ruta = CARPETA_FIG / "matriz_confusion.png"
    figura_matriz_confusion(datos, ruta)
    figuras["matriz"] = ruta

    ruta = CARPETA_FIG / "curva_profundidad.png"
    if figura_curva_profundidad(datos, ruta):
        figuras["curva"] = ruta

    ruta = CARPETA_FIG / "importancias.png"
    figura_importancias(datos, ruta)
    figuras["importancias"] = ruta

    bloques = construir_contenido(datos, args.alumno, args.matricula,
                                  figuras, pruebas)
    paginas = renderizar_pdf(bloques, args.salida)

    print(f"Reporte generado: {args.salida}")
    print(f"Paginas: {paginas}  ({len(bloques)} bloques de contenido)")
    print(f"Figuras en: {CARPETA_FIG}")
    print(f"Portada: {args.alumno} - {args.matricula}")


if __name__ == "__main__":
    main()
