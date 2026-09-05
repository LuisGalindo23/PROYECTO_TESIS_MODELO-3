"""
Extracción de características (features) de un correo electrónico para alimentar al clasificador SVM de phishing.

Se combinan dos tipos de features:
1. Texto: matriz TF-IDF sobre "asunto + cuerpo".
2. Ingenieradas: señales típicas de phishing (URLs sospechosas, urgencia, solicitud de datos sensibles, etc.).

Ambas se concatenan en una sola matriz dispersa (scipy.sparse.hstack)
antes de entrenar o predecir con el SVM.
"""

import ipaddress #Modulo para validar direcciones IP (IPv4/IPv6) de forma correcta
import re #Modulo para declarar exprensiones regulares de Python
from urllib.parse import urlparse #Modulo para descomponer una URL en sus componentes

import tldextract #Para calcular el dominio raiz real (usa la Public Suffix List, entiende TLDs compuestos como .com.mx)

import numpy as np #Modulo para construir un arreglo de 7 numeros
from scipy.sparse import csr_matrix, hstack #Modulo para convertir arreglo en matriz
#dispersa compatible
from sklearn.feature_extraction.text import TfidfVectorizer
#Importa la clase de scikit-learn que calcula el TF-IDF del texto
#(frecuencia de término × frecuencia inversa de documento).
import pandas as pd  # Modulo para construir el DataFrame de un unico correo

# Stopwords en español para el TF-IDF. Base "stopwords" de NLTK (biblioteca de código abierto en Python diseñada para el procesamiento del lenguaje natural).
# Palabras/términos muy comunes en un idioma que no aportan un significado relevante.

STOPWORDS_ESPANOL = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como",
    "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
    "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre",
    "era", "erais", "eran", "eras", "eres", "es", "esa", "esas",
    "ese", "eso", "esos", "esta", "estaba", "estabais", "estaban", "estabas",
    "estad", "estada", "estadas", "estado", "estados", "estamos", "estando", "estar",
    "estaremos", "estará", "estarán", "estarás", "estaré", "estaréis", "estaría", "estaríais",
    "estaríamos", "estarían", "estarías", "estas", "este", "estemos", "esto", "estos",
    "estoy", "estuve", "estuviera", "estuvierais", "estuvieran", "estuvieras", "estuvieron", "estuviese",
    "estuvieseis", "estuviesen", "estuvieses", "estuvimos", "estuviste", "estuvisteis", "estuviéramos", "estuviésemos",
    "estuvo", "está", "estábamos", "estáis", "están", "estás", "esté", "estéis",
    "estén", "estés", "fue", "fuera", "fuerais", "fueran", "fueras", "fueron",
    "fuese", "fueseis", "fuesen", "fueses", "fui", "fuimos", "fuiste", "fuisteis",
    "fuéramos", "fuésemos", "ha", "habida", "habidas", "habido", "habidos", "habiendo",
    "habremos", "habrá", "habrán", "habrás", "habré", "habréis", "habría", "habríais",
    "habríamos", "habrían", "habrías", "habéis", "había", "habíais", "habíamos", "habían",
    "habías", "han", "has", "hasta", "hay", "haya", "hayamos", "hayan",
    "hayas", "hayáis", "he", "hemos", "hube", "hubiera", "hubierais", "hubieran",
    "hubieras", "hubieron", "hubiese", "hubieseis", "hubiesen", "hubieses", "hubimos", "hubiste",
    "hubisteis", "hubiéramos", "hubiésemos", "hubo", "la", "las", "le", "les",
    "lo", "los", "me", "mi", "mis", "mucho", "muchos", "muy",
    "más", "mí", "mía", "mías", "mío", "míos", "nada", "ni",
    "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros",
    "o", "os", "otra", "otras", "otro", "otros", "para", "pero",
    "poco", "por", "porque", "que", "quien", "quienes", "qué", "se",
    "sea", "seamos", "sean", "seas", "sentid", "sentida", "sentidas", "sentido",
    "sentidos", "seremos", "será", "serán", "serás", "seré", "seréis", "sería",
    "seríais", "seríamos", "serían", "serías", "seáis", "sido", "siente", "sin", "sintiendo",
    "sobre", "sois", "somos", "son", "soy", "su", "sus", "suya",
    "suyas", "suyo", "suyos", "sí", "también", "tanto", "te", "tendremos",
    "tendrá", "tendrán", "tendrás", "tendré", "tendréis", "tendría", "tendríais", "tendríamos",
    "tendrían", "tendrías", "tened", "tenemos", "tenga", "tengamos", "tengan", "tengas",
    "tengo", "tengáis", "tenida", "tenidas", "tenido", "tenidos", "teniendo", "tenéis",
    "tenía", "teníais", "teníamos", "tenían", "tenías", "ti", "tiene", "tienen",
    "tienes", "todo", "todos", "tu", "tus", "tuve", "tuviera", "tuvierais",
    "tuvieran", "tuvieras", "tuvieron", "tuviese", "tuvieseis", "tuviesen", "tuvieses", "tuvimos",
    "tuviste", "tuvisteis", "tuviéramos", "tuviésemos", "tuvo", "tuya", "tuyas", "tuyo",
    "tuyos", "tú", "un", "una", "uno", "unos", "vosotras", "vosotros",
    "vuestra", "vuestras", "vuestro", "vuestros", "y", "ya", "yo", "él",
    "éramos",
]

#Utilizado para la búsqueda de URLs, no considera-> espacios,<>,",'
_PATRON_URL = re.compile(r"https?://[^\s<>\"']+")

# Utilizado para extraer dominios de las URLs
_EXTRACTOR_DOMINIOS = tldextract.TLDExtract(suffix_list_urls=())

# Acortadores de URL comúnmente abusados en campañas de phishing
_ACORTADORES_CONOCIDOS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
}

#Validar si el domino se trata de una IPv4 o IPv6
def validar_direccion_ip(host: str) -> bool:
    """
    Indica si el dominio de la URL es una dirección IP literal (IPv4 o IPv6), en vez de un nombre de dominio.
    A diferencia de una regex de dígitos, valida rangos reales (ej. rechaza "999.999.999.999") y cubre IPv6.
    """
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


#Palabras utilizadas frecuentemente en campañas de phishing
_PALABRAS_URGENCIA = [
    "urgente", "urgentemente", "inmediato", "inmediatamente", "ahora mismo",
    "bloqueada", "bloqueado", "suspendida", "suspendido", "verifique",
    "verificar ahora", "expira", "vence hoy", "última oportunidad",
    "actúe ahora", "cuenta será cerrada", "seleccionado"
]

#Palabras que hace referencia a datos sensibles
_PATRONES_DATOS_SENSIBLES = [
    "contraseña", "número de tarjeta", "numero de tarjeta", "cvv",
    "cédula", "cedula", "clic aquí", "clic aqui", "haga clic",
    "confirme sus datos", "actualice sus datos bancarios", "datos personales"
]

#Saludos frecuentemente utilizados en campañas de phishing
_SALUDOS_GENERICOS = [
    "estimado cliente", "estimado usuario", "estimado/a cliente",
    "apreciado cliente", "querido usuario", "estimado señor/señora", "estimado ganador"
]


def contar_urls(texto: str) -> int:
    """Cuenta cuántas URLs (http/https) aparecen en el texto."""
    return len(_PATRON_URL.findall(texto or ""))


def _dominio_raiz(url_o_dominio: str) -> str:
    """
    Reduce una URL o un dominio a su dominio raiz.
    """

    try:
        extraido = _EXTRACTOR_DOMINIOS(url_o_dominio)
    except Exception:
        return ""
    return f"{extraido.domain}.{extraido.suffix}" if extraido.suffix else extraido.domain


def _hostname_seguro(url: str) -> str:
    """
    Extrae el dominio de una URL (para validar_direccion_ip), sin dejar que una URL mal formada tumbe la evaluacion del correo entero.
    """
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def tiene_url_sospechosa(texto: str) -> bool:
    """
    Detecta si alguna URL del texto es sospechosa:
    - Usa una IP literal en vez de un dominio (típico de phishing improvisado).
    - Usa un acortador de URLs conocido (oculta el destino real), incluyendo subdominios de un acortador conocido (ej. "mirror1.bit.ly").
    """
    #Realiza la busqueda de URLs en el asunto y cuerpo de correo
    for url in _PATRON_URL.findall(texto or ""): #En caso sea vacio, se reemplaza por un espacio en blanco
        if validar_direccion_ip(_hostname_seguro(url)): #Validar si el dominio se trata de una IP
            return True
        if _dominio_raiz(url) in _ACORTADORES_CONOCIDOS: #Validar si el dominio raiz es un acortador conocido
            return True
    return False



def contar_palabras_urgencia(texto: str) -> int:
    """Cuenta cuántas frases/palabras de urgencia típicas de phishing aparecen."""
    texto_normalizado = (texto or "").lower() #Se transforma en minuscula. En caso sea vacio, se reemplaza por espacio en blanco
    return sum(1 for palabra in _PALABRAS_URGENCIA if palabra in texto_normalizado) #Contabiliza las palabras de urgencia


def proporcion_mayusculas(asunto: str) -> float:
    """
    Proporción de letras mayúsculas sobre el total de letras del asunto.
    El phishing abusa de MAYÚSCULAS para transmitir urgencia/alarma.
    """
    letras = [c for c in (asunto or "") if c.isalpha()] #Solo lista los caracteres, descartando espacio, digitos, simbolos, etc.
    if not letras:
        return 0.0
    mayusculas = [c for c in letras if c.isupper()] #Contabiliza los caracteres en mayuscula
    return len(mayusculas) / len(letras) #Porcentaje de mayusculas en el correo


def solicita_datos_sensibles(texto: str) -> bool:
    """Detecta si el texto pide explícitamente datos sensibles o clics inmediatos."""
    texto_normalizado = (texto or "").lower() #Se transforma en minuscula. En caso sea vacio, se reemplaza por espacio en blanco 
    return any(patron in texto_normalizado for patron in _PATRONES_DATOS_SENSIBLES) #Devuelve verdadero apenas si encuentra un dato sensible en el correo


def saludo_generico(texto: str) -> bool:
    """
    Detecta saludos genéricos ("Estimado cliente") en vez de un nombre propio, señal típica de campañas de phishing masivas no personalizadas.
    """
    texto_normalizado = (texto or "").lower() #Se transforma en minuscula. En caso sea vacio, se reemplaza por espacio en blanco
    return any(saludo in texto_normalizado for saludo in _SALUDOS_GENERICOS) #Devuelve verdadero apenas si encuentra un saludo generico en el correo


# Nombres de las 7 features ingenieradas, en el mismo orden que las arma
# extraer_features_numericas. Se reutiliza en evaluación para mostrar a
# detalle qué señales detectó el modelo en cada correo.
NOMBRES_FEATURES = [
    "urls",
    "url_sospechosa",
    "urgencia",
    "mayusculas",
    "datos_sensibles",
    "saludo_generico",
]


def extraer_features_numericas(asunto: str, cuerpo: str, remitente: str) -> list:
    """
    Combina todas las features ingenieradas de un correo en un solo vector en un orden estable y documentado:
    [urls, url_sospechosa, urgencia, mayusculas, datos_sensibles, saludo_generico]
    """

    texto_completo = f"{asunto}\n{cuerpo}"

    #Si es True es 1 y False es 0
    return [
        float(contar_urls(texto_completo)),
        float(tiene_url_sospechosa(texto_completo)),
        float(contar_palabras_urgencia(texto_completo)),
        float(proporcion_mayusculas(asunto)),
        float(solicita_datos_sensibles(texto_completo)), 
        float(saludo_generico(texto_completo)),
    ]


def crear_vectorizador_tfidf() -> TfidfVectorizer:
    """Crea el vectorizador TF-IDF usado para transformar el texto del correo en una matriz numerica."""
    return TfidfVectorizer(stop_words=STOPWORDS_ESPANOL, max_features=3000) #Ignora las palabras "STOPWORDS"


def construir_matriz_features(df, vectorizador: TfidfVectorizer, ajustar: bool): #df= Data Frame (Correos)
    """
    Construye la matriz de features final (TF-IDF + ingenieradas) para un DataFrame con columnas 'asunto', 'cuerpo', 'remitente'.

    ajustar=True  -> ajusta (fit) el vectorizador con estos textos.
    ajustar=False -> reutiliza un vectorizador ya ajustado.
    """

    textos = (df["asunto"].fillna("") + " " + df["cuerpo"].fillna("")).tolist() #Lista la combinación del asunto y cuerpo de correo en String.
    if ajustar:
        matriz_tfidf = vectorizador.fit_transform(textos) #El vectorizador aprende el vocabulario (qué palabras existen, sus pesos IDF)
    else:
        matriz_tfidf = vectorizador.transform(textos) #El vectorizador reutiliza el vocabulario aprendido previamente

    remitentes = df["remitente"].fillna("").tolist() #Remitentes vacios (NaN) se tratan como string vacio, igual que asunto/cuerpo
    features_numericas = np.array([ #Almacena el vector procesado
        extraer_features_numericas(fila["asunto"], fila["cuerpo"], remitente) #Extrae el vector de las 6 ingenieradas
        for (_, fila), remitente in zip(df.iterrows(), remitentes) #Itera correo por (indice, fila), "_" descarta el indice
    ])

    #Transforma el formato de "features_numericas" a spmatrix
    #Hstack unifica las 2 matrices en uno solo para ser utilizado en el modelo SVM
    return hstack([matriz_tfidf, csr_matrix(features_numericas)])


def _coef_del_modelo(modelo):
    """
    Extrae los coeficientes lineales del modelo.
    """
    if hasattr(modelo, "coef_"):
        return modelo.coef_
    return modelo.calibrated_classifiers_[0].estimator.coef_


def explicar_clasificacion(
    modelo, vectorizador, asunto: str, cuerpo: str, remitente: str, top_n: int = 5, X=None,
) -> dict:
    """
    Descompone la decision del SVM lineal para un unico correo: cuanto contribuyo cada feature ingeniada y que palabras del texto (TF-IDF)
    empujaron mas hacia "phishing".
    """

    if X is None:
        df_correo = pd.DataFrame([{"asunto": asunto, "cuerpo": cuerpo, "remitente": remitente}])
        X = construir_matriz_features(df_correo, vectorizador, ajustar=False)
    valores = X.toarray()[0]
    pesos = _coef_del_modelo(modelo).toarray()[0]

    nombres_palabras = vectorizador.get_feature_names_out()
    num_palabras_tfidf = len(nombres_palabras)

    if len(pesos) != num_palabras_tfidf + len(NOMBRES_FEATURES):
        raise ValueError(
            f"Modelo y vectorizador desalineados: coef_ tiene {len(pesos)} pesos, "
            f"se esperaban {num_palabras_tfidf + len(NOMBRES_FEATURES)}"
        )

    valores_tfidf = valores[:num_palabras_tfidf]
    pesos_tfidf = pesos[:num_palabras_tfidf]
    valores_ingenieradas = valores[num_palabras_tfidf:]
    pesos_ingenieradas = pesos[num_palabras_tfidf:]

    features_ingenieradas = [
        (nombre, float(valor), float(valor * peso))
        for nombre, valor, peso in zip(NOMBRES_FEATURES, valores_ingenieradas, pesos_ingenieradas)
    ]

    contribuciones_palabras = [
        (nombres_palabras[i], float(valores_tfidf[i]), float(valores_tfidf[i] * pesos_tfidf[i]))
        for i in range(num_palabras_tfidf)
        if valores_tfidf[i] != 0.0
    ]
    contribuciones_palabras.sort(key=lambda item: item[2], reverse=True) #Ordena la lista contribuciones_palabras de mayor a menor contribución,
    #para poder quedarse después solo con las palabras que más pesaron.

    return {
        "features_ingenieradas": features_ingenieradas,
        "top_palabras": contribuciones_palabras[:top_n],
    }
