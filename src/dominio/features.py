"""
Extracción de características (features) de un correo electrónico
para alimentar al clasificador SVM de phishing.

Se combinan dos tipos de features:
1. Texto: matriz TF-IDF sobre "asunto + cuerpo".
2. Ingenieradas: señales típicas de phishing (URLs sospechosas, urgencia,
   solicitud de datos sensibles, etc.).

Ambas se concatenan en una sola matriz dispersa (scipy.sparse.hstack)
antes de entrenar o predecir con el SVM.

"""

import ipaddress #Modulo para validar direcciones IP (IPv4/IPv6) de forma correcta
import re #Modulo de exprensiones regulares de Python
from urllib.parse import urlparse #Modulo para descomponer una URL en sus componentes

import numpy as np #Modulo para construir un arreglo de 7 numeros
from scipy.sparse import csr_matrix, hstack #Modulo para convertir arreglo en matriz
#dispersa compatible
from sklearn.feature_extraction.text import TfidfVectorizer
#Importa la clase de scikit-learn que calcula el TF-IDF del texto
#(frecuencia de término × frecuencia inversa de documento).
import pandas as pd  # Modulo para construir el DataFrame de un unico correo

# Stopwords en español para el TF-IDF. Se define una lista.

STOPWORDS_ESPANOL = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
    "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como",
    "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta",
    "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
    "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
    "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro",
    "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes",
    "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas",
    "algo", "nosotros", "es", "son", "fue", "ser", "tiene", "han",
]

#Utilizado para la búsqueda de URLs, no considera-> espacios,<>,",'
_PATRON_URL = re.compile(r"https?://[^\s<>\"']+")

# Acortadores de URL comúnmente abusados en campañas de phishing
_ACORTADORES_CONOCIDOS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
}

#Validar si el domino se trata de una IPv4 o IPv6
def validar_direccion_ip(host: str) -> bool:
    """
    Indica si `host` es una dirección IP literal (IPv4 o IPv6), en vez de
    un nombre de dominio. A diferencia de una regex de dígitos, valida
    rangos reales (ej. rechaza "999.999.999.999") y cubre IPv6.
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


def _dominio_de_url(url: str) -> str:
    """
    Extrae el dominio (netloc) de una URL, en minúsculas y sin 'www.'.
    """
    dominio = urlparse(url).netloc.lower()
    return dominio[4:] if dominio.startswith("www.") else dominio


def tiene_url_sospechosa(texto: str) -> bool:
    """
    Detecta si alguna URL del texto es sospechosa:
    - Usa una IP literal en vez de un dominio (típico de phishing improvisado).
    - Usa un acortador de URLs conocido (oculta el destino real).
    """
    #Realiza la busqueda de URLs en el asunto y cuerpo de correo
    for url in _PATRON_URL.findall(texto or ""): #En caso sea vacio, se reemplaza por un espacio en blanco
        dominio = _dominio_de_url(url)
        # .hostname (no dominio.split(":")[0]) descarta el puerto Y los
        # corchetes de un host IPv6 literal correctamente. Un split manual
        # por ":" rompe con IPv6 (ej. "http://[2001:db8::1]/x"): el primer
        # ":" que encuentra esta DENTRO de los corchetes, no separando el
        # puerto, asi que terminaria comparando "[" en vez de la IP real.
        host = (urlparse(url).hostname or "").lower()
        if validar_direccion_ip(host): #Validar si el dominio se trata de una IP
            return True
        if dominio in _ACORTADORES_CONOCIDOS: #Validar si el dominio se trata de un acortar conocido
            return True
    return False


def dominio_coincide(remitente: str, texto: str) -> bool:
    """
    Compara el dominio del remitente con los dominios de las URLs del cuerpo.
    Un correo legítimo normalmente enlaza a su propio dominio; el phishing
    suele enlazar a un dominio distinto al que dice representar.
    Si no hay URLs en el texto, se considera que "coincide" (no hay señal
    de alarma que reportar).
    """
    urls = _PATRON_URL.findall(texto or "") #Lista los URLs presentes en el correo y si está vacio se reemplaza por un espacio en blanco
    if not urls:
        return True
    dominio_remitente = (remitente or "").split("@")[-1].strip().lower() #Separa el dominio antes y despues del arroba, se utiliza el despues en minuscula
    dominios_en_texto = {_dominio_de_url(url) for url in urls} #Dominio de los URLs listados
    return dominio_remitente in dominios_en_texto #Busqueda del dominio remitente en los dominios de las URLs


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
    Detecta saludos genéricos ("Estimado cliente") en vez de un nombre propio,
    señal típica de campañas de phishing masivas no personalizadas.
    """
    texto_normalizado = (texto or "").lower() #Se transforma en minuscula. En caso sea vacio, se reemplaza por espacio en blanco
    return any(saludo in texto_normalizado for saludo in _SALUDOS_GENERICOS) #Devuelve verdadero apenas si encuentra un saludo generico en el correo


# Nombres de las 7 features ingenieradas, en el mismo orden que las arma
# extraer_features_numericas. Se reutiliza en evaluación para mostrar a
# detalle qué señales detectó el modelo en cada correo.
NOMBRES_FEATURES = [
    "urls",
    "url_sospechosa",
    "dominio_coincide",
    "urgencia",
    "mayusculas",
    "datos_sensibles",
    "saludo_generico",
]


def extraer_features_numericas(asunto: str, cuerpo: str, remitente: str) -> list:
    """
    Combina todas las features ingenieradas de un correo en un solo vector
    numérico de longitud fija (7), en un orden estable y documentado:
    [urls, url_sospechosa, dominio_coincide, urgencia, mayusculas,
     datos_sensibles, saludo_generico]
    """
    texto_completo = f"{asunto}\n{cuerpo}"

    #Si es True es 1 y False es 0
    return [
        float(contar_urls(texto_completo)),
        float(tiene_url_sospechosa(texto_completo)),
        float(dominio_coincide(remitente, texto_completo)),
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
    Construye la matriz de features final (TF-IDF + ingenieradas) para un
    DataFrame con columnas 'asunto', 'cuerpo', 'remitente'.

    ajustar=True  -> ajusta (fit) el vectorizador con estos textos (usar solo
                      en entrenamiento, con el set de entrenamiento).
    ajustar=False -> reutiliza un vectorizador ya ajustado (usar en
                      evaluación y en el monitor de Outlook en producción).
    """
    textos = (df["asunto"].fillna("") + " " + df["cuerpo"].fillna("")).tolist() #Lista la combinación del asunto y cuerpo de correo en String.
    if ajustar:
        matriz_tfidf = vectorizador.fit_transform(textos) #El vectorizador aprende el vocabulario (qué palabras existen, sus pesos IDF)
    else:
        matriz_tfidf = vectorizador.transform(textos) #El vectorizador utiliza el vocabulario aprendido previamente

    remitentes = df["remitente"].fillna("").tolist() #Remitentes vacios (NaN) se tratan como string vacio, igual que asunto/cuerpo
    features_numericas = np.array([ #Almacena el vector procesado
        extraer_features_numericas(fila["asunto"], fila["cuerpo"], remitente) #Extrae el vector de las 7 ingenieradas
        for (_, fila), remitente in zip(df.iterrows(), remitentes) #Itera correo por (indice, fila), "_" descarta el indice
    ])

    #Transforma el formato de "features_numericas" a spmatrix
    #Hstack unifica las 2 matrices en uno solo para ser utilizado en el modelo SVM
    return hstack([matriz_tfidf, csr_matrix(features_numericas)])


def _coef_del_modelo(modelo):
    """
    Extrae los coeficientes lineales del modelo, sea un SVC(kernel="linear")
    directo o uno envuelto en CalibratedClassifierCV (ver entrenar_modelo.py
    -- reemplaza a SVC(probability=True), deprecado). CalibratedClassifierCV
    no expone coef_ en su nivel superior a proposito (la calibracion en si
    no es un modelo lineal); con ensemble=False el SVM real queda anidado
    en un unico calibrated_classifiers_[0].estimator (se ajusta una sola
    vez sobre todos los datos, la validacion cruzada solo calibra).
    """
    if hasattr(modelo, "coef_"):
        return modelo.coef_
    return modelo.calibrated_classifiers_[0].estimator.coef_


def explicar_clasificacion(
    modelo, vectorizador, asunto: str, cuerpo: str, remitente: str, top_n: int = 5, X=None,
) -> dict:
    """
    Descompone la decision del SVM lineal para un unico correo: cuanto
    contribuyo cada feature ingeniada y que palabras del texto (TF-IDF)
    empujaron mas hacia "phishing".

    `X` es opcional: si el llamante ya construyo la matriz de features para
    este mismo correo (p.ej. procesar_correo, que la necesita ademas para
    clasificar), se puede pasar aqui para no recalcular el TF-IDF y las 7
    features ingenieradas por segunda vez. Si no se provee, se construye
    igual que antes (asunto/cuerpo/remitente son obligatorios en ambos
    casos, para armar el DataFrame internamente cuando X es None).

    Requiere un SVM lineal binario (SVC(kernel="linear") directo, o
    envuelto en CalibratedClassifierCV -- ver _coef_del_modelo) con
    modelo.classes_ == ['benigno', 'phishing'] (mismo invariante que ya
    asume ETIQUETA_PHISHING en monitor_outlook.py): coef_ positivo empuja
    hacia phishing, negativo hacia benigno.
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
    contribuciones_palabras.sort(key=lambda item: item[2], reverse=True)

    return {
        "features_ingenieradas": features_ingenieradas,
        "top_palabras": contribuciones_palabras[:top_n],
    }
