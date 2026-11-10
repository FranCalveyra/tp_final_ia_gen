"""Lectura de trazas del kit: parámetros normativos que vio cada una y fila equivalente del historial.

Las trazas no traen el `Caso` original. Todo se reconstruye de lo que el entorno devolvió
(nota de triage, respuestas, verificación y líneas «Valores: ...» de las búsquedas) y de la
ficha registrada. Un formato desconocido levanta `ValueError`: no se adivina.
"""

import re

PATRON_ENCABEZADO = re.compile(r"^## (NOR-[A-Z]{3}-\d{2})\b", re.M)
_CREDENCIAL = re.compile(
    r"la credencial de (.+?) \((obra social|prepaga)\), número de afiliado ([0-9][0-9/-]*[0-9])")
_DECLARADA = re.compile(r"dice que tiene (.+) \((obra social|prepaga)\), número de afiliado (\S+)\.$")
_ART = re.compile(r"dice que la ART de su empleador es (.+)\. CUIL del paciente: (\S+)\.$")
_VERIFICACION = re.compile(r" informa que la afiliación .+ está (activa|inactiva)\.$")


def parametros_observados(traza) -> dict[str, str]:
    """{código: valor como texto} de las líneas «Valores:» de sus búsquedas; error si se contradicen."""
    valores = {}
    for paso in traza["pasos"]:
        if paso["herramienta"] != "buscar_normativa":
            continue
        for linea in paso["resultado"].splitlines():
            if not linea.startswith("Valores: "):
                continue
            for asignacion in linea[len("Valores: "):].removesuffix(".").split("; "):
                codigo, sep, valor = asignacion.partition(" = ")
                if not sep or not re.fullmatch(r"P-[A-Z]{3}-\d+", codigo) or not valor:
                    raise ValueError("línea de valores con formato desconocido")
                if valores.setdefault(codigo, valor) != valor:
                    raise ValueError(f"valores contradictorios de {codigo} en una traza")
    return valores


def secciones_en(texto: str) -> list[str]:
    """Códigos de sección de los encabezados `## NOR-...` de una observación de búsqueda."""
    return PATRON_ENCABEZADO.findall(texto)


def _campo(lineas: dict[str, str], prefijo: str) -> str:
    if prefijo not in lineas:
        raise ValueError(f"nota de triage sin «{prefijo}»")
    return lineas[prefijo]


def _sin_punto(texto: str) -> str:
    return texto[:-1] if texto.endswith(".") else texto


def _credenciales(texto: str) -> str:
    if texto in ("no presentó credencial.",):
        return ""
    entradas = _CREDENCIAL.findall(texto)
    if not entradas:
        raise ValueError("credenciales con formato desconocido")
    return "; ".join(f"{entidad} ({tipo}) {numero}" for entidad, tipo, numero in entradas)


def fila_triage(nota: str) -> dict[str, str]:
    """Campos de triage con el formato de `historial/historial.csv`."""
    lineas = {}
    for linea in nota.splitlines()[1:]:
        clave, sep, valor = linea.partition(": ")
        if sep:
            lineas[clave] = valor
    nombre_edad = _sin_punto(_campo(lineas, "Paciente"))
    nombre, sep, edad = nombre_edad.rpartition(", ")
    if not sep or not edad.endswith(" años"):
        raise ValueError("edad con formato desconocido")
    documento = _campo(lineas, "Documento")
    if documento == "no presentó documento.":
        documento_tipo = documento_numero = ""
    elif documento.startswith("presentó "):
        documento_tipo, _, documento_numero = _sin_punto(documento[len("presentó "):]).partition(" ")
    else:
        raise ValueError("documento con formato desconocido")
    credenciales = lineas.get("Credenciales de cobertura", lineas.get("Credencial de cobertura"))
    if credenciales is None:
        raise ValueError("nota de triage sin credenciales")
    puede = _campo(lineas, "Puede responder preguntas")
    acompanante = _sin_punto(_campo(lineas, "Acompañante"))
    trabajo = _campo(lineas, "Se lesionó o enfermó trabajando o en el trayecto al trabajo")
    if trabajo == "no.":
        lesion = "no"
    elif trabajo.startswith("sí, "):
        lesion = _sin_punto(trabajo[len("sí, "):])
    else:
        raise ValueError("lesión laboral con formato desconocido")
    return {
        "nombre": nombre,
        "edad": edad.removesuffix(" años"),
        "nivel": _sin_punto(_campo(lineas, "Nivel de triage")),
        "motivo": _sin_punto(_campo(lineas, "Motivo de consulta")),
        "nota_enfermeria": _campo(lineas, "Nota de enfermería"),
        "sospecha_sepsis": _sin_punto(_campo(lineas, "Sospecha de sepsis")),
        "llegada": _sin_punto(_campo(lineas, "Llegada")),
        "documento_tipo": documento_tipo,
        "documento_numero": documento_numero,
        "credenciales": _credenciales(credenciales),
        "puede_responder": "sí" if puede == "sí." else "no",
        "acompanante": "" if acompanante == "ninguno" else acompanante,
        "lesion_trabajo": lesion,
    }


def _texto_ficha(valor) -> str:
    if valor is None:
        return ""
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    return str(valor)


def fila_desde_traza(traza) -> dict[str, str]:
    """Fila con las columnas de `historial/historial.csv` reconstruida desde la traza."""
    pasos = traza["pasos"]
    if not pasos or pasos[0]["herramienta"] != "leer_triage" or pasos[-1]["herramienta"] != "registrar_decision":
        raise ValueError("la traza debe empezar en leer_triage y terminar en registrar_decision")
    fila = {"id": traza["id"], **fila_triage(pasos[0]["resultado"]),
            "cobertura_declarada": "", "art_declarada": "", "entidad_verificada": "",
            "numero_verificado": "", "respuesta_verificacion": ""}
    for paso in pasos[1:-1]:
        herramienta, args, resultado = paso["herramienta"], paso["argumentos"], paso["resultado"]
        if herramienta == "preguntar" and args.get("tema") == "cobertura":
            if resultado.endswith("dice que no tiene obra social ni prepaga."):
                fila["cobertura_declarada"] = "ninguna"
            elif coincidencia := _DECLARADA.search(resultado):
                entidad, tipo, numero = coincidencia.groups()
                fila["cobertura_declarada"] = f"{entidad} ({tipo}) {numero}"
        elif herramienta == "preguntar" and args.get("tema") == "art":
            if coincidencia := _ART.search(resultado):
                fila["art_declarada"] = f"{coincidencia.group(1)} CUIL {coincidencia.group(2)}"
        elif herramienta == "verificar_afiliacion":
            if fila["entidad_verificada"]:
                raise ValueError("más de una verificación de afiliación")
            coincidencia = _VERIFICACION.search(resultado)
            if not coincidencia:
                raise ValueError("respuesta de verificación desconocida")
            fila["entidad_verificada"] = args["entidad"]
            fila["numero_verificado"] = args["numero_afiliado"]
            fila["respuesta_verificacion"] = coincidencia.group(1)
    ficha = pasos[-1]["argumentos"]
    for campo in ("registro", "numero_documento", "box_antes_del_registro", "circuito",
                  "plazo_circuito_min", "cobertura", "entidad_a_notificar", "aviso_al_voluntariado"):
        fila[campo] = _texto_ficha(ficha[campo])
    return fila
