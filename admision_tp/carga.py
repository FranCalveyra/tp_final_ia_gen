"""Carga de Gemma 4 solo con el modelo de lenguaje (inferencia y entrenamiento)."""


def cargar_solo_texto(transformers, modelo: str, revision: str, **opciones):
    """El agente es solo texto: sin torres de visión ni audio no se crean ni se cargan sus pesos.

    Gemma4Config declara ambas configuraciones opcionales y Gemma4Model no crea la torre ni su
    embedder cuando son None (modeling_gemma4.py:2164-2180). El forward solo las usa si recibe
    pixel_values o input_features, así que el texto no cambia. Los pesos de las torres quedan como
    claves no usadas del checkpoint. El LoRA apunta a language_model, que no se toca.
    """
    config = transformers.AutoConfig.from_pretrained(modelo, revision=revision)
    for parte in ("vision_config", "audio_config"):
        if hasattr(config, parte):
            setattr(config, parte, None)
    return transformers.AutoModelForCausalLM.from_pretrained(modelo, revision=revision, config=config,
                                                            **opciones)
