"""Generación local por lotes; no conoce ni ejecuta herramientas."""

from .configuracion import Configuracion


class ModeloLocal:
    def __init__(self, modelo, procesador, configuracion: Configuracion):
        self.modelo, self.procesador, self.configuracion = modelo, procesador, configuracion

    @classmethod
    def cargar(cls, configuracion: Configuracion):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("El baseline requiere una GPU CUDA; no se carga en CPU")
        from transformers import AutoModelForCausalLM, AutoProcessor
        procesador = AutoProcessor.from_pretrained(
            configuracion.modelo, revision=configuracion.revision, padding_side="left")
        modelo = AutoModelForCausalLM.from_pretrained(
            configuracion.modelo, revision=configuracion.revision,
            dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        if configuracion.adapter_repo:
            modelo = cls._fusionar_adapter(modelo, configuracion)
        modelo.eval()
        return cls(modelo, procesador, configuracion)

    @staticmethod
    def _fusionar_adapter(modelo, configuracion: Configuracion):
        """Aplica el LoRA y lo fusiona en memoria: la inferencia no paga capas extra."""
        from peft import PeftConfig, PeftModel

        origen = {"revision": configuracion.adapter_revision,
                  "subfolder": configuracion.adapter_subcarpeta or None}
        config = PeftConfig.from_pretrained(configuracion.adapter_repo, **origen)
        # `config.revision` es la revisión de la base con la que se entrenó el adapter.
        if (config.base_model_name_or_path, config.revision) != (configuracion.modelo, configuracion.revision):
            raise ValueError("El adapter fue entrenado sobre otra base o revisión")
        peft = PeftModel.from_pretrained(modelo, configuracion.adapter_repo, config=config, **origen)
        return peft.merge_and_unload()

    def __call__(self, conversaciones: list[list[dict]]) -> list[str]:
        if not conversaciones:
            return []
        import torch

        respuestas = []
        for inicio in range(0, len(conversaciones), self.configuracion.lote):
            lote = conversaciones[inicio:inicio + self.configuracion.lote]
            entradas = self.procesador.apply_chat_template(
                lote, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                padding=True, truncation=False, return_dict=True, return_tensors="pt")
            ancho = entradas["input_ids"].shape[-1]
            if ancho + self.configuracion.max_tokens_nuevos > self.configuracion.max_contexto:
                raise ValueError("Contexto excedido: no se recorta la conversación silenciosamente")
            entradas = entradas.to(self.modelo.device)
            with torch.inference_mode():
                salida = self.modelo.generate(**entradas, do_sample=False,
                    max_new_tokens=self.configuracion.max_tokens_nuevos)
            textos = self.procesador.batch_decode(salida[:, ancho:], skip_special_tokens=True)
            if len(textos) != len(lote):
                raise RuntimeError("El modelo no devolvió una respuesta por conversación")
            respuestas.extend(textos)
        return respuestas
