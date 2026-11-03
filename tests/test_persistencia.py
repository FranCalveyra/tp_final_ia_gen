"""Persistencia y locks con dobles; ninguna llamada a Modal ni entrenamiento."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from admision_tp.persistencia import crear_paquete, ejecutar_persistente, validar_paquete
from admision_tp.preparacion import _canonical, _sha, preparar_corrida
from admision_tp.tokenizacion import tokenizar_preparacion
from test_conversaciones import registro
from test_tokenizacion import TokenizadorArtificial


class Locks(dict):
    def put(self, key, value, *, skip_if_exists):
        if key in self:
            return False
        self[key] = value
        return True


class Volume:
    def __init__(self):
        self.eventos = []
    def reload(self):
        self.eventos.append("reload")
    def commit(self):
        self.eventos.append("commit")


class PersistenciaTests(unittest.TestCase):
    def paquete(self, raiz):
        datos, config = raiz / "datos.jsonl", raiz / "config.json"
        datos.write_text(json.dumps(registro()) + "\n")
        config.write_bytes(Path("configuraciones/sft-ejemplo.json").read_bytes())
        preparar_corrida(datos, config, raiz / "preparada")
        tokenizar_preparacion(raiz / "preparada", raiz / "tokens", TokenizadorArtificial())
        return crear_paquete(raiz / "preparada", raiz / "tokens")

    def test_paquete_solo_allowlist_valida_y_rechaza_codigo_diferente(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            paquete = self.paquete(raiz)
            validar_paquete(paquete)
            for cambio in ("../escape", "codigo", "datos"):
                malo = copy.deepcopy(paquete)
                if cambio == "codigo":
                    malo["codigo"]["archivos"]["uv.lock"] = "f" * 64
                elif cambio == "datos":
                    malo["archivos"]["tokens/tokens.jsonl"] += b"alterado"
                else:
                    malo["archivos"][cambio] = b"texto"
                with self.assertRaises(ValueError):
                    validar_paquete(malo)
            original = raiz / "preparada/manifiesto.json"
            contenido = original.read_bytes()
            original.unlink()
            destino = raiz / "enlace.json"
            destino.write_bytes(contenido)
            original.symlink_to(destino)
            with self.assertRaises(ValueError):
                crear_paquete(raiz / "preparada", raiz / "tokens")

    def test_recarga_persiste_entradas_y_cierre_sin_git_remoto(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            paquete, volume, locks = self.paquete(raiz), Volume(), Locks()
            montaje = raiz / "volume"
            montaje.mkdir()
            def backend(*args):
                self.assertEqual(volume.eventos, ["reload", "commit", "commit"])
                self.assertEqual(args[-2]["codigo"], paquete["codigo"])
                return {"run_id": "simulado"}
            with patch("admision_tp.entrenamiento._codigo", side_effect=AssertionError("sin .git remoto")):
                estado = ejecutar_persistente(paquete, "corrida_1", montaje, volume, locks, backend=backend)
            self.assertEqual(estado["estado"], "finalizada")
            self.assertEqual(volume.eventos, ["reload", "commit", "commit", "commit"])
            self.assertEqual(locks, {})
            self.assertEqual((montaje / "corrida_1/transferencia/tokens/tokens.jsonl").read_bytes(),
                             paquete["archivos"]["tokens/tokens.jsonl"])
            with self.assertRaises(FileExistsError):
                ejecutar_persistente(paquete, "corrida_1", montaje, volume, locks, backend=backend)

    def test_lock_existente_no_se_borra_y_nombres_se_rechazan(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            paquete, volume, locks = self.paquete(raiz), Volume(), Locks(corrida="otro")
            with self.assertRaises(ValueError):
                ejecutar_persistente(paquete, "corrida", raiz, volume, locks)
            self.assertEqual(locks["corrida"], "otro")
            self.assertEqual(volume.eventos, [])
            for nombre, checkpoint in (("../otra", None), ("bien", "../checkpoint-50")):
                with self.assertRaises(ValueError):
                    ejecutar_persistente(paquete, nombre, raiz, volume, locks, checkpoint=checkpoint)

    def test_error_persiste_diagnostico_y_libera_lock(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            paquete, volume, locks = self.paquete(raiz), Volume(), Locks()
            def falla(*args):
                raise RuntimeError("simulado")
            with self.assertRaises(RuntimeError):
                ejecutar_persistente(paquete, "fallida", raiz, volume, locks, backend=falla)
            self.assertEqual(locks, {})
            self.assertEqual(volume.eventos, ["reload", "commit", "commit", "commit"])
            self.assertEqual(json.loads((raiz / "fallida/entrenamiento/estado.json").read_text()),
                             {"estado": "fallida", "error_tipo": "RuntimeError"})

    def test_reanuda_mismo_checkpoint_rechaza_cambios(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            paquete, volume, locks = self.paquete(raiz), Volume(), Locks()
            ejecutar_persistente(paquete, "corrida", raiz, volume, locks, backend=lambda *a: {})
            salida = raiz / "corrida/entrenamiento"
            cp = salida / "checkpoint-50"
            cp.mkdir()
            for nombre in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                           "adapter_config.json", "adapter_model.safetensors"):
                (cp / nombre).write_bytes(b"simulado")
            (cp / "corrida.json").write_bytes((salida / "corrida.json").read_bytes())
            (cp / "integridad.json").write_bytes(_canonical({p.name: _sha(p.read_bytes()) for p in cp.iterdir()}))
            def backend(*args):
                self.assertEqual(args[-1], cp)
                return {}
            ejecutar_persistente(paquete, "corrida", raiz, volume, locks,
                                checkpoint="checkpoint-50", backend=backend)
            alterado = copy.deepcopy(paquete)
            alterado["codigo"]["commit"] = "b" * 40
            with self.assertRaises(ValueError):
                ejecutar_persistente(alterado, "corrida", raiz, volume, locks,
                                    checkpoint="checkpoint-50", backend=backend)

    def test_publicar_exige_online_antes_de_tocar_lock_o_volume(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            paquete, volume, locks = self.paquete(raiz), Volume(), Locks()
            with self.assertRaises(ValueError):
                ejecutar_persistente(paquete, "corrida", raiz, volume, locks, publicar=True,
                                    backend=lambda *a: {})
            self.assertEqual((volume.eventos, locks, list(raiz.glob("corrida*"))), ([], {}, []))

    def test_publicar_llega_al_entrenador_y_la_reanudacion_conserva_recibos(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            paquete, volume, locks = self.paquete(raiz), Volume(), Locks()
            recibo = [{"checkpoint": "checkpoint-50", "hf_repo": "o/r", "hf_revision": "a" * 40}]
            def backend(*args):
                auditoria, salida = args[-2], args[-3]
                self.assertTrue(auditoria["publicar"])
                (salida / "publicaciones.json").write_bytes(_canonical(recibo))
                cp = salida / "checkpoint-50"
                cp.mkdir()
                for n in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                          "adapter_config.json", "adapter_model.safetensors"):
                    (cp / n).write_bytes(b"simulado")
                (cp / "corrida.json").write_bytes(_canonical(auditoria))
                (cp / "integridad.json").write_bytes(_canonical({p.name: _sha(p.read_bytes()) for p in cp.iterdir()}))
                return {"publicaciones": recibo}
            estado = ejecutar_persistente(paquete, "corrida", raiz, volume, locks,
                                         online=True, publicar=True, backend=backend)
            self.assertEqual(estado["publicaciones"], recibo)
            self.assertEqual(volume.eventos[-1], "commit")  # Recibos confirmados en el Volume.
            self.assertEqual(json.loads((raiz / "corrida/entrenamiento/publicaciones.json").read_text()), recibo)
            def reanuda(*args):
                self.assertEqual(json.loads((args[-3] / "publicaciones.json").read_text()), recibo)
                return {}
            ejecutar_persistente(paquete, "corrida", raiz, volume, locks, online=True, publicar=True,
                                checkpoint="checkpoint-50", backend=reanuda)
            for modo in ({"online": True}, {"online": True, "publicar": False}):
                with self.assertRaises(ValueError):  # No cambiar el modo al reanudar.
                    ejecutar_persistente(paquete, "corrida", raiz, volume, locks,
                                        checkpoint="checkpoint-50", backend=reanuda, **modo)


if __name__ == "__main__":
    unittest.main()
