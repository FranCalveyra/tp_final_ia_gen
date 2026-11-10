"""Estado Git del manifiesto de conversaciones: la propia salida no ensucia el árbol."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.generar_conversaciones import _git


def _repo(raiz: Path) -> None:
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=raiz, check=True)
    (raiz / "codigo.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=raiz, check=True)
    subprocess.run(["git", "commit", "-qm", "inicial"], cwd=raiz, check=True)


class EstadoGitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name)
        _repo(self.raiz)
        self.salida = self.raiz / "datos" / "conversaciones" / "v2-baseline"

    def tearDown(self):
        self.tmp.cleanup()

    def test_salida_y_su_temporal_no_cuentan_como_cambios(self):
        self.salida.mkdir(parents=True)
        (self.salida / "conversaciones.jsonl").write_text("{}\n", encoding="utf-8")
        temporal = self.salida.parent / ".v2-baseline-abc123"
        temporal.mkdir()
        (temporal / "auditoria.jsonl").write_text("{}\n", encoding="utf-8")
        self.assertIs(_git(self.raiz, self.salida)["dirty"], False)

    def test_otros_cambios_si_cuentan(self):
        self.salida.mkdir(parents=True)
        (self.raiz / "nuevo.py").write_text("y = 2\n", encoding="utf-8")
        self.assertIs(_git(self.raiz, self.salida)["dirty"], True)
        (self.raiz / "nuevo.py").unlink()
        (self.raiz / "codigo.py").write_text("x = 2\n", encoding="utf-8")
        self.assertIs(_git(self.raiz, self.salida)["dirty"], True)

    def test_otra_version_de_datos_sin_versionar_si_cuenta(self):
        otra = self.salida.parent / "v2-prefijos"
        otra.mkdir(parents=True)
        (otra / "conversaciones.jsonl").write_text("{}\n", encoding="utf-8")
        self.assertIs(_git(self.raiz, self.salida)["dirty"], True)


if __name__ == "__main__":
    unittest.main()
