import json
from importlib.metadata import version

from openqml.cli import main


def test_doctor_and_cli_conversion(tmp_path, capsys):
    assert main(["doctor"]) == 0
    assert json.loads(capsys.readouterr().out)["openqml"] == version("openqml")
    source, output = tmp_path / "bell.qasm", tmp_path / "bell.json"
    source.write_text("OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0],q[1];")
    argv = ["convert", str(source), "--from", "openqasm3", "--to", "json", "-o", str(output)]
    assert main(argv) == 0
    assert json.loads(output.read_text())["num_qubits"] == 2
    assert main(argv) == 2
    assert main(argv + ["--force"]) == 0


def test_cli_inspection_optimization_and_diagnostics(tmp_path, capsys):
    from openqml import Circuit

    source, destination = tmp_path / "circuit.json", tmp_path / "optimized.json"
    source.write_text(Circuit(2).h(0).x(1).x(1).crx(0.43, 0, 1).to_json())
    assert main(["inspect", str(source), "--from", "json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["gates"] == 4 and inspected["num_qubits"] == 2
    assert (
        main(
            [
                "optimize",
                str(source),
                "--from",
                "json",
                "--basis",
                "p,ry,cz",
                "-o",
                str(destination),
            ]
        )
        == 0
    )
    optimized = Circuit.from_json(destination.read_text())
    assert {op.name for op in optimized.operations} <= {"p", "ry", "cz"}
    assert (
        main(["convert", str(source), "--from", "json", "--to", "openqasm3", "--diagnostics"]) == 0
    )
    captured = capsys.readouterr()
    assert captured.out.startswith("OPENQASM 3.0;")
    assert json.loads(captured.err)[0]["code"] == "EXACT_GATE_DECOMPOSITION"
