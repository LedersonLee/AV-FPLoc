"""Package-only entry points; no datasets, renderer, or original workspace needed."""
import ast
import importlib
from pathlib import Path
import subprocess
import sys

import avfploc.preprocessing.hssd as package


def test_all_cli_help_entrypoints():
    root = Path(package.__file__).parent
    for path in sorted(root.glob('*.py')):
        tree = ast.parse(path.read_text())
        if not any(isinstance(n, ast.If) and '__name__' in ast.unparse(n.test) for n in tree.body):
            continue
        result = subprocess.run(
            [sys.executable, '-m', package.__name__ + '.' + path.stem, '--help'],
            capture_output=True, text=True, timeout=20,
        )
        assert result.returncode == 0, (path.name, result.stderr)


def test_renderer_adapter_uses_packaged_scheduler():
    from avfploc.acoustic.render_adapter import positive_tools, supervisor_tools
    assert positive_tools().__name__ == package.__name__ + '.hssd_positive_pairs'
    assert supervisor_tools().__name__ == package.__name__ + '.hssd_rir_supervisor'


def test_builder_requires_explicit_split(monkeypatch):
    import pytest
    builder = importlib.import_module(package.__name__ + '.build_hssd_unloc_dataset')
    monkeypatch.setattr(sys, 'argv', ['builder', '--hssd-root', '/unused', '--output-root', '/unused'])
    with pytest.raises(SystemExit) as error:
        builder.parse_args()
    assert error.value.code == 2


def test_builder_refuses_existing_output(tmp_path, monkeypatch):
    import json
    import pytest
    builder = importlib.import_module(package.__name__ + '.build_hssd_unloc_dataset')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train': ['a'], 'val': [], 'test': []}))
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('original')
    monkeypatch.setattr(builder, 'scan_semantic_scenes', lambda p: (['a'], []))
    monkeypatch.setattr(sys, 'argv', ['builder', '--hssd-root', str(tmp_path),
        '--output-root', str(output), '--split-manifest', str(split)])
    with pytest.raises(FileExistsError):
        builder.main()
    assert marker.read_text() == 'original'
