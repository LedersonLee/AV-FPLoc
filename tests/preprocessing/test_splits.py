import pytest
from avfploc.preprocessing.hssd.splits import validate_split, load_split


def test_preserves_split_order():
    split = {"train": ["b", "a"], "val": ["c"], "test": ["d"]}
    assert validate_split(split) == split


@pytest.mark.parametrize("split", [
    {"train": ["a"], "val": ["a"], "test": []},
    {"train": ["a", "a"], "val": [], "test": []},
    {"train": ["a"], "val": []},
])
def test_rejects_invalid_split(split):
    with pytest.raises(ValueError):
        validate_split(split)


def test_load_existing_split(tmp_path):
    path = tmp_path / "split.json"
    path.write_text('{"train":["a"],"val":["b"],"test":["c"]}')
    assert load_split(path)["test"] == ["c"]
