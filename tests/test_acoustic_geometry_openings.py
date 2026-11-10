import json
from track_b.acoustic_geometry.render_pairs import preserve_openings


def test_opening_control_preserves_transforms_and_stage(tmp_path):
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'objects': {'paths': {'.json': ['objects/openings/door.object_config.json', 'objects/furniture/chair.object_config.json']}}, 'stages': {'paths': {'.json': ['stage.json']}}, 'scene_instances': {'paths': {}}, 'semantic_scene_descriptor_instances': {}}))
    door = {'template_name': 'objects/door', 'translation': [1, 2, 3], 'rotation': [0, 0, 0, 1]}
    scene = {'stage_instance': {'template_name': 'stages/example'}, 'object_instances': [door, {'template_name': 'chair'}]}
    instance, config, count = preserve_openings(scene, registry, tmp_path / 'control')
    actual = json.loads(instance.read_text())
    assert count == 1 and actual['object_instances'] == [door]
    assert actual['stage_instance'] == scene['stage_instance']
    assert len(scene['object_instances']) == 2
    assert json.loads(config.read_text())['objects']['paths']['.json'][0] == str(tmp_path / 'objects/openings/door.object_config.json')
