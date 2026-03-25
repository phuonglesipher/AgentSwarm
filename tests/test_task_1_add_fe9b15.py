import importlib

def test_gameplay_change_smoke():
    for name in ['solution', 'main', 'gameplay_change_summary']:
        module = importlib.import_module(name)
        if hasattr(module, 'build_gameplay_change_summary'):
            result = module.build_gameplay_change_summary()
            assert result['implementation_status'] == 'ready-for-review'
            return
    raise AssertionError('builder not found')