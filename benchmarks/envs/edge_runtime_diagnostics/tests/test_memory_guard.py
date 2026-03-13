from src.memory_guard import should_pause_for_memory


def test_should_pause_for_memory():
    assert should_pause_for_memory(512, 1024) is True
    assert should_pause_for_memory(2048, 1024) is False
