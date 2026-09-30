from src import config


def test_possession_defaults():
    assert config.POSSESSION_MAX_DISTANCE_PX == 80
    assert config.POSSESSION_SMOOTHING_WINDOW == 5


def test_device_is_known():
    assert config.DEVICE in {"cuda", "mps", "cpu"}


def test_roboflow_assets_present():
    # setup.sh from the Roboflow sports repo must have been run.
    assert config.PLAYER_DETECTION_MODEL_PATH.stat().st_size > 0
    assert config.BALL_DETECTION_MODEL_PATH.stat().st_size > 0
    assert config.DEFAULT_SOURCE_VIDEO.stat().st_size > 0
