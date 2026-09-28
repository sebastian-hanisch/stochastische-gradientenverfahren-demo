import sgd_constants as C


def test_all_presets_have_valid_sigma():
    for key, preset in C.PRESETS.items():
        assert C.SIGMA_MIN <= preset["sigma"] <= C.SIGMA_MAX


def test_preset_rauschfrei_is_zero():
    assert C.PRESETS["rauschfrei"]["sigma"] == 0.0


def test_preset_starkes_rauschen_is_positive():
    assert C.PRESETS["starkes_rauschen"]["sigma"] > 0.0
