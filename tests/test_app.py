from streamlit.testing.v1 import AppTest

_APP_TIMEOUT = 120


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=_APP_TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Über mich" in c for c in captions)


def test_preset_rauschfrei():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Rauschfrei"][0]
    btn.click().run()
    assert not at.exception


def test_preset_starkes_rauschen():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Starkes Rauschen"][0]
    btn.click().run()
    assert not at.exception


def test_slider_extreme_values_do_not_crash():
    at = _fresh()
    sigma_slider = [s for s in at.slider if s.label == "Rauschstärke σ"][0]
    sigma_slider.set_value(sigma_slider.max).run()
    assert not at.exception
    sigma_slider.set_value(sigma_slider.min).run()
    assert not at.exception
