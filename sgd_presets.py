"""Permalink-Sync (Query-Parameter <-> Session-State) und Presets."""
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

import sgd_constants as C


@dataclass(frozen=True)
class SettingSpec:
    key: str
    param: str
    default: Any
    cast: Callable[[str], Any]
    bounds: tuple | None = None


SETTING_SPECS = [
    SettingSpec("sigma", "sigma", C.SIGMA_DEFAULT, float, (C.SIGMA_MIN, C.SIGMA_MAX)),
]


def init_session_state_defaults() -> None:
    for spec in SETTING_SPECS:
        if spec.key not in st.session_state:
            st.session_state[spec.key] = spec.default


def load_permalink_settings() -> None:
    params = st.query_params
    for spec in SETTING_SPECS:
        if spec.param in params and spec.key not in st.session_state:
            raw = params[spec.param]
            try:
                value = spec.cast(raw)
            except (TypeError, ValueError):
                continue
            if spec.bounds is not None:
                lo, hi = spec.bounds
                value = min(max(value, lo), hi)
            st.session_state[spec.key] = value


def sync_query_params(values: dict) -> None:
    for spec in SETTING_SPECS:
        if spec.key in values:
            st.query_params[spec.param] = str(values[spec.key])


def store_from_widget(key: str) -> None:
    st.session_state[key] = st.session_state[f"widget_{key}"]


def apply_preset(preset_key: str) -> None:
    preset = C.PRESETS[preset_key]
    if "sigma" in preset:
        st.session_state["sigma"] = preset["sigma"]
        st.session_state["widget_sigma"] = preset["sigma"]
