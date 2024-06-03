import logging
import os
from typing import Any

import pandas as pd
import streamlit as st
import vertexai
from countryinfo import CountryInfo
from dotenv import load_dotenv

from common import HintType, configs, get_distance
from hint import AudioHint, ImageHint, TextHint, TextHintVertex


def setup_models(_cache: Any, configs: dict) -> None:
    """Setups all hint models.

    Args:
        _cache (st.session_state): Streamlit cache object
        configs (dict): Configurations used by the models
    """
    for model_type in _cache["hint_types"]:
        if _cache["model"][model_type] is None:
            if model_type == HintType.TEXT.value:
                _cache["model"][model_type] = setup_text_hint(configs)
            elif model_type == HintType.IMAGE.value:
                _cache["model"][model_type] = setup_image_hint(configs)
            elif model_type == HintType.AUDIO.value:
                _cache["model"][model_type] = setup_audio_hint(configs)


@st.cache_resource()
def setup_text_hint(configs: dict) -> TextHint | TextHintVertex:
    """Setups the text hint model.

    Args:
        configs (dict): Configurations used by the model

    Returns:
        TextHint | TextHintVertex: Hint model
    """
    with st.spinner("Loading text model..."):
        if configs["vertex"]["to_use"]:
            model_configs = configs["vertex"][HintType.TEXT.value.lower()]
            if not st.session_state["vertex_initialized"]:
                setup_vertex(
                    configs["vertex"]["project"],
                    configs["vertex"]["location"],
                )
            textHint = TextHintVertex(configs=model_configs)
        else:
            model_configs = configs["local"][HintType.TEXT.value.lower()]
            model_configs["hf_access_token"] = os.environ["HF_ACCESS_TOKEN"]
            textHint = TextHint(configs=model_configs)
        textHint.initialize()
    return textHint


@st.cache_resource()
def setup_image_hint(configs: dict) -> ImageHint:
    """Setups the image hint model.

    Args:
        configs (dict): Configurations used by the model

    Returns:
        ImageHint: Hint model
    """
    with st.spinner("Loading image model..."):
        model_configs = configs["local"][HintType.IMAGE.value.lower()]
        imageHint = ImageHint(configs=model_configs)
        imageHint.initialize()
    return imageHint


@st.cache_resource()
def setup_audio_hint(configs: dict) -> AudioHint:
    """Setups the audio hint model.

    Args:
        configs (dict): Configurations used by the model

    Returns:
        AudioHint: Hint model
    """
    with st.spinner("Loading audio model..."):
        model_configs = configs["local"][HintType.AUDIO.value.lower()]
        audioHint = AudioHint(configs=model_configs)
        audioHint.initialize()
    return audioHint


@st.cache_resource()
def setup_vertex(project: str, location: str) -> None:
    """Setups the Vertex AI project.

    Args:
        project (str): Vertex AI project name
        location (str): Vertex AI project location
    """
    vertexai.init(project=project, location=location)
    logger.info("Vertex AI setup finished")


@st.cache_resource()
def get_country_list() -> pd.DataFrame:
    """Builds a database of countries and metadata.

    Returns:
        pd.DataFrame: Country database
    """
    country_list = list(CountryInfo().all().keys())

    country_df = {}
    for country in country_list:
        try:
            area = CountryInfo(country).area()
            country_df[country] = area
        except:
            pass

    country_df = pd.DataFrame(country_df.items(), columns=["country", "area"])
    return country_df


def pick_country(country_df: pd.DataFrame) -> str:
    """Selects a country, the probability of each country is related to its area size.

    Args:
        country_df (pd.DataFrame): Database of country and their metadata

    Returns:
        str: The selected country
    """
    country = country_df.sample(n=1, weights="area")["country"].iloc[0]
    return country


def reset_cache() -> None:
    """Reset the Streamlit APP cache."""
    country_df = get_country_list()
    st.session_state["country_list"] = country_df["country"].values.tolist()
    st.session_state["country"] = pick_country(country_df)
    st.session_state["hint_types"] = []
    st.session_state["n_hints"] = 1
    st.session_state["game_started"] = False
    st.session_state["vertex_initialized"] = False
    st.session_state["model"] = {
        HintType.TEXT.value: None,
        HintType.IMAGE.value: None,
        HintType.AUDIO.value: None,
    }


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Gen AI GeoGuesser",
    page_icon="🌎",
)

if not st.session_state:
    load_dotenv()
    reset_cache()

st.title("Generative AI GeoGuesser 🌎")

st.markdown("### Guess the country based on hints generated by AI")

col1, col2 = st.columns([2, 1])

with col1:
    st.session_state["hint_types"] = st.multiselect(
        "Chose which hint types you want",
        [x.value for x in HintType],
        default=st.session_state["hint_types"],
    )

with col2:
    st.session_state["n_hints"] = st.slider(
        "Number of hints",
        min_value=1,
        max_value=5,
        value=st.session_state["n_hints"],
    )

start_btn = st.button("Start game")

if start_btn:
    if not st.session_state["hint_types"]:
        st.error("Pick at least one hint type")
        reset_cache()
    else:
        print(f'Chosen country "{st.session_state["country"]}"')

        setup_models(st.session_state, configs)

        for hint_type in st.session_state["hint_types"]:
            with st.spinner(f"Generating {hint_type} hint..."):
                st.session_state["model"][hint_type].generate_hint(
                    st.session_state["country"],
                    st.session_state["n_hints"],
                )

        st.session_state["game_started"] = True

if st.session_state["game_started"]:
    game_col1, game_col2, game_col3 = st.columns([2, 1, 1])

    with game_col1:
        guess = st.selectbox("Country guess", ([""] + st.session_state["country_list"]))
    with game_col2:
        guess_btn = st.button("Make a guess")
    with game_col3:
        reset_btn = st.button("Reset game")

    if guess_btn:
        if st.session_state["country"] == guess:
            st.success("Correct guess you won!")
            st.balloons()
        else:
            if guess:
                country_latlong = CountryInfo(st.session_state["country"]).latlng()
                guess_latlong = CountryInfo(guess).latlng()
                distance = int(get_distance(country_latlong, guess_latlong))
                st.error(
                    f"""
                    Wrong guess, you missed the correct country by {distance} KM.
                    The correct answer was {st.session_state["country"]}.
                    """
                )
            else:
                st.error("Pick a country.")

    if reset_btn:
        reset_cache()

if st.session_state["game_started"]:
    tabs = st.tabs([f"{x} hint" for x in st.session_state["hint_types"]])

    for tab_idx, tab in enumerate(tabs):
        hint_type = st.session_state["hint_types"][tab_idx]
        with tab:
            if st.session_state["model"][hint_type]:
                for hint_idx, hint in enumerate(
                    st.session_state["model"][hint_type].hints
                ):
                    st.markdown(f"#### Hint #{hint_idx+1}")
                    if hint_type == HintType.TEXT.value:
                        st.write(hint["text"])
                    elif hint_type == HintType.IMAGE.value:
                        st.image(hint["image"])
                    elif hint_type == HintType.AUDIO.value:
                        st.audio(hint["audio"], sample_rate=hint["sample_rate"])
