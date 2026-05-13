"""
Path A v2 helpers: initial-prompt hotwords + output post-processing.

Used by streaming demo and eval scripts to bias the decoder toward
Nigerian proper nouns and to normalize trivial output formatting.
"""
import re

# Single-paragraph "context" sentence in the target domain. Whisper's
# initial_prompt is interpreted as prior speech context — keep it
# natural-sounding, ~50–200 tokens. Includes the proper nouns we saw
# the v1 model miss + the canonical Nigerian Pidgin function-word set.
INITIAL_PROMPT = (
    "dis na bbc news pidgin tori about buhari tinubu atiku saraki "
    "tony nwoye femi otedola akinwunmi ambode oseloka henry obaze "
    "zainab balogun jimoh moshood and aisha for nigeria politics "
    "for states like lagos anambra delta kogi niger abuja kano rivers "
    "edo ogun salford and offa with organizations like apc pdp nema "
    "jamb frsc brt jp morgan and wikipedia pipo dey tok say di dey na "
    "wey pikin tori sabi hapun sometin anytin becos redi alredi neva "
    "dem una abi oga chillax snakebite"
)


_DIGIT_PAIR = re.compile(r"(\d) (\d)")
_PUNCT = re.compile(r"[.,!?;:\"]")          # training labels have none of these
_INTRA_NUM_COMMA = re.compile(r"(\d),(\d)")  # "60,000" -> "60000"


def postprocess(text: str) -> str:
    """Apply formatting normalizations that match the training-label conventions."""
    # Strip commas inside numbers first ("60,000" -> "60000").
    while True:
        new = _INTRA_NUM_COMMA.sub(r"\1\2", text)
        if new == text:
            break
        text = new
    # Drop punctuation that the labels don't use.
    text = _PUNCT.sub("", text)
    # Merge digit groups separated by single spaces ("60 000" -> "60000").
    while True:
        new = _DIGIT_PAIR.sub(r"\1\2", text)
        if new == text:
            break
        text = new
    text = re.sub(r" +", " ", text).strip()
    return text


def transcribe(model, audio, *, use_hotwords: bool = True, use_postprocess: bool = True) -> str:
    """Run faster-whisper with the v2 decode pipeline."""
    kwargs = {"language": "en", "task": "transcribe", "beam_size": 1}
    if use_hotwords:
        kwargs["initial_prompt"] = INITIAL_PROMPT
    segments, _ = model.transcribe(audio, **kwargs)
    text = "".join(s.text for s in segments).strip().lower()
    if use_postprocess:
        text = postprocess(text)
    return text
