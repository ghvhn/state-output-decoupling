"""Recognize explicit comparison structure, without validating its meaning.

This conservative text check distinguishes a stated spectrum contrast from
independent memory recaps. It does not establish that the contrast is true,
causal, or a good name for a probe. No semantic seed or model prompt is used.
"""

import math
import re


_MEMORY_GROUP = re.compile(
    r"\bmemory\s*#?\s*\d+\b(?:\s*(?:,\s*#?\s*\d+|and\s*(?:memory\s*)?#?\s*\d+)\b)*",
    re.IGNORECASE)
_LOW = re.compile(r"\b(?:lower|low)(?:[-\s]+(?:readings?|positions?|scores?|values?|ends?|sides?|poles?))?\b", re.IGNORECASE)
_HIGH = re.compile(r"\b(?:higher|high)(?:[-\s]+(?:readings?|positions?|scores?|values?|ends?|sides?|poles?))?\b", re.IGNORECASE)
_NEGATIVE = re.compile(r"\bnegative[-\s]+(?:sides?|readings?|positions?|scores?|ends?|poles?)\b", re.IGNORECASE)
_POSITIVE = re.compile(r"\bpositive[-\s]+(?:sides?|readings?|positions?|scores?|ends?|poles?)\b", re.IGNORECASE)
_RELATION = re.compile(
    r"\b(?:whereas|while|but|unlike|versus|vs\.?|contrast\w*|compar\w*|"
    r"differ\w*|rather\s+than|instead\s+of|more|less|stronger|weaker|greater|fewer|shift\w*|switch\w*|"
    r"transition\w*|move\w*|change\w*)\b", re.IGNORECASE)
_DIRECTIONAL = re.compile(
    r"\b(?:as|when|with)\s+(?:(?:the|measured|probe|centered)\s+)*"
    r"(?:readings?|positions?|scores?|values?)\s+"
    r"(?:increase\w*|decrease\w*|rise\w*|fall\w*|grow\w*|become\s+(?:higher|lower))\b",
    re.IGNORECASE)
_TRANSITION = re.compile(r"\bfrom\b.{1,500}\bto\b|\b(?:more|less|become\w*|shift\w*|switch\w*|transition\w*|move\w*)\b", re.IGNORECASE)
_REFERENCE_PREFIX = re.compile(
    r"\b(?:unlike|compared\s+(?:with|to)|(?:in\s+)?contrast\s+(?:between|with|to))\s*$",
    re.IGNORECASE)
_JOINT_DIFFERENCE = re.compile(r"^\s*(?:differ\w*|contrast\w*)\b", re.IGNORECASE)
_HEADING = re.compile(
    r"^(?:(?:negative|positive|lower|higher)\s+(?:side|end)(?:\s+memories)?|"
    r"key\s+points|open\s+questions|summary|comparison|contrast|definition)\s*:?$",
    re.IGNORECASE)

# Remove comparison scaffolding when checking for descriptive content. This
# is deliberately not a required vocabulary of actions, emotions, or traits.
_SCAFFOLD = set("""
    a an the i me my we our us you your it its they their them he his she her
    is am are was were be being been do does did has have had can could may
    might would should will of in on at by for with from to and or not as
    that this these those show shows shown showing mean means refer refers
    appear appears seem seems memory memories reading readings position
    positions score scores value values measured centered spectrum probe
    axis side sides end ends pole poles lower low higher high negative positive
    both each all same shared topic topics whereas while but unlike versus vs
    compare compares comparison compared contrasting contrast contrasts
    differ differs different difference rather than instead more less
    shift shifts shifted change changes changed switch switches transition
    transitions move moves model ai conversational language
""".split())


def _blocks(text):
    text = re.sub(r"<<.*?(?:>>|$)", "", text[:16000], flags=re.DOTALL)
    paragraphs = []
    current = []

    def flush():
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for line in text.splitlines():
        stripped = line.strip()
        plain = re.sub(r"[*_`#]", "", stripped).strip()
        if not stripped or _HEADING.fullmatch(plain) or stripped.startswith("#"):
            flush()
            continue
        # Independent bullet/list items are not one connected comparison.
        if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", stripped):
            flush()
            paragraphs.append(re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", plain))
        else:
            current.append(plain)
    flush()
    return paragraphs


def _content_words(block):
    text = _MEMORY_GROUP.sub("", block)
    text = re.sub(r"\bposition\s*=\s*[+-]?[\d.]+", "", text, flags=re.IGNORECASE)
    return {word for word in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
            if len(word) > 2 and word not in _SCAFFOLD}


def _memory_references(block):
    return {int(value) for match in _MEMORY_GROUP.finditer(block)
            for value in re.findall(r"\d+", match.group(0))}


def _reference_comparison(block, scope):
    """Comparison words must connect referenced ends, not another topic."""
    groups = [(match, _memory_references(match.group(0)))
              for match in _MEMORY_GROUP.finditer(block)]

    def opposing(left, right):
        if scope is None:
            return bool(left - right) and bool(right - left)
        return ((bool(left & scope["lower"]) and bool(right & scope["upper"])) or
                (bool(left & scope["upper"]) and bool(right & scope["lower"])))

    for index, (left_match, left_ids) in enumerate(groups):
        for right_match, right_ids in groups[index + 1:]:
            if not opposing(left_ids, right_ids):
                continue
            between = block[left_match.end():right_match.start()]
            prefix = block[max(0, left_match.start() - 80):left_match.start()]
            if _RELATION.search(between) or _REFERENCE_PREFIX.search(prefix):
                return True
        # A joint subject can state a difference directly: "Memory 4 and
        # 55 differ...". Merely mentioning both memories before discussing
        # differences between some other objects does not qualify.
        joint_ends = ((bool(left_ids & scope["lower"]) and bool(left_ids & scope["upper"]))
                      if scope is not None else len(left_ids) >= 2)
        if joint_ends and _JOINT_DIFFERENCE.search(block[left_match.end():]):
            return True
    return False


def _selected_scope(selected):
    if selected is None:
        return None
    readings = {}
    for index, row in selected:
        value = row.get("reading")
        if isinstance(value, bool):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            readings[int(index)] = value
    ordered = sorted(readings, key=lambda index: (readings[index], index))
    midpoint = len(ordered) // 2
    return {
        "readings": readings,
        "lower": set(ordered[:midpoint]),
        "upper": set(ordered[midpoint:]),
        "varying": bool(ordered) and readings[ordered[-1]] > readings[ordered[0]],
        "negative": any(value < 0 for value in readings.values()),
        "positive": any(value > 0 for value in readings.values()),
    }


def assess_definition_attempt(text, selected=None):
    """Return whether the text articulates a substantive, connected contrast.

    ``selected`` supplies actual ``(memory_index, row)`` examples. Referenced
    IDs must then exist, and lower/upper groups follow measured ordering,
    including spectra entirely on one side of zero. With ``selected=None``
    legacy summaries can only be checked for structure, not memory grounding.
    """
    text = str(text or "").strip()
    if not text:
        return {"accepted": False, "reason": "No generated interpretation."}
    blocks = _blocks(text)
    scope = _selected_scope(selected)
    references = set().union(*(_memory_references(block) for block in blocks))
    if scope is not None:
        if references - scope["readings"].keys():
            return {"accepted": False, "reason": "Memory references do not match supplied examples."}
        if not scope["varying"] or not scope["lower"] or not scope["upper"]:
            return {"accepted": False, "reason": "Supplied examples do not establish varying measured ends."}

    absent_side = False
    for block in blocks:
        negative = bool(_NEGATIVE.search(block))
        positive = bool(_POSITIVE.search(block))
        if scope is not None and ((negative and not scope["negative"]) or
                                  (positive and not scope["positive"])):
            absent_side = True
            continue
        # Two distinct descriptive words keep a bare assertion of a
        # difference from passing. Their meanings are not judged here.
        if len(_content_words(block)) < 2:
            continue
        named_ends = (bool(_LOW.search(block)) and bool(_HIGH.search(block))) or (negative and positive)
        explicit_pair = ((named_ends and bool(_RELATION.search(block))) or
                         _reference_comparison(block, scope))
        directional = bool(_DIRECTIONAL.search(block)) and bool(_TRANSITION.search(block))
        if explicit_pair or directional:
            reason = "Explicit contrast structure found; its meaning remains a hypothesis."
            if scope is None:
                reason += " Memory grounding was not checked."
            return {"accepted": True, "reason": reason, "contrast": block}
    reason = ("Named spectrum sides are absent from supplied measurements." if absent_side else
              "No substantive, connected comparison of measured lower and higher behavior.")
    return {"accepted": False, "reason": reason}
