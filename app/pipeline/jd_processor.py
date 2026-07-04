"""
Step 1 of the pipeline: turn a raw Job Description string into structured,
labeled requirements using GLiNER (zero-shot NER).

Mirrors the notebook's `extract_from_jd`, but the labels are kept generic
("job title", "years of experience", ...) so the same module can be reused
if you swap GLiNER models later.

Before NER, the JD is split into sections using the *same section keys*
as `cv_processor.split_cv_sections` (skills, experience, education,
languages, summary) plus a JD-only "requirements" bucket — this is what
lets `matcher.hard_match` compare "like with like" later, and lets GLiNER
focus on the parts of the JD that actually contain requirements instead
of boilerplate (company blurb, benefits, legal footer, ...).
"""

import html
import re
import unicodedata
from typing import Dict, List

from pipeline.models import ModelRegistry

# ---------------------------------------------------------------------------
# JD text cleaning
# ---------------------------------------------------------------------------
# Deliberately kept in its own section, with its OWN regex constants, and
# never merged into JD_SECTION_PATTERNS. JD_SECTION_PATTERNS is for header /
# section-type detection (requirements vs skills vs ...); this block only
# normalizes the raw text *before* that detection runs. Mixing the two would
# mean every future "one more edge case" fix has to be threaded through the
# header-matching alternations (like the "skills" pattern), which is exactly
# what we want to avoid.
#
# Real-world JD payloads commonly arrive mangled in one of these ways:
#   - Literal escape sequences as text: a client sends the string
#     "Requirements\\n5+ years" (backslash + "n", two characters) instead of
#     an actual newline, usually from double-encoding JSON or copy/pasting
#     from a log/terminal. `text.split("\n")` in split_jd_sections then sees
#     ZERO real line breaks and the entire JD collapses into one "header"
#     bucket - no section ever gets detected.
#   - Real "\r\n" / lone "\r" line endings (Windows-authored JDs).
#   - Non-breaking spaces (\xa0), zero-width spaces/joiners, and BOM marks
#     copy-pasted from Word/Google Docs/web pages.
#   - HTML remnants (<br>, <p>, <li>, &nbsp;, &amp;, ...) when a JD was
#     scraped or pasted from an HTML source instead of plain text.
#   - Exotic bullet glyphs (•, ‣, ▪, ●, ▶, ✓, »...) that make an otherwise
#     short header-like line fail to read as a header, or clutter content.
#   - Runs of 3+ blank lines / repeated horizontal whitespace.
_LITERAL_ESCAPE_PATTERN = re.compile(r"\\r\\n|\\n|\\r|\\t")
_LITERAL_ESCAPE_MAP = {
    r"\r\n": "\n",
    r"\n": "\n",
    r"\r": "\n",
    r"\t": " ",
}

_HTML_BLOCK_BREAK_TAGS = re.compile(
    r"</?\s*(br|p|div|li|ul|ol|tr|table|h[1-6])\s*/?\s*>", re.IGNORECASE
)
_HTML_ANY_TAG = re.compile(r"<[^>]+>")

_BULLET_CHARS = re.compile(r"^[\s]*[•‣▪●◦▶○✓✔·∙–—*-]\s*")
_ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")


def clean_jd_text(text: str) -> str:
    """
    Normalize a raw JD payload before it ever reaches split_jd_sections /
    GLiNER. Idempotent (safe to call more than once) and safe on already
    clean text.

    Order matters:
      1. Unescape HTML entities (&nbsp;, &amp;, ...) first, since some of
         them decode INTO the raw whitespace characters step 2-4 clean up.
      2. Convert literal backslash-escape sequences ("\\n", "\\r\\n", "\\t"
         as literal text) into real newlines/spaces.
      3. Turn HTML block-level tags into real newlines, then strip any
         remaining tags.
      4. Strip zero-width/BOM characters and normalize unicode whitespace
         (NBSP, thin space, etc.) to a plain ASCII space.
      5. Strip a single leading bullet glyph per line.
      6. Collapse excess horizontal whitespace and blank lines.
    """
    if not text:
        return text

    cleaned = html.unescape(text)

    def _replace_escape(match: "re.Match") -> str:
        return _LITERAL_ESCAPE_MAP[match.group(0)]

    cleaned = _LITERAL_ESCAPE_PATTERN.sub(_replace_escape, cleaned)

    if "<" in cleaned and ">" in cleaned:
        cleaned = _HTML_BLOCK_BREAK_TAGS.sub("\n", cleaned)
        cleaned = _HTML_ANY_TAG.sub("", cleaned)

    cleaned = _ZERO_WIDTH_CHARS.sub("", cleaned)
    cleaned = "".join(
        " " if unicodedata.category(ch) == "Zs" else ch for ch in cleaned
    )

    lines = cleaned.split("\n")
    lines = [_BULLET_CHARS.sub("", line) for line in lines]
    cleaned = "\n".join(lines)

    cleaned = _MULTI_SPACES.sub(" ", cleaned)
    cleaned = _MULTI_BLANK_LINES.sub("\n\n", cleaned)
    lines = [line.strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned


# Same category keys cv_processor.split_cv_sections uses, plus JD-only
# buckets ("requirements", "nice_to_have", "job_meta") for headers that
# don't map 1:1 to a CV section header.
#
# Dict ORDER matters: split_jd_sections tries patterns in insertion order
# and stops at the first match, so more specific patterns are listed
# before more general ones that could otherwise swallow them, e.g.:
#   - "nice_to_have" before "requirements": "Preferred Qualifications"
#     must resolve to nice_to_have, not get merged into hard requirements.
#   - "job_meta" before "experience": "Experience Level" is a metadata
#     field (seniority), not a "work experience / responsibilities" body.
#
# Each pattern also tolerates the qualifying words real JDs actually use
# around a core noun ("Required Qualifications", "Minimum Qualifications",
# "Technical Skills", "Soft Skills", ...) instead of requiring an exact
# bare-word match, which was the main reason headers were falling through
# unrecognized.
JD_SECTION_PATTERNS = {
    "nice_to_have": (
        r"(?:preferred|desired|good[\s-]+to[\s-]+have|nice[\s-]+to[\s-]+have|"
        r"bonus|pluses?|optional|extra)\s*"
        r"(?:points?)?\s*"
        r"(?:qualifications?|requirements?|skills?)?"
    ),
    "job_meta": (
        r"employment\s+type|experience\s+level|seniority(\s+level)?|"
        r"job\s+type|work\s+arrangement|work\s+mode|"
        r"location|salary|compensation|benefits?|perks?"
    ),
    "requirements": (
        r"(?:(?:required|minimum|must[\s-]have|essential|mandatory|basic|"
        r"key|other|additional)\s+)?(?:requirements?|qualifications?)"
        r"|what\s+you('ll)?\s+(need|bring)"
        r"|must\s+have"
    ),
    "experience": (
        r"(work\s+)?experience|responsibilities|duties|"
        r"what\s+you('ll)?\s+do|role\s+overview"
    ),
    "education": r"education(al\s+background)?|academic(\s+background)?|degrees?",
    "skills": (
        r"(?:technical|soft|hard|core|key|general)?\s*skills?"
        r"|tech(nology)?\s+stack|tools?"
        r"|programming(\s+languages?)?|libraries|frameworks?"
        r"|data\s+analysis|machine\s+learning|artificial\s+intelligence"
    ),
    "languages": r"languages?|spoken\s+languages?",
    "summary": (
        r"summary|about\s+(the\s+role|us|the\s+company)|overview"
        r"|what\s+you('ll)?\s+learn"
    ),
}

JD_LABELS = [
    "job title",
    "years of experience",
    "programming language or technical skill",
    "soft skill or personality trait",
    "education degree",
    "field of study",
    "spoken or written language",
    "preferred technology tool listed as a plus or optional",
    "city country or region where job is based",
    "job type or work arrangement",
]

# Map GLiNER's natural-language labels -> the canonical keys used
# downstream by the hard matcher (kept identical to the notebook's keys).
LABEL_KEY_MAP = {
    "job title": "required job title",
    "years of experience": "required years of experience",
    "programming language or technical skill": "required hard skill",
    "soft skill or personality trait": "required soft skill",
    "education degree": "required education degree",
    "field of study": "required field of study",
    "spoken or written language": "required spoken language",
    "preferred technology tool listed as a plus or optional": "nice_to_have_skill",
    "city country or region where job is based": "work_location",
    "job type or work arrangement": "job_type",
}


# ---------------------------------------------------------------------------
# Deterministic post-processing (after GLiNER, before the final bucketing)
# ---------------------------------------------------------------------------
# GLiNER's own judgement on "is this required or optional" is inherently
# fuzzy, and real JDs often say the quiet part out loud right next to the
# item itself - e.g. a "Libraries" list like:
#     Scikit-learn, TensorFlow (preferred), PyTorch (preferred), XGBoost (preferred)
# where only the parenthetical actually marks which ones are optional.
# Nothing here touches JD_SECTION_PATTERNS - this only re-checks entities
# GLiNER already returned, using the literal JD text as ground truth.
_OPTIONAL_INLINE_MARKER = re.compile(
    r"\(\s*(?:preferred|optional|nice[\s-]to[\s-]have|bonus|a\s+plus|desired)\s*\)",
    re.IGNORECASE,
)

# Closed, well-known vocabulary for employment type / work arrangement.
# Used as a deterministic safety net alongside GLiNER's "job type or work
# arrangement" label - a short field like "Full-time" is exactly the kind
# of thing a general-purpose zero-shot label can miss, and there's no
# reason to leave "job_type" empty when the literal word is right there
# in the job_meta section.
_JOB_TYPE_VOCAB = re.compile(
    r"\b(full[\s-]?time|part[\s-]?time|contract(?:or)?|freelance|"
    r"intern(?:ship)?|temporary|temp|permanent|remote|hybrid|on[\s-]?site)\b",
    re.IGNORECASE,
)


def _looks_optional_inline(entity_text: str, ner_input: str) -> bool:
    """True if entity_text is immediately followed (within a short
    window, allowing for punctuation) by an inline optional marker in
    the source text, e.g. "TensorFlow (preferred)". Checks every
    occurrence of entity_text in the text, since the same skill can be
    mentioned more than once (marked in one place, unmarked in another)."""
    if not entity_text:
        return False
    haystack = ner_input.lower()
    needle = entity_text.lower()
    start = haystack.find(needle)
    while start != -1:
        window = ner_input[start : start + len(needle) + 20]
        if _OPTIONAL_INLINE_MARKER.search(window):
            return True
        start = haystack.find(needle, start + 1)
    return False


def _came_from_nice_to_have_section(entity_text: str, sections: Dict[str, str]) -> bool:
    """Coarse provenance check: the entity's text shows up in the
    nice_to_have section but nowhere in requirements/skills. Used as a
    secondary signal alongside the inline marker above."""
    if not entity_text:
        return False
    text = entity_text.lower()
    nice = sections.get("nice_to_have", "").lower()
    hard = (sections.get("requirements", "") + " " + sections.get("skills", "")).lower()
    return text in nice and text not in hard


def _is_actually_optional(entity_text: str, ner_input: str, sections: Dict[str, str]) -> bool:
    return _looks_optional_inline(entity_text, ner_input) or _came_from_nice_to_have_section(
        entity_text, sections
    )


def _job_type_fallback(sections: Dict[str, str]) -> List[str]:
    """Deterministic supplement for job_type: scans the job_meta section
    text directly for standard employment-type wording. Only adds
    matches - never removes anything GLiNER already found - so this
    can only fill genuine gaps, not override the model."""
    text = sections.get("job_meta", "")
    return [m.group(0).lower() for m in _JOB_TYPE_VOCAB.finditer(text)]


def _bucket_entities(entities: List[Dict], ner_input: str, sections: Dict[str, str]) -> Dict[str, List[str]]:
    result = {key: [] for key in LABEL_KEY_MAP.values()}
    for entity in entities:
        key = LABEL_KEY_MAP.get(entity.get("label"))
        if not key:
            continue
        text_value = entity["text"].lower().strip()
        if key in ("required hard skill", "required soft skill") and _is_actually_optional(
            entity["text"], ner_input, sections
        ):
            key = "nice_to_have_skill"
        result[key].append(text_value)

    result["job_type"].extend(_job_type_fallback(sections))

    for key in result:
        result[key] = sorted(set(result[key]))

    return result


def split_jd_sections(text: str) -> Dict[str, str]:
    """Same header-detection approach as cv_processor.split_cv_sections.

    Runs `clean_jd_text` first so mangled input (literal "\\n" sequences,
    HTML remnants, non-breaking spaces, stray bullets, ...) doesn't prevent
    header detection below - see the "JD text cleaning" section above for
    why that matters.

    IMPORTANT (bug fix): real JDs frequently repeat headers that map to the
    *same* canonical key at different points in the document (e.g. a
    "Technical Skills" header followed later by a "Tools" sub-heading —
    both map to "skills"). The original implementation did
    `sections[current] = []` on every header match, which reset the list
    and silently discarded everything collected under that key so far.
    We now use `setdefault` so a repeated header appends to the existing
    section instead of wiping it.
    """
    text = clean_jd_text(text)
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {}
    current = "header"
    sections[current] = []

    for line in lines:
        stripped = line.strip()
        matched = False
        for section, pattern in JD_SECTION_PATTERNS.items():
            header_match = re.match(r"^(" + pattern + r")\s*(?::\s*(.*))?$", stripped.lower())
            if header_match and len(stripped) < 60:
                current = section
                sections.setdefault(current, [])
                inline_content = header_match.group(2)
                if inline_content and inline_content.strip():
                    sections[current].append(inline_content.strip())
                matched = True
                break
        if not matched:
            sections[current].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def _ner_text(jd_text: str, sections: Dict[str, str]) -> str:
    """
    Prefer the requirements/skills/experience/education sections for NER
    (denser signal, less boilerplate). Falls back to the full JD text if
    section-splitting didn't find anything useful (e.g. unstructured JD
    with no headers at all).

    "header" and "summary" are included on purpose (Fix 1): the job title
    almost always sits in the first line(s) of the posting or in the
    "About the role" blurb, not under "Requirements"/"Skills"/etc. Without
    them, GLiNER never sees the title text at all and "required job title"
    comes back empty on most real-world JDs.

    "nice_to_have" and "job_meta" are included on purpose (Fix 2): a
    "Preferred/Nice to Have" section is exactly where the "preferred
    technology tool listed as a plus or optional" label gets its
    signal, and a "job_meta" section (Employment Type, Experience Level,
    Location, ...) is exactly where "job type or work arrangement" and
    "city country or region" entities live. Leaving either out means
    GLiNER never sees that text at all.
    """
    focused = " \n".join(
        sections.get(key, "")
        for key in (
            "header",
            "summary",
            "requirements",
            "nice_to_have",
            "skills",
            "experience",
            "education",
            "languages",
            "job_meta",
        )
        if sections.get(key)
    )
    return focused if len(focused) > 50 else jd_text


# ---------------------------------------------------------------------------
# Per-section label routing
# ---------------------------------------------------------------------------
# Instead of concatenating every relevant section into one blob and asking
# GLiNER to search for all JD_LABELS in it, we run GLiNER once PER detected
# section, each time restricted to only the labels that section can
# plausibly contain. A JD with 7 recognized sections means (up to) 7
# separate GLiNER calls instead of 1 - each scoped to a small piece of text
# and a small label set, so the model isn't asked to look for e.g. "field
# of study" inside the "Employment Type / Location" blurb, and 2-3 labels
# is a much easier zero-shot task than all 10 at once.
#
# A section that's empty (or wasn't detected at all in this JD) is simply
# skipped - no call is made for it.
SECTION_LABEL_MAP: Dict[str, List[str]] = {
    "header": ["job title"],
    "summary": ["job title", "years of experience"],
    "requirements": [
        "years of experience",
        "programming language or technical skill",
        "soft skill or personality trait",
        "education degree",
        "field of study",
    ],
    "skills": [
        "programming language or technical skill",
        "soft skill or personality trait",
    ],
    "experience": [
        "years of experience",
        "programming language or technical skill",
        "soft skill or personality trait",
    ],
    "education": ["education degree", "field of study"],
    "languages": ["spoken or written language"],
    "nice_to_have": [
        "preferred technology tool listed as a plus or optional",
        "programming language or technical skill",
        "soft skill or personality trait",
    ],
    "job_meta": [
        "city country or region where job is based",
        "job type or work arrangement",
    ],
}


def _run_ner_per_section(ner_model, sections: Dict[str, str], threshold: float) -> List[Dict]:
    """Run GLiNER once per non-empty section that appears in
    SECTION_LABEL_MAP, each call scoped to only that section's relevant
    labels. Returns the combined entity list, exactly like one big call
    would have - `_bucket_entities` doesn't need to know sectioning
    happened."""
    entities: List[Dict] = []
    for section_key, labels in SECTION_LABEL_MAP.items():
        section_text = sections.get(section_key, "")
        if not section_text.strip():
            continue
        entities.extend(
            ner_model.predict_entities(
                section_text,
                labels,
                threshold=threshold,
                flat_ner=True,
            )
        )
    return entities


def _run_ner(jd_text: str, threshold: float):
    """Shared by extract_from_jd / extract_from_jd_with_sections: clean,
    split, run GLiNER, return everything the bucketing step needs.

    If the section splitter found real section boundaries beyond just
    "header" (i.e. this looks like a properly structured JD), GLiNER runs
    once PER section with a section-specific label subset - see
    SECTION_LABEL_MAP and _run_ner_per_section.

    Otherwise (a short or unstructured JD where everything landed in the
    catch-all "header" bucket, with no real headers to split on) we fall
    back to the original single whole-text pass over every label - this is
    deliberately kept as-is since it's the behavior that already performs
    well on short JDs.
    """
    jd_text = clean_jd_text(jd_text)
    sections = split_jd_sections(jd_text)
    ner_input = _ner_text(jd_text, sections)

    non_header_content_len = sum(
        len(text) for key, text in sections.items() if key != "header"
    )

    ner_model = ModelRegistry.ner()

    if non_header_content_len > 50:
        entities = _run_ner_per_section(ner_model, sections, threshold)
    else:
        entities = ner_model.predict_entities(
            ner_input,
            JD_LABELS,
            threshold=threshold,
            flat_ner=True,
        )

    return entities, ner_input, sections


def extract_from_jd(jd_text: str, threshold: float = 0.3) -> Dict[str, List[str]]:
    """
    Split the JD into sections, run GLiNER over the requirements-relevant
    sections, and bucket entities by canonical label.

    Returns a dict keyed by LABEL_KEY_MAP values, e.g.:
        {
          "required hard skill": ["python", "tensorflow", ...],
          "required years of experience": ["5 years"],
          ...
        }
    """
    entities, ner_input, sections = _run_ner(jd_text, threshold)
    return _bucket_entities(entities, ner_input, sections)


def extract_from_jd_with_sections(jd_text: str, threshold: float = 0.3) -> Dict:
    """
    Same as extract_from_jd, but also returns the section split — used by
    routers/jobs.py so the sections (esp. "requirements"/"skills") can be
    persisted and reused for the embedding text (build_jd_query) and for
    the reranker's JD-side text later.
    """
    entities, ner_input, sections = _run_ner(jd_text, threshold)
    result = _bucket_entities(entities, ner_input, sections)
    return {"extracted": result, "sections": sections}


def build_jd_query(jd_extracted: Dict[str, List[str]], sections: Dict[str, str] = None) -> str:
    """
    Build a focused query string from JD extracted fields (+ raw
    skills/requirements section text, if available) for embedding.
    """
    parts = []
    if jd_extracted.get("required hard skill"):
        parts.append("Skills: " + ", ".join(jd_extracted["required hard skill"]))
    if jd_extracted.get("required job title"):
        parts.append("Role: " + ", ".join(jd_extracted["required job title"]))
    if jd_extracted.get("required years of experience"):
        parts.append("Experience: " + ", ".join(jd_extracted["required years of experience"]))
    if jd_extracted.get("required education degree"):
        parts.append("Education: " + ", ".join(jd_extracted["required education degree"]))

    if sections:
        if sections.get("requirements"):
            parts.append("Requirements: " + sections["requirements"])
        elif sections.get("skills"):
            parts.append("Requirements: " + sections["skills"])

    return " | ".join(parts)
