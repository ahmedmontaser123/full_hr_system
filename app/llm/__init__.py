from .base import BaseLoader
from .evaluator import Evaluator
from .questions_generator import QuestionsGenerator
from .classification import ClassificationQuestion

def __getattr__(name):
    if name == "Transcript":
        from .transcript import Transcript
        return Transcript
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
