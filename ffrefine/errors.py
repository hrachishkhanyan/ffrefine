"""Exception hierarchy for ffrefine."""


class FFRefineError(Exception):
    """Base class for all ffrefine failures."""


class ParseError(FFRefineError):
    """A chemistry input file could not be parsed."""


class ValidationError(FFRefineError):
    """Inputs parsed but are mutually inconsistent."""


class ExternalToolError(FFRefineError):
    """An external binary failed or produced unusable output."""
