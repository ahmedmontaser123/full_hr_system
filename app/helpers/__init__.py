from .config import Settings, get_settings

def get_pool():
    from .database import get_pool as _get_pool
    return _get_pool()

def close_pool():
    from .database import close_pool as _close_pool
    return _close_pool()
