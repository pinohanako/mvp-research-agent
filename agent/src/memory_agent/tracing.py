from functools import wraps
from langfuse import get_client, observe

langfuse = get_client()

def traced(func):
    @observe(name=func.__name__)
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper