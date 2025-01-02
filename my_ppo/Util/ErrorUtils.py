import traceback
from functools import wraps

def error_handler(func):
    """
    汎用的なエラーハンドリング用デコレータ
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # スタックトレースの取得
            error_details = traceback.format_exc()
            print(f"Error in function '{func.__name__}':\n{error_details}")
            raise RuntimeError(f"An error occurred in {func.__name__}: {e}")
    return wrapper
