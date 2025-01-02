from .ErrorUtils import error_handler
import time, datetime

@error_handler
def get_current_time_string():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")