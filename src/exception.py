import sys

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = sys.exc_info()
    error_message = f"{error} at line {exc_tb.tb_lineno} in {exc_tb.tb_frame.f_code.co_filename}: {error_detail}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

