class NoChangeError(Exception):
    pass

def check_change(old, new):
    if str(old) == str(new):
        raise NoChangeError
