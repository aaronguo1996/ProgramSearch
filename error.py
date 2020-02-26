# after some operation, the output string remains the same as input
# mark this as an useless action and resample one
class NoChangeError(Exception):
    pass

# if the committed string is not a prefix of desired output
# mark this committed as useless and resample
class ConcatError(Exception):
    pass

# we need more insertion points for regex, resample
class SmallSampleError(Exception):
    pass

def check_change(old, new):
    if str(old) == str(new):
        # print('Old:', str(old))
        # print('New:', str(new))
        raise NoChangeError
