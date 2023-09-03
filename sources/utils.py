# Code authored by X. X.
import cv2

def load_image(my_file, flip_color = False):
    loaded_file = cv2.imread(my_file)
    if flip_color:
        loaded_file = 255 - loaded_file
    my_shape = loaded_file.shape
    if len(my_shape) == 2:
        return loaded_file
    if len(my_shape) == 3:
        if my_shape[2] == 1:
            return loaded_file[:,:,0]
        if my_shape[2] == 3:
            return cv2.cvtColor(loaded_file, cv2.COLOR_BGR2GRAY)
    raise NameError('Wrong image format')
    

# Code authored by bitagoras    
# Originated from https://stackoverflow.com/questions/2524853/how-should-i-put-try-except-in-a-single-line
class trialContextManager:
    def __enter__(self): pass
    def __exit__(self, *args): return True