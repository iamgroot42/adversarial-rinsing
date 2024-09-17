
class Removal:
    def __init__(self, args):
        self.args = args
    
    def remove_watermark(self, original_image):
        raise NotImplementedError("Subclasses must implement this method")
