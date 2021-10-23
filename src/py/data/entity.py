class Utterance(object):

    def __init__(self):
        self.text: str = None
        self.feature_name: str = None
        self.feature_type: str = None # SLOT/SLOT+VALUE/INTENT
        self.intent_name: str = None
        self.part_type: str = None  # TRAIN/DEV/TEST
        self.slots: dict = None


    def __str__(self):
        return self.__dict__.__str__()
