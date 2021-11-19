class Utterance(object):

    def __init__(self, text: str = None, feature_name: str = None, part_type: str = None, slots: dict = None,
                 intent_name: str = None, feature_type: str = None, domain=None):
        self.text = text
        self.feature_name = feature_name
        self.feature_type = feature_type  # SLOT/SLOT+VALUE/INTENT/DOMAIN/CROSS_DOMAIN
        self.intent_name = intent_name
        self.part_type = part_type  # TRAIN/DEV/TEST
        self.slots = slots
        self.domain = domain  # Not available for SNIPS data

    def __str__(self):
        return self.__dict__.__str__()
