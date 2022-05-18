# from allennlp.data.tokenizers import Token as OToken


# class Token(OToken):
#     def __new__(cls, text: str = None,
#         idx: int = None,
#         lemma_: str = None,
#         pos_: str = None,
#         tag_: str = None,
#         dep_: str = None,
#         ent_type_: str = None,
#         text_id: int = None,
#         edx: int = None
#     ):
#         self = super(Token, cls).__new__(cls, text, idx, lemma_, pos_, tag_, dep_, ent_type_, text_id)
#         self.edx = edx
#         return self
class Token(object):
    def __init__(self, text: str = None,
                 idx: int = None,
                 edx: int = None
                 ):
        self.text = text
        self.idx = idx
        self.edx = edx