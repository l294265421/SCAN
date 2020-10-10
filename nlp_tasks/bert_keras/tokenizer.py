# -*- coding: utf-8 -*-


from keras_bert import Tokenizer


class TokenizerReturningSpace(Tokenizer):
    """
    增加处理空格的能力
    """
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R


class EnglishTokenizer(Tokenizer):
    """
    增加处理空格的能力
    """
    pass
