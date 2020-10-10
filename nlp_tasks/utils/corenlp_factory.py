# -*- coding: utf-8 -*-


import logging

from nlp_tasks.utils import my_corenlp
from nlp_tasks.common import common_path

MODEL_DIR = common_path.original_data_dir_big + 'stanford-corenlp-full-2018-02-27/'


def create_corenlp_server(start_new_server=False, lang='en', port=8081):
    path_or_host = MODEL_DIR
    if not start_new_server:
        path_or_host = 'http://localhost'
    return my_corenlp.StanfordCoreNLP(path_or_host, lang=lang, quiet=False, logging_level=logging.INFO, memory='4g',
                    port=port)


if __name__ == '__main__':
    create_corenlp_server(start_new_server=True, lang='en')
