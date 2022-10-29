# The code and data for the paper "Sentence Constituent-Aware Aspect-Category Sentiment Analysis with Graph Attention Networks"

# Requirements
- Python 3.6.8
- torch==1.2.0
- pytorch-transformers==1.1.0
- allennlp==0.9.0

# Supported datasets
- SemEval-2014-Task-4-REST-DevSplits (Rest14)
- MAMSACSA (MAMS-ACSA)
- SemEval-141516-LARGE-REST

# Instructions:
Before excuting the following commands, replace glove.840B.300d.txt(http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip), bert-base-uncased.tar.gz(https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and vocab.txt(https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt) with the corresponding absolute paths in your computer. 

## SCAN
### Train
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_sentence_constituency.py --embedding_filepath glove.840B.300d.txt --data_type constituency --current_dataset SemEval-2014-Task-4-REST-DevSplits --joint_type joint --attention_warmup_init False --acd_sc_encoder_mode same --acd_encoder_mode mixed --bert False --pair False --lstm_or_fc_after_embedding_layer lstm --aspect_graph gat --sentiment_graph gat --batch_size 32 --train True --evaluate False

### Evaluate
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_sentence_constituency.py --embedding_filepath glove.840B.300d.txt --data_type constituency --current_dataset SemEval-2014-Task-4-REST-DevSplits --joint_type joint --attention_warmup_init False --acd_sc_encoder_mode same --acd_encoder_mode mixed --bert False --pair False --lstm_or_fc_after_embedding_layer lstm --aspect_graph gat --sentiment_graph gat --batch_size 32 --train False --evaluate True

## SCAN-BERT
### Train
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_sentence_constituency.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type constituency-bert-pair --current_dataset SemEval-2014-Task-4-REST-DevSplits --joint_type pipeline --attention_warmup_init True --acd_encoder_mode_for_sentiment_attention_warmup mixed --gnn_for_sentiment_attention_warmup gat --acd_sc_encoder_mode mixed --bert True --pair True --lstm_or_fc_after_embedding_layer lstm --aspect_graph gat --sentiment_graph gat --batch_size 16 --train True --evaluate False

### Evaluate
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_sentence_constituency.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type constituency-bert-pair --current_dataset SemEval-2014-Task-4-REST-DevSplits --joint_type pipeline --attention_warmup_init True --acd_encoder_mode_for_sentiment_attention_warmup mixed --gnn_for_sentiment_attention_warmup gat --acd_sc_encoder_mode mixed --bert True --pair True --lstm_or_fc_after_embedding_layer lstm --aspect_graph gat --sentiment_graph gat --batch_size 16 --train False --evaluate True

## SCAN-BERT-AVE
### Train
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_sentence_constituency.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type constituency-bert-pair --current_dataset SemEval-2014-Task-4-REST-DevSplits --joint_type pipeline --attention_warmup_init True --acd_encoder_mode_for_sentiment_attention_warmup mixed --gnn_for_sentiment_attention_warmup gat --acd_sc_encoder_mode mixed --bert True --pair True --lstm_or_fc_after_embedding_layer lstm --aspect_graph gat --sentiment_graph average --batch_size 16 --train True --evaluate False

### Evaluate
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_sentence_constituency.py --embedding_filepath glove.840B.300d.txt --bert_file_path bert-base-uncased.tar.gz --bert_vocab_file_path vocab.txt --data_type constituency-bert-pair --current_dataset SemEval-2014-Task-4-REST-DevSplits --joint_type pipeline --attention_warmup_init True --acd_encoder_mode_for_sentiment_attention_warmup mixed --gnn_for_sentiment_attention_warmup gat --acd_sc_encoder_mode mixed --bert True --pair True --lstm_or_fc_after_embedding_layer lstm --aspect_graph gat --sentiment_graph average --batch_size 16 --train False --evaluate True

## Visualization
After models are trained, we can visualize the attention weights by adding one extra option to the commands mentioned above, --visualize_attention True. For example,
python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_sentence_constituency.py --embedding_filepath glove.840B.300d.txt --data_type constituency --current_dataset SemEval-2014-Task-4-REST-DevSplits --joint_type joint --attention_warmup_init False --acd_sc_encoder_mode same --acd_encoder_mode mixed --bert False --pair False --lstm_or_fc_after_embedding_layer lstm --aspect_graph gat --sentiment_graph gat --batch_size 32 --train False --evaluate False --visualize_attention True

## Run the models for multiple times
In order to run the models for multiple times, we can use the shell script, repeat.sh, to run the commands mentioned above by replacing the "python" in the commands  with:

sh repeat.sh 0-0-0,0-0-1,0-0-2,0-0-3,0-0-4

where 0-0-0 is the name of the first run.

## Citation
```
@InProceedings{10.1007/978-3-030-60450-9_64,
    author="Li, Yuncong
    and Yin, Cunxiang
    and Zhong, Sheng-hua",
    editor="Zhu, Xiaodan
    and Zhang, Min
    and Hong, Yu
    and He, Ruifang",
    title="Sentence Constituent-Aware Aspect-Category Sentiment Analysis with Graph Attention Networks",
    booktitle="Natural Language Processing and Chinese Computing",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="815--827",
    abstract="Aspect category sentiment analysis (ACSA) aims to predict the sentiment polarities of the aspect categories discussed in sentences. Since a sentence usually discusses one or more aspect categories and expresses different sentiments toward them, various attention-based methods have been developed to allocate the appropriate sentiment words for the given aspect category and obtain promising results. However, most of these methods directly use the given aspect category to find the aspect category-related sentiment words, which may cause mismatching between the sentiment words and the aspect categories when an unrelated sentiment word is semantically meaningful for the given aspect category. To mitigate this problem, we propose a Sentence Constituent-Aware Network (SCAN) for aspect-category sentiment analysis. SCAN contains two graph attention modules and an interactive loss function. The graph attention modules generate representations of the nodes in sentence constituency parse trees for the aspect category detection (ACD) task and the ACSA task, respectively. ACD aims to detect aspect categories discussed in sentences and is a auxiliary task. For a given aspect category, the interactive loss function helps the ACD task to find the nodes which can predict the aspect category but can't predict other aspect categories. The sentiment words in the nodes then are used to predict the sentiment polarity of the aspect category by the ACSA task. The experimental results on five public datasets demonstrate the effectiveness of SCAN (Data and code can be found at https://github.com/l294265421/SCAN).",
    isbn="978-3-030-60450-9"
}
```