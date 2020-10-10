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
