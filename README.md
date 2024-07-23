
# Prerequisites

- Download [Glove](https://nlp.stanford.edu/projects/glove/)
  - Twitter 50 dim: `wget https://nlp.stanford.edu/data/glove.twitter.27B.zip`
- [Tweebo parser](https://github.com/ikekonglp/TweeboParser)
  - Download the pretrained models from [here](http://www.cs.cmu.edu/~ark/TweetNLP/pretrained_models.tar.gz)

# Prepare the inputs

## Tokenize:
```bash …/TweeboParser/tokenize_and_tag_only.sh <root_dir> …/TweeboParser/ark-tweet-nlp-0.3.2 <working directory> …/pretrained_models/tagging_model …/TweeboParser/scripts input.csv```

## Extract the argument:
```python discourseParsing/ArgumentExtractor.py input.csv_tagger.out n```

## Format into inputs:
```python discourseParsing/utils/InputMaker.py input.csv_tagger.out_args.csv input.csv_tagger.out_args_train_meta.csv```

# Training:
```python train.py --tr_file input.csv_tagger.out_args.dict --tr_meta_file input.csv_tagger.out_args_train_meta.odict --cuda --input_dim 200 --hidden_dim 200 --word_embedding_dict …/glove/glove_200.dict --attn_act ReLU```
- Saves the model 

# Prediction:
```python predict.py --test_file test_file.csv_tagger.out_args.dict --cuda --input_dim 200 --hidden_dim 200 --word_embedding_dict …/glove/glove_200.dict --attn_act ReLU --model model.ptstdict --split_one_arg```
- Split one argument into two if that’s the only argument in the whole message, otherwise the message will be ignored if it’s too short to get any discourse relations

# Pretrained models and dataset
Please send an email to vvaradarajan@cs.stonybrook.edu to get access the social media treebank and pretrained models.

# Citation
If you are using oour data oor models, please cite our work:
```
@inproceedings{son-etal-2022-discourse,
    title = "Discourse Relation Embeddings: Representing the Relations between Discourse Segments in Social Media",
    author = "Son, Youngseo  and
      Varadarajan, Vasudha  and
      Schwartz, H. Andrew",
    booktitle = "Proceedings of the Workshop on Unimodal and Multimodal Induction of Linguistic Structures (UM-IoS)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.umios-1.5",
    doi = "10.18653/v1/2022.umios-1.5",
    pages = "45--55",
  }
```
