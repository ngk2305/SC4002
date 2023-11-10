# RCNN for Text Classification in PyTorch

PyTorch implementation of "[Recurrent Convolutional Neural Network for Text Classification](http://zhengyima.com/my/pdfs/Textrcnn.pdf) (2015)"



## Model

![model](https://user-images.githubusercontent.com/53588015/96370598-5c3b7100-1199-11eb-9bbe-903d4ba8aeda.png)



## Requirements

```
PyTorch
sklearn
nltk
pandas
gensim
```



## Dataset

 **TREC Dataset** [[Download](https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi?resource=download)] 

| DATASET | COUNTS  |
| :-----: | :-----: |
|  TRAIN  | 5000 |
|  VALID  | 500  |
|  TEST   |  500  |

**Classes**

Original classes were 0, 1, 2, 3, 4, and 5 each, but changed them into 0, 1, 2, 3, 4 by merging 2 and 5 into 2.

* 0: Concept question

* 1: Object question

* 2: Others

* 3: Who question

* 4: Number question 

## Use custom dataset:

# To use custom dataset, put them in data folder under format: 
- train.csv
- dev.csv
- test.csv
# Then run Word2Vec.py to get the converted data

## Training

To train,

```
python main.py --epochs 10
```

To train and want to see test set result,

```
python main.py --epochs 10 --test_set
```



## Result


For the test set,

|* TEST SET *| |ACC| 90.6000 |PRECISION| 0.9063 |RECALL| 0.9026 |F1| 0.9017

The confusion Matrix is like below,

```
[134   2   2   0   0]
[12 68  6  6  2]
[ 2  4 82  1  1]
[ 0  2  0 63  0]
[  5   0   1   1 106]
```



## Reference

* Lai, S., Xu, L., Liu, K., & Zhao, J. (2015, February). Recurrent convolutional neural networks for text classification. In *Twenty-ninth AAAI conference on artificial intelligence*. [[Paper](http://zhengyima.com/my/pdfs/Textrcnn.pdf)]
