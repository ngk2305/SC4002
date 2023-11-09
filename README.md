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

 **AG NEWS Dataset** [[Download](https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms)] : This link is from TORCHTEXT.DATASETS.

| DATASET | COUNTS  |
| :-----: | :-----: |
|  TRAIN  | 4000 |
|  VALID  | 500  |
|  TEST   |  500  |

**Classes**

Original classes were 0, 1, 2, 3, 4, and 5 each, but changed them into 0, 1, 2, 3, 4 by merging 2 and 5 into 2.

* 0: Concept question

* 1: Object question

* 2: Others

* 3: Who question

* 4: Number question 

  

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

For test set,

| Accuracy | Precision | Recall |   F1   |
| :------: | :-------: | :----: | :----: |
|  0.754   |  0.7741   | 0.7630 | 0.7630 |

Confusion Matrix is like below,

```
[85  6  7  0  4]
[41 55  5  2  5]
[ 9  0 72  2  5]
[ 7 17  4 82  2]
[ 4  2  1  0 83]
```



## Reference

* Lai, S., Xu, L., Liu, K., & Zhao, J. (2015, February). Recurrent convolutional neural networks for text classification. In *Twenty-ninth AAAI conference on artificial intelligence*. [[Paper](http://zhengyima.com/my/pdfs/Textrcnn.pdf)]
