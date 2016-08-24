Neural Networks in Keras for Numerai 
--------------

Practice training neural networks in Theano/Keras for Numerai submissions. (https://numer.ai/)

Uses feed-forward neural network with 1 hidden layer, sigmoid activation function to predict Numerai scores. 

Number of hidden layer nodes and regularization parameter selected using manual k-fold cross validation. 

Requirements: 

``` 
sudo pip install keras
```

Command-line options:

```
-n: [number of possible nodes in hidden layer] 
-r: [possible values of lambda] 
-k (optional): number of folds in cross validation
```


Example:  

```
python main.py -n 50 75 100 150 -r .01 .015 .02 .025 -k 10
```
