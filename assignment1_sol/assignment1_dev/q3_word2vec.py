import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30
    
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (np.amax(np.fabs(x - np.array([[0.6,0.8],[0.4472136,0.89442719]]))) <= 1e-6)
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE
    #由中心词推导外围词的概率
    probabilities = softmax(predicted.dot(outputVectors.T))
    cost = -np.log(probabilities[target])#获取指定目标外围词的误差,Cross Entropy
    delta = probabilities #1*5
    delta[target] -= 1
    N = delta.shape[0]#5
    D = predicted.shape[0]#3
    grad = delta.reshape((N,1)) * predicted.reshape((1,D))#输出词词向量变化值,5*3
    gradPred = (delta.reshape((1,N)).dot(outputVectors)).flatten()#输出中心词词向量变化量，即predicted vector,1*3
    
    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    
    indices = [target]
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]#添加除了target词以外的词
        
    labels = np.array([1] + [-1 for k in xrange(K)])
    vecs = outputVectors[indices,:]#uo，11*3
    
    t = sigmoid(vecs.dot(predicted) * labels)#对应nagetive sampling 公式,11*1
    cost = -np.sum(np.log(t))#对应nagetive sampling 公式
    
    delta = labels * (t - 1)#11*1
    gradPred = delta.reshape((1,K+1)).dot(vecs).flatten()#对应δJ/δvc
    gradtemp = delta.reshape((K+1,1)).dot(predicted.reshape(
        (1,predicted.shape[0])))#对应δJ/δuo,δJ/δuk，此处合并写，11*3
    for k in xrange(K+1):
        grad[indices[k]] += gradtemp[k,:]#将--对应的--目标词向量和negative sampling中的词向量变化量更新到输出词向量中
    
#     t = sigmoid(predicted.dot(outputVectors[target,:]))
#     cost = -np.log(t)
#     delta = t - 1

#     gradPred += delta * outputVectors[target, :]
#     grad[target, :] += delta * predicted
    
#     for k in xrange(K):
#         idx = dataset.sampleTokenIdx()
        
#         t = sigmoid(-predicted.dot(outputVectors[idx,:]))
#         cost += -np.log(t)
#         delta = 1 - t

#         gradPred += delta * outputVectors[idx, :]
#         grad[idx, :] += delta * predicted
    
    ### END YOUR CODE
    
    return cost, gradPred, grad

#inputVectors代表的是每个输入词的词向量，而outputVectors代表的是输出词的词向量预测值！！！！！
#用当前中心词词向量乘以输出词词向量得到指定的外围词的分数，然后用softmax计算指定外围词的误差以及词变化量
#更新当前中心词词变化量，更新所有输出词词变化量
#重复以上步骤直到文本中的所有外围词都被用到
#最后将累计中心词的词变化量更新到所有的输入词向量中
#这里只进行一次输入输出词向量更新，以后可以到sgd中进行循环更新
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    #假如输入词向量5行3列，输出词向量5行3列
    currentI = tokens[currentWord]
    predicted = inputVectors[currentI, :]#获取中心词的词向量
    
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)#输入词向量变化量
    gradOut = np.zeros(outputVectors.shape)
    for cwd in contextWords:
        idx = tokens[cwd]
		#中心词不变，计算用当前中心词推断指定外围词的误差
        cc, gp, gg = word2vecCostAndGradient(predicted, idx, outputVectors, dataset)
        cost += cc
        gradOut += gg#更新输出词词向量变化量
        gradIn[currentI, :] += gp#更新当前中心词词向量变化量
    
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)#5*3
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    
    D = inputVectors.shape[1]#5
    predicted = np.zeros((D,))
    
    indices = [tokens[cwd] for cwd in contextWords]
    for idx in indices:
        predicted += inputVectors[idx, :]#将周围词词向量相加来得到中心词的词向量
    
    cost, gp, gradOut = word2vecCostAndGradient(predicted, tokens[currentWord], outputVectors, dataset)
    for idx in indices:
        gradIn[idx, :] += gp#只更新文本框中出现的周围词的词向量
    
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
		# 更新输入词向量的词向量变化量，以方便sgd更新x -= step * grad
		# 注意此处根据当前中心词算出的中心词词向量变化量将被加到所有的输入词向量中
		# 根据当前中心词算出的外围词词向量变化量将被加到所有的输出词词向量中
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4) #从0-4中随机产生一个整数

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()