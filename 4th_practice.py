import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


mnist_test=dset.MNIST("C:\Users\최가원\Desktop\minist_test",
                      train=False, transform=transforms.ToTensor(), target_transform=None, download=True)


print "mnist_test 길이: ", len(mnist_test)

#데이터 하나의 형태
image, label= mnist_test.__getitem__(0) #0번째 데이터
print "image data 형태: ", image.size()
print "label: ", label

#그리기
img= image.numpy() #image 타입을 numpy로 변환(1, 28, 28)
plt.title("label: %d"%label) #표 title
plt.imshow(img[0], cmap='gray') #이미지 보여주기(gray)
plt.show()


def sigmoid(x):
    return 1/(1+np.exp(-x))
  
  
def softmax(x):
    e_x= np.exp(x)
    return e_x/np.sum(e_x)
  
  
#Multi-layered perceptron 다중퍼셉트론 (xor을 풀기위한)
# # of units in each layer: 28*28 - 256 - 128 - 10
class MyMLP:
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        # n_input= input layer, n_hidden1= hidden layer(w+x), n_output= output layer
        # W^(1): layer1 -> layer2에 매핑되는 weight
        
        # W^(1)
        self.W1= np.zeros((n_hidden1, n_input), dtype= np.float32) # W1(256, 28*28)
        self.b1= np.zeros((n_hidden1, ), dtype=np.float32) # bias1
        
        # W^(2)
        self.W2= np.zeros((n_hidden2, n_hidden1), dtype= np.float32) # W2(128, 256)
        self.b2= np.zeros((n_hidden2, ), dtype=np.float32) # bias2
        
        # W^(3)
        self.W3= np.zeros((n_output, n_hidden2), dtype= np.float32) # W3(10, 128)
        self.b3= np.zeros((n_output), dtype=np.float32) # bias3
        
        
    def __call__(self, x):
        # (1, 28, 28) -> (28, 28)
        x = x.reshape(-1) #이미지를 일렬로 피기
        
        h1= sigmoid(np.dot(self.W1, x) + self.b1) # W1(256, 28*28), x(28*28), b1(256) -> h1(256)
        h2= np.dot(self.W2, h1) + self.b2 # W2(128, 256), h1(256), b2(128) -> h2(128)
        out= np.dot(self.W3, h2) + self.b3 # W3(10, 128), h2(128), b3(10) -> out(10)
        
        return softmax(out) #(10) 10개의 유닛 return
        
        
model= MyMLP(28*28, 256, 128, 10)


#shape 출력
print model.W1.shape, model.b1.shape
print model.W2.shape, model.b2.shape
print model.W3.shape, model.b3.shape


weights= np.load('./nets/mlp_weight.npz')
model.W1= weights['W1']
model.b1= weights['b1']
model.W2= weights['W2']
model.b2= weights['b2']
model.W3= weights['W3']
model.b3= weights['b3']

#shape 출력
print model.W1.shape, model.b1.shape
print model.W2.shape, model.b2.shape
print model.W3.shape, model.b3.shape


mysum= 0

m= len(mnist_test)
cnt=0
for i in range(m):
    image, label= mnist_test.__getitem__(i) #i번째 데이터
    output= model(image)
    
    if(i%1000==0):
        img= image.numpy() #image 타입을 numpy로 변환 (1, 28, 28)
        
        #출력결과 모든 pred=0으로 나옴...
        # print output -> 모든 output 출력값이 [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1] 왜!?
        #output 이 문제니까 model 레벨 즉, MyMLP 클래스에서 문제가 있다는 건데 도무지 눈을 씻고 찾아봐도 오류가 보이지 않습니다...
        pred_label= np.argmax(output)
        plt.title("pred: %d, label: %d"%(pred_label, label))
        plt.imshow(img[0], cmap='gray')
        plt.show()
        
    cnt+=1
    mysum+= (np.argmax(output)==label) #output=결과값, label=정답레이블
    
print "정확도: %.2f"%((float(mysum)/cnt)*100.0)


