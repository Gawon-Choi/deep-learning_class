import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np


mnist_train=dset.MNIST("", train=True, transform=transforms.ToTensor(), #train용
                       target_transform=None, download=True)
mnist_test=dset.MNIST("", train=False, transform=transforms.ToTensor(), #test용
                     target_transform=None, download=True)


print "mnist_train 길이: ", len(mnist_train)
print "mnist_test 길이: ", len(mnist_test)

#데이터 하나 형태
image, label= mnist_train.__getitem__(0) #train을 위한 데이터[0] -> 0번째
print "image data 형태: ", image.size()
print "label: ", label

#그리기
img= image.numpy() #image 타입을 numpy로 변환 (1, 28, 28)
plt.title("label: %d" %label)
plt.imshow(img[0], cmap='gray') #0번째 img(train data)를 gray로 보여주기
plt.show()


#hyper parameters -> # of epoch(학습시키는 것을 반복하는 횟수)
batch_size= 1024 
learning_rate= 0.01 # 일반적으로 0.1, 0.01, 0.001, ...등의 값을 learning rate로 삼음

num_epoch= 400 # -> train dataset으로 총 400번 반복하여 학습
#1 epoch: 60,000(# of train data)개의 train data로 '총 1 번' 반복하여 학습 


#DataLoader: batch_size만큼 데이터를 끊는 역할(덩어리로 나눔)
#shuffle: data를 뒤섞는 것(True: 섞는다/ False:섞지 않는다)
#통상적으로 test data는 shuffle 하지 않음
#drop_last=True -> batch_size만큼 나눌 때 나머지는 버려라

train_loader= torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                         shuffle=True, num_workers=2,
                                         drop_last=True)

test_loader= torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,
                                         shuffle=False, num_workers=2,
                                         drop_last=True)


n= 3 #sample로 그려볼 데이터 개수
for i, [imgs, labels] in enumerate(test_loader):
    #test dataset의 img, label을 batch_size(=1024)만큼 [imgs, labels]에 삽입
    if i>5:
        break
    
    print "[%d]" %i
    print "한 번에 로드되는 데이터 크기: ", len(imgs) # ==batch_size
    
    #그리기 -> 1줄에 3개씩 img를 보여줘
    for j in range(n):
        img = imgs[j].numpy() #image 타입을 numpy로 변환 (1, 28, 28)
        img = img.reshape(img.shape[1], img.shape[2]) 
        #(1, 28, 28)(numpy 모양) --(reshape)--> (28, 28)
        #print img.shape
        
        plt.subplot(1, n, j+1) #(1,3) 형태 플랏의 j번째 자리에 그리겠다
        plt.imshow(img, cmap='gray')
        plt.title("label: %d" %labels[j])
    plt.show()
    
    
    #모델 선언
# *  퍼셉트론(2 hidden layer)  *
model= nn.Sequential( #nn-> 맨 상단에 import해둠
    nn.Linear(28*28, 256),#input layer(size fix) -> hidden layer1(수정가능)
    nn.Sigmoid(),
    nn.Linear(256, 128), #hidden layer1 -> hidden layer2(수정가능)
    nn.Linear(128, 10), #hidden layer2 -> output layer(size fix)
)
#파라미터 보기
#print(list(model.parameters())) 
# -> 여기서 출력된 값으로 해당 model의 w1, b1, w2, b2, w3, b3을 파악할 수 있다. 
# .npz 파일 만들고 싶으면 w1, b1, ... 변수에 대입해주면 된다. (6th 실습 02 1:00:00)


def ComputeAccr(dloader, imodel):
    correct= 0
    total= 0
    
    for j, [imgs, labels] in enumerate(dloader): #batch_size(=1024)만큼
        img= imgs #x값
        label= Variable(labels) #model에 data를 넣을 땐 'Variable()'에 씌워주고 진행
        
        #(batch_size, 1, 28, 28) --(reshape)-> (batch_size, 28, 28)
        img= img.reshape((img.shape[0], img.shape[2], img.shape[3]))
        
        #(batch_size, 28, 28) --(reshape)-> (batch_size, 28*28) 
        #28*28: 이미지를 한 줄로 쭉 펴줌
        img = img.reshape((img.shape[0], img.shape[1]*img.shape[2]))
        img= Variable(img, requires_grad=False)
        #requires_grad=False -> foward prop
        #requires_grad=True -> backward prop
        
        #output= 1개의 dloader(=덩어리)에 대한 결과(=input layer에 대한 결과)
        output= imodel(img) #foward prop
        _, output_index= torch.max(output, 1)
        #1은 img.shape[1]을 줄여달라는 것! 
        #10 units 중 max인 unit을 뽑아서 1 unit으로 줄여줘(10 -> 1) ~.~
        
        total += label.size(0) #label: y값(정답 테이블)
        correct +=(output_index == label).sum().float()
        #output_index: y^, label: y
    print("Accuracy of Test Data: {}".format(100*correct/total)) #정확도
        
      
      
ComputeAccr(test_loader, model) #2 hidden layer 퍼셉트론 model로 정확도 구하기
      
      
loss_func= nn.CrossEntropyLoss()
optimizer= optim.SGD(model.parameters(), lr=learning_rate)
#optimization: 전달인자에 파라미터(w,b)가 들어간다.


#num_epoch= 400
netname= './nets/mip_weight.pkl' #저장할 파일위치/명
for i in range(num_epoch):
    for j, [imgs, labels] in enumerate(train_loader): #batch_size 만큼
        img=imgs #(batch_size, 1, 28, 28)
        label= Variable(labels) #(batch_size)
        
        #(batch_size, 1, 28, 28) --(reshape)-> (batch_size, 28, 28)
        img= img.reshape((img.shape[0], img.shape[2], img.shape[3]))
        
        #(batch_size, 28, 28) --(reshape)-> (batch_size, 28*28) 
        #28*28: 이미지를 한 줄로 쭉 펴줌
        img = img.reshape((img.shape[0], img.shape[1]*img.shape[2]))
        img= Variable(img, requires_grad=True)
        #requires_grad=True -> backward prop 역행하겠다~
        
        optimizer.zero_grad() #optimizer을 0으로 초기화
        output= model(img) #forward prop, img= input layer= 28*28
        loss= loss_func(output, label) #logit(# of classes), target(1)
        # class가 unit 말하는 것 맞나? ㅠ 아닌가~~
        #target(1): img.shape[1]을 10 -> 1로 줄인 output => torch.max(output, 1)
        
        loss.backward() #back prop
        optimizer.step() #weight 조정 (: w 업데이트)
        #수업 시간에 했던 학습 모델 ~.~
        #J에서부터 역행하면서 미분하고, 미분해서 더한 값을 각 unit에 저장...(동적프)
        #더이상 미분할 w가 없을 때까지(hidden layer1까지) 반복해서
        #w값 업데이트 했던 그거 하는 듯 아마...
        
        if i%50==0:
            print("%d.."%i)
            ComputeAccr(test_loader, model)
            print loss
            
            #학습된 파라미터 저장
            #netname= './nets/mip_weight.pkl' #저장할 파일위치/명
            torch.save(model, netname, )
            #model= torch.load(netname) -> 저장된 학습 model 불러오기
            
            
ComputeAccr(test_loader, model) #정확도 test
