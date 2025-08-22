# 퍼셉트론
신경망

# ANN
ANN은 DNN을 포함한다.
은닉층이 존재한다


# dropout
훈련할때만 사용해야한다.

# Freeze & Mode(eval, train)

훈련 모드시 dropout을 활성화하고
평가 모드시 dropout을 비활성화 하는 것을 자동으로 해줌.

# Autograd
동적 학습

/* nn 실습 */
import torch
import torch.nn as nn

fc1 = nn.Linear(3,1)

print("Weight requires_grad:", fc1.weight.requires_grad)
print("Bias requires_grad:", fc1.bias.requires_grad)


/* 코드 설명 */
// ANN_with_Pytorch.py
optimizer.zero_grad() // 기존의 그리드를 초기화함.

output = model(data)
loss = criterion(output, target) // 그래프를 그림.

loss.backward() // gradient를 구하는 연산.

optimizer.step() // 계산한 gradient 저장.