import math

def inverse_sigmoid(x):
    return math.log(x / (1 - x))


print(inverse_sigmoid(0.01))
print(inverse_sigmoid(0.99))