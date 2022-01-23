import numpy as np
import cv2
from tqdm import tqdm
import time
from scipy import optimize

#New complete class, with changes:
class Neural_Network(object):
    def __init__(self, Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = 60*64
        self.outputLayerSize = 64
        self.hiddenLayerSize = 30

        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
    
    def Adam(self, X, y, lr = 1e-2, beta1 = 0.9, beta2 = 0.9, epsilon = 1e-6, num_iterations = 2000):
        loss = []
        gradients = self.computeGradients(X, y)
        #initialize first moment mt
        mt = np.zeros(len(gradients))
        #initialize second moment vt
        vt = np.zeros(len(gradients))
        for t in range(num_iterations):
            #compute gradients with respect to theta
            gradients = self.computeGradients(X, y)
            #update first moment mt as given in equation 
            mt = beta1 * mt + (1. - beta1) * gradients
            #update second moment vt as given in equation 
            vt = beta2 * vt + (1. - beta2) * gradients ** 2
            #compute bias-corected estimate of mt (21)
            mt_hat = mt / (1. - beta1 ** (t+1))
            #compute bias-corrected estimate of vt (22)
            vt_hat = vt / (1. - beta2 ** (t+1))
            #update the model parameter as given in (23)
            params = self.getParams()
            New_Params = params - (lr / (np.sqrt(vt_hat) + epsilon)) * mt_hat
            self.setParams(New_Params)
            loss.append(self.costFunction(X, y))
            
        return New_Params, loss

def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''
    

    # You may make changes here if you wish. 
    # Import Steering Angles CSV
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    total_frames=len(frame_nums)
    train_image = np.empty((total_frames, 3840))
    for i in range(total_frames):
        image = cv2.imread(path_to_images + '/' 
                        + str(int(frame_nums[i])).zfill(4) + '.jpg')
        image = cv2.resize(image, (60, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image/255
        image = np.ravel(image)
        train_image[i] = image
    bin_place = np.zeros((total_frames,1))    
    angle_bins = np.zeros((total_frames, 64))
    bins = np.linspace(-180,180,64)
    bin_place = np.digitize(steering_angles, bins)
    angle_bins_1 = np.zeros((1,64))
    for i in range(total_frames):
        a=(bin_place[i])
        b=a+1
        angle_bins_1[0][a:b]=[1]
        angle_bins[i]=angle_bins_1
        angle_bins_1 = np.zeros((1,64))

    train_angle = angle_bins
    

    
    X = train_image
    y = train_angle          
    NN=Neural_Network(Lambda=0.00001)
    loss = [] 
    New_Params,loss=NN.Adam(X, y, lr = 1e-2, beta1 = 0.9, beta2 = 0.9, epsilon = 1e-6, num_iterations = 2000)
    NN.setParams(New_Params)      
    return NN


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    image = cv2.imread(image_file)
    image = cv2.resize(image, (60, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image/255
    image = np.ravel(image)
    T=NN.forward(image)
    bins = np.linspace(-180,180,64)
    return bins[np.argmax(T)]
