
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import pylab as pl
from codecs import decode
import struct

class mlpga:
    """ A Multi-Layer Perceptron using GA learning"""
    
    def __init__(self,inputs,targets,nhidden, \
                 nEpochs, floatLength=64, \
                 populationSize=100,mutationProb=-1,crossover='un',nElite=4,tournament=True, \
                 beta=1, momentum=0.9,outtype='linear'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network 
        #self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        #self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

        self.error = np.zeros((1,2))
                
        #self.stringLength = stringLength
        self.stringLength = floatLength*nhidden*self.nin + floatLength*nhidden*self.nout

        self.floatLength = floatLength # length of a float as a satring of zeroand ones (decode)
        self.stringLength = self.stringLength + (floatLength * self.nhidden * 2)
        print("----- GA StringLength "+ str(self.stringLength))

        # Population size should be even
        if np.mod(populationSize,2)==0:
            self.populationSize = populationSize
        else:
            self.populationSize = populationSize+1

        if mutationProb < 0:
            self.mutationProb = 1/self.stringLength
        else:
             self.mutationProb = mutationProb
                   
        self.nEpochs = nEpochs
        self.crossover = crossover
        self.nElite = nElite
        self.tournment = tournament

        #self.population = np.random.rand(self.populationSize,self.stringLength)
        # Initialise network as a population of GA genes for weights1 and weights2
        self.population = np.zeros((self.populationSize,self.stringLength))
        #self.population = np.where(self.population<0.5,0,1)
        for i in range(np.shape(self.population)[0]):
            self.population[i,:] = self.initPopulationFromWeights(i)

        self.inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        self.targets = targets


    #==========================================================================
    # Functions for encode decode strings from a float
    #==========================================================================
    def bin_to_float(self,b):
        """ Convert binary string to a float. """
        bf = self.int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
        return struct.unpack('>d', bf)[0]
    
    
    def int_to_bytes(self, n, length):  # Helper function
        """ Int/long to byte string.
            Python 3.2+ has a built-in int.to_bytes() method that could be used
            instead, but the following works in earlier versions including 2.x.
        """
        return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

    def float_to_bin(self, value):  # For testing.
        """ Convert float to 64-bit binary string. """
        [d] = struct.unpack(">Q", struct.pack(">d", value))
        return '{:064b}'.format(d)

    # returns binary string
#    def float_to_bin(self, num):
#      return str(format(struct.unpack('!I', struct.pack('!f', num))[0], '032b'))
    
    # returns float
#    def bin_to_float(self, binary):
#      return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]


    def decodeFloatString(self, population, i, offset, k):
        floatStart = offset + k*self.floatLength
        floatEnd = offset + (k+1)*self.floatLength
        nextFloat = population[i,floatStart:floatEnd]
         # Converting integer list to string list 
        intStrs = [str(zeroOrOne) for zeroOrOne in nextFloat] 
        floatStr = ''.join(intStrs)
        try:
            aWeight = self.bin_to_float(floatStr)
        except ValueError:
            print('Non-numeric data found:' + floatStr)                        
        return aWeight


    def encodeFloatString(self, population, weight, i, offset, k):
         # Converting string to integer list
        floatStr = self.float_to_bin(weight)
        floatStart = offset + k*self.floatLength
        floatEnd = offset + (k+1)*self.floatLength
        intArray = [int(zeroOrOne) for zeroOrOne in floatStr] 
        population[floatStart:floatEnd] = intArray
        return population
    
    #==========================================================================
    # The population uses genes of string meaning the weights1 and weights 2 of the mlp fwd
    #==========================================================================
    #==========================================================================
    # Return create ini population from random weights
    def initPopulationFromWeights(self, i):
        # offset for point of next rows inside pop string (mapped to shape of weights)
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
        offset = 0 
        population = np.zeros(self.stringLength)
        # self.weights1
        for j in range(self.nin+1):
            for k in range(self.nhidden):
                 population = self.encodeFloatString(population, self.weights1[j,k], i, offset, k)
            offset += self.nhidden * self.floatLength
        
        # self.weights2
        for j in range(self.nhidden+1):
            for k in range(self.nout):
                 population = self.encodeFloatString(population, self.weights2[j,k], i, offset, k)
            offset += self.nout * self.floatLength

        return population
    
    # Return weights1 and weights2 
    def populationAsWeights(self, population, i):
        # offset for point of next rows inside pop string (mapped to shape of weights)
        offset = 0 
        weights1 = np.zeros(np.shape(self.weights1))
        weights2 = np.zeros(np.shape(self.weights2))
        # self.weights1
        for j in range(self.nin+1):
            for k in range(self.nhidden):
                weights1[j,k] = self.decodeFloatString(population, i, offset, k)
            offset += self.nhidden * self.floatLength
        
        # self.weights2
        for j in range(self.nhidden+1):
            for k in range(self.nout):
                weights2[j,k] = self.decodeFloatString(population, i, offset, k)
            offset += self.nout * self.floatLength

        return weights1, weights2
    
    
    # Fitness function using population size, and stringlengths to create weights1 and weights2
    # Use mlpfwd to find the weights error
    def bankfitness(self, population):
        T = 10000
        fitness = np.ones((np.shape(population)[0],1))
        population = population.astype('int')
        # Populate weights1 ( remember the one extra row for -1 bias input [nin+1] ~!~~)
        for i in range(np.shape(population)[0]):
            weights1, weights2 = self.populationAsWeights(population, i)
            self.weights1 = weights1
            self.weights2 = weights2
            validOut = self.mlpfwd(self.inputs)
            #
            try:
                error = (0.5*np.sum((validOut-self.targets)**2))
                # A percentage i couldnt find - it uses error rate and not a percentage of confusion
                # matrix
                if error > T or np.isnan(error):
                    fitness[i] = 0
                else:
                    maximize = (T - error)
                    # 10000 - 0 = 100
                    # 10000 - 245 = 9745

                    if np.isnan(maximize):
                        fitness[i] = 0
                    else:
                        fitness[i] = maximize
                        # Found the string representation of floats alters in GA crossover to
                        # overflow floats??
#                        nclasses = np.shape(self.targets)[1]
#                        if nclasses==1:
#                            nclasses = 2
#                            outputs = np.where(validOut>0.5,1,0)
#                            targets = self.targets
#                        else:
#                            # 1-of-N encoding
#                            outputs = np.argmax(validOut,1)
#                            targets = np.argmax(self.targets,1)
#                
#                        cm = np.zeros((nclasses,nclasses))
#                        for i in range(nclasses):
#                            for j in range(nclasses):
#                                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))
#                
#                        fitness[i] = np.trace(cm)/np.sum(cm)*100

            except FloatingPointError:
                fitness[i] = 0

        fitness = np.squeeze(fitness)

        return fitness
           
        
    def runGA(self):
        """The basic loop"""
        #pl.ion()
        #plotfig = pl.figure()

        bestfit = np.zeros(self.nEpochs)

        for i in range(self.nEpochs):
            # Compute fitness of the population
            fitness = self.bankfitness(self.population)
            fitness = fitness.astype('float')

            # Pick parents -- can do in order since they are randomised
            newPopulation = self.fps(self.population,fitness)

            # Apply the genetic operators
            if self.crossover == 'sp':
                newPopulation = self.spCrossover(newPopulation)
            elif self.crossover == 'un':
                newPopulation = self.uniformCrossover(newPopulation)
            newPopulation = self.mutate(newPopulation)

            # Apply elitism and tournaments if using
            if self.nElite>0:
                newPopulation = self.elitism(self.population,newPopulation,fitness)
    
            if self.tournment:
                newPopulation = self.tournament(self.population,newPopulation,fitness)#,self.fitnessFunction)
    
            self.population = newPopulation
            bestfit[i] = fitness.max()

            if (np.mod(i,100)==0):
                print (i, fitness.max())
            #pl.plot([i],[fitness.max()],'r+')

        self.population = self.population.astype('int')
        bestp = np.where(fitness==fitness.max())
        print('bestp =' , bestp, bestp[0], bestp[0][0])
        self.populationAsWeights(self.population, bestp[0][0])
        pl.title("Best fit equilibrium of the fitness result" )
        pl.plot(bestfit,'kx-')
        #pl.show()
    
    def fps(self,population,fitness):

        # Scale fitness by total fitness
        fitness = fitness/np.sum(fitness)
        fitness = 10.*fitness/fitness.max()
        
        # Put repeated copies of each string in according to fitness
        # Deal with strings with very low fitness
        j=0
        while j<len(fitness) and np.round(fitness[j])<1: 
            j = j+1
        
        newPopulation = np.kron(np.ones((int(np.round(fitness[j])),1)),population[j,:])

        # Add multiple copies of strings into the newPopulation
        for i in range(j+1,self.populationSize):
               #print(np.round(fitness[i]))
            #try:
               if np.round(fitness[i])>=1:
                   newPopulation = np.concatenate((newPopulation,np.kron(np.ones((int(np.round(fitness[i])),1)),population[i,:])),axis=0)
            #except:
            #    pass

        # Shuffle the order (note that there are still too many)
        indices = np.arange(np.shape(newPopulation)[0])

        np.random.shuffle(indices)

        newPopulation = newPopulation[indices[:self.populationSize],:]
        
        newsizepop = np.shape(newPopulation)[0]
        if newsizepop < self.populationSize:
            # add that many more in
            #print("Problem with population size shrunk!")
            while np.shape(newPopulation)[0] < self.populationSize:
               newPopulation = np.concatenate((newPopulation,newPopulation),axis=0)
            newPopulation = newPopulation[:self.populationSize,:]
               
            
        return newPopulation    

    def spCrossover(self,population):
        # Single point crossover
        newPopulation = np.zeros(np.shape(population))
        crossoverPoint = np.random.randint(0,self.stringLength,self.populationSize)
        #print("crossover ", crossoverPoint, np.shape(population))
        #try:
        for i in range(0,self.populationSize,2):
                newPopulation[i,:crossoverPoint[i]] = population[i,:crossoverPoint[i]]
                newPopulation[i+1,:crossoverPoint[i]] = population[i+1,:crossoverPoint[i]]
                newPopulation[i,crossoverPoint[i]:] = population[i+1,crossoverPoint[i]:]
                newPopulation[i+1,crossoverPoint[i]:] = population[i,crossoverPoint[i]:]
        #except:
        #    print("Cross over problem")
        return newPopulation

    def uniformCrossover(self,population):
        # Uniform crossover
        newPopulation = np.zeros(np.shape(population))
        which = np.random.rand(self.populationSize,self.stringLength)
        which1 = which>=0.5
        for i in range(0,self.populationSize,2):
            newPopulation[i,:] = population[i,:]*which1[i,:] + population[i+1,:]*(1-which1[i,:])
            newPopulation[i+1,:] = population[i,:]*(1-which1[i,:]) + population[i+1,:]*which1[i,:]
        return newPopulation
        
    def mutate(self,population):
        # Mutation
        whereMutate = np.random.rand(np.shape(population)[0],np.shape(population)[1])
        population[np.where(whereMutate < self.mutationProb)] = 1 - population[np.where(whereMutate < self.mutationProb)]
        return population

    def elitism(self,oldPopulation,population,fitness):
        best = np.argsort(fitness)
        best = np.squeeze(oldPopulation[best[-self.nElite:],:])
        indices = np.arange(np.shape(population)[0])
        np.random.shuffle(indices)
        population = population[indices,:]
        population[0:self.nElite,:] = best
        return population

    def tournament(self,oldPopulation,population,fitness):#,fitnessFunction):
        newFitness = self.bankfitness(population)
        for i in range(0,np.shape(population)[0],2):
            f = np.concatenate((fitness[i:i+2],newFitness[i:i+2]),axis=0)
            #f = np.concatenate((fitness[i:i+2],newFitness[i:i+2]),axis=1)
            indices = np.argsort(f)
            if indices[-1]<2 and indices[-2]<2:
                population[i,:] = oldPopulation[i,:]
                population[i+1,:] = oldPopulation[i+1,:]
            elif indices[-1]<2:
                if indices[0]>=2:
                    population[i+indices[0]-2,:] = oldPopulation[i+indices[-1]]
                else:
                    population[i+indices[1]-2,:] = oldPopulation[i+indices[-1]]
            elif indices[-2]<2:
                if indices[0]>=2:
                    population[i+indices[0]-2,:] = oldPopulation[i+indices[-2]]
                else:
                    population[i+indices[1]-2,:] = oldPopulation[i+indices[-2]]
        return population
    
    
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print("error")

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        #print("BestFit at epoch end ", self.bestFit[self.nEpochs-1])
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)
