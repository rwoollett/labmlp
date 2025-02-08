
# Rodney Woollett 91109565
# See "Main process" at end of file to change part to 1,2, or 3

#import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import mlp
import mlpga
import som
import pcn

def readPreProcessBank(bankfile, part):
    # Read the dataset in (code from sheet)
    # Convert categorized string into numeric values for calculation
    job2Nm = {"admin.":0.1,"blue-collar":0.2,"entrepreneur":0.3,"housemaid":0.4,"management":0.5, \
          "retired":0.6,"self-employed":0.7,"services":0.8,"student":0.9,"technician":0.10, \
          "unemployed":0.11,"unknown":0.12}
    #marital = {"divorced":0.1,"married":0.2,"single":0.3,"unknown":0.4}
    #education = {"basic.4y":0.1,"basic.6y":0.2,"basic.9y":0.3,"high.school":0.4,"illiterate":0.5, \
    #       "professional.course":0.6, "university.degree":0.7, "unknown":0.8}
    contact = {"cellular":0.1,"telephone":0.2}
    month = {"jan":0.1, "feb":0.2, "mar":0.3, "apr":0.4, "may":0.5, "jun":0.6, "jul":0.7, "aug":0.8, \
         "sep":0.9, "oct":0.10, "nov":0.11, "dec":0.12}
    #dow = {"mon":0.1,"tue":0.2,"wed":0.3,"thu":0.4,"fri":0.5}
    poutcome = {"failure":0.1,"nonexistent":0.2,"success":0.3}
    yesOrNo = {'yes':1,'no':0}
    #yesOrNoOrUnknown = {'unknown': 0.3, 'yes':0.2,'no':0.1}
#    job2Nm = {"admin.":1,"blue-collar":2,"entrepreneur":3,"housemaid":4,"management":5, \
#          "retired":6,"self-employed":7,"services":8,"student":9,"technician":10, \
#          "unemployed":11,"unknown":12}
#    marital = {"divorced":1,"married":2,"single":3,"unknown":4}
#    education = {"basic.4y":1,"basic.6y":2,"basic.9y":3,"high.school":4,"illiterate":5, \
#           "professional.course":6, "university.degree":7, "unknown":8}
#    contact = {"cellular":1,"telephone":2}
#    month = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, \
#         "sep":9, "oct":10, "nov":11, "dec":12}
#    dow = {"mon":1,"tue":2,"wed":3,"thu":4,"fri":5}
#    poutcome = {"failure":1,"nonexistent":2,"success":3}
#    yesOrNo = {'yes':1,'no':0}
#    yesOrNoOrUnknown = {'unknown': 3, 'yes':2,'no':1}

    # Quoted string from reader are stripped of quotes - its all lowercase in file
    convert2yn = lambda x: yesOrNo[x.strip('"')]
    #convert2ynk = lambda x: yesOrNoOrUnknown[x.strip('"')]
    convert2job = lambda x: job2Nm[x.strip('"')]
    #convert2marital = lambda x: marital[x.strip('"')]
    #convert2education = lambda x: education[x.strip('"')]
    convert2contact = lambda x: contact[x.strip('"')]
    convert2month = lambda x: month[x.strip('"')]
    #convert2dow = lambda x: dow[x.strip('"')]
    convert2poutcome = lambda x: poutcome[x.strip('"')]
    #convert2duration = lambda x: float(x or 1)# - float(x))  # set as no change in all (removed)
    convert2Float = lambda x: float(x.strip('"'))
    
    # Read CSV file with loadtxt
     # social and economic context attributes could be removed to avoid classifiying into group
     #15,16,17,18,19, dow 9 edu 3 other numrical data,11,12,13,14,  ,3,8,9,14 4,5,6,7,
    if part == 1 or part ==2:
        bank = np.loadtxt(bankfile , delimiter=';', skiprows=1, \
                   usecols = (1,7,8,14,15,16,17,19,20 ), \
                    converters = {\
                                  1: convert2job, \
                                  #2: convert2marital, \
                                  #3: convert2education, \
                                  #4: convert2ynk, \
                                  #5: convert2ynk, 
                                  #6: convert2ynk, \
                                  7: convert2contact, 
                                  8: convert2month, \
                                  #9: convert2dow, 
                                  #10: convert2duration, \
                                  14: convert2poutcome, \
                                  # social and economic context attributes numerical
                                  # except daily indicator of euribor3m
                                  15: convert2Float, \
                                  16: convert2Float, \
                                  17: convert2Float, \
                                  19: convert2Float, \
                                  # outcome yes or no
                                  20: convert2yn}, encoding="ascii")
    else: # part 3   
        #
        bank = np.loadtxt(bankfile , delimiter=';', skiprows=1, \
                   usecols = (1,7,8,14,15,16,17,19,20 ), \
                    converters = {\
                                  1: convert2job, \
                                  #2: convert2marital, \
                                  #3: convert2education, \
                                  #4: convert2ynk, \
                                  #5: convert2ynk, 
                                  #6: convert2ynk, \
                                  7: convert2contact, 
                                  8: convert2month, \
                                  #9: convert2dow, 
                                  #10: convert2duration, \
                                  14: convert2poutcome, \
                                  # social and economic context attributes numerical
                                  # except daily indicator of euribor3m
                                  15: convert2Float, \
                                  16: convert2Float, \
                                  17: convert2Float, \
                                  19: convert2Float, \
                                  # outcome yes or no
                                  20: convert2yn}, encoding="ascii")

    # Ages groups into 5 groups
    #bank[np.where(bank[:,0]<=30),0] = 1
    #bank[np.where((bank[:,0]>30) & (bank[:,0]<=50)),0] = 2
    #bank[np.where(bank[:,0]>50),0] = 3
    #bank[np.where((bank[:,0]>40) & (bank[:,0]<=50)),0] = 3
    #bank[np.where((bank[:,0]>50) & (bank[:,0]<=60)),0] = 4

    return bank
     

# get 50 / 50 of yes to no outpcomes
def getEqualTargets(data):
    shape_of_data = np.shape(data)
    # find yes and no
    yes_rows = np.where(data[:, shape_of_data[1] - 1] == 1)
    no_rows = np.where(data[:, shape_of_data[1] - 1] == 0)
    nselect = len(yes_rows[0]) if len(yes_rows[0]) < len(no_rows[0]) else len(no_rows[0])

    # Randomise index
    np.random.shuffle(yes_rows[0])
    np.random.shuffle(no_rows[0])
    yes_set = data[yes_rows[0][:nselect], :]
    no_set = data[no_rows[0][:nselect], :]
    data_set = np.concatenate((yes_set, no_set), axis=0)
    np.random.shuffle(data_set)
    return data_set


# Using n-k hold out cross validation create the set at k network 
def kNetwork(kindex, k, train, tgt):
    data_sp = np.shape(train)
    amt_block = int(data_sp[0]//k)
    
    train_in = []
    train_tgt = []
    valid_in = []
    valid_tgt = []
    #print("train",np.shape(train))
    isFirstDone = False
    for kIter in range(k):
        chunks =   train[kIter*amt_block:(kIter+1)*amt_block+1,:]
        chunks_tgt = tgt[kIter*amt_block:(kIter+1)*amt_block+1,:]
        # kindex is point for valid set
        if kIter+1 == kindex: # 
            valid_in = chunks
            valid_tgt = chunks_tgt
        elif not isFirstDone:
            train_in = chunks
            train_tgt = chunks_tgt
            isFirstDone = True
        else:
            #print("knet",data_sp,amt_block,np.shape(train_in), np.shape(chunks))
            train_in = np.concatenate((train_in, chunks),axis=0)
            train_tgt = np.concatenate((train_tgt, chunks_tgt),axis=0)
    
    print("knet", np.shape(train_in), np.shape(valid_in), np.shape(train_tgt), np.shape(valid_tgt))
    
    return (train_in, valid_in, train_tgt, valid_tgt)


# The mlp process
def mlpProcess(cvaltrain_in, cvaltrian_tgt, test_in,test_tgt):

    hiddenAmountWithColours = {1:['','rs','gv','b^'],2:['','rs','gv','b^'], 3:['','rs','gv','b^']}
    #,10:'bo',20:'co',50:'ko'}
    count = 0
    best_val_error = 100000
    best_k_network = np.nan
    best_net = None #mlp.mlp(cvaltrain_in,cvaltrian_tgt,i,beta=1,momentum=0.85,outtype='linear')

    for i in hiddenAmountWithColours.keys(): #[1,2,5,10,20,50]:
        print("----- "+str(i))
        # n-k hold out cross validate
        for k in range(1,4):
            (ktrain_in, kvalid_in, ktrian_tgt, kvalid_tgt) = kNetwork(k, 4, cvaltrain_in, cvaltrian_tgt)
            # mlp with a k th network
            net = mlp.mlp(ktrain_in,ktrian_tgt,i,beta=1,momentum=0.85,outtype='linear')
            #net.mlptrain(train_in,train_tgt,0.1,1000)
            valid_error = net.earlystopping(ktrain_in,ktrian_tgt,kvalid_in,kvalid_tgt,0.1)
            if valid_error < best_val_error:
                best_val_error = valid_error
                best_k_network = k
                best_net = net

            print("current network",k, valid_error)
            indicesn = np.where(net.error[:,0]>0)

            plt.plot(net.error[indicesn,0], net.error[indicesn,1],hiddenAmountWithColours[i][k])

        plt.title("Figure " + str(count + 1) + ": KCross with "+ str(i) + " hidden layers") 
        plt.figure(count+2)
        count += 1
                
    # get network with best valid_error
    print("best network",best_k_network, best_val_error)
    best_net.confmat(test_in,test_tgt)
    

    
# The mlp process with GA
def mlpProcessGA(cvaltrain_in, cvaltrian_tgt, test_in, test_tgt):
    
    #np.seterr(all='raise')
    hiddenAmountWithColours = {2:'go'}#,2:'c*'}#,10:'bo',20:'co',50:'ko'}
    for hiddenAmount in hiddenAmountWithColours.keys(): #[1,2,5,10,20,50]:
        #str_size = 64*i*data_sp[1]+64*i*np.shape(cvaltrian_tgt)[1]
        print("-----hidden layers "+str(hiddenAmount))
        net = mlpga.mlpga(cvaltrain_in,cvaltrian_tgt, hiddenAmount, \
                          301, floatLength=64, populationSize=100, \
                          #mutationProb=-1, \
                          mutationProb=0.09, \
                          crossover='sp' , nElite=4, tournament=False)
        net.runGA()
        net.confmat(test_in,test_tgt)

 
# The SOM classify and perceptron SOM classify
def processSOMClassify(cvaltrain_in, cvaltrian_tgt, test_in, test_tgt):
    
    score = np.zeros((9,1))
    count = 0
    best_score = 0
    best_net = None 
    best_count = 0
    for x in [20]:
        for y in [20]:
            net = som.som(x, y, cvaltrain_in, usePCA=0)
            net.somtrain(cvaltrain_in,100)
    
            # Store the best node for each training input
            best = np.zeros(np.shape(cvaltrain_in)[0],dtype=int)
            for i in range(np.shape(cvaltrain_in)[0]):
                best[i],activation = net.somfwd(cvaltrain_in[i,:])

            plt.plot(net.map[0,:],net.map[1,:],'k.',ms=6)
            where0 = np.where(cvaltrian_tgt[:,0] == 0)
            plt.plot(net.map[0,best[where0]],net.map[1,best[where0]],'rs',ms=12)
            where1 = np.where(cvaltrian_tgt[:,0] == 1)
            plt.plot(net.map[0,best[where1]],net.map[1,best[where1]],'gv',ms=12)
            plt.axis([-0.1,1.1,-0.1,1.1])
            plt.axis('off')
            plt.title("Figure "+ str(count + 1))
            plt.figure(count+2)
            
            # Find places where the same neuron represents different classes
            i0 = np.where(cvaltrian_tgt[:,0]==0)
            nodes0 = np.unique(best[i0])
            i1 = np.where(cvaltrian_tgt[:,0]==1)
            nodes1 = np.unique(best[i1])
        
            doubles = np.in1d(nodes0,nodes1,assume_unique=True)
            score[count] = x*y + 10 * len(nodes0[doubles])
            if score[count] > best_score:
                best_net = net
                best_count = count + 1
            count += 1

    print (score)
    
    # Now pick the best and use a perceptron with the activations
    print (np.argmax(score))
    print(best , np.shape(cvaltrain_in))
    print("Output after SOM train and select best SOM network (Figure " + str(best_count) + ")")

    # Get activations from SOM
    activations = np.zeros(np.shape(cvaltrain_in))
    activations = np.zeros((np.shape(cvaltrain_in)[0],(best_net.x*best_net.y)))
    for i in range(np.shape(cvaltrain_in)[0]):
        best, activations[i,:] = best_net.somfwd(cvaltrain_in[i,:])

    # See the train input and confusion matrix with perceptron
    # Use perceptron as SOM activatrions as input - this does reflect to how 
    # bad the SOM cant classify the the bank data. Always a neoron best that has two classifications
    print("Training data:")
    print(best , np.shape(activations))
    p1 = pcn.pcn(activations, cvaltrian_tgt)
    p1.pcntrain(activations, cvaltrian_tgt,0.25,100)
    p1.confmat(activations, cvaltrian_tgt)

    # Use the test input with best network SOM activation
    activations = np.zeros(np.shape(test_in))
    activations = np.zeros((np.shape(test_in)[0],(best_net.x*best_net.y)))
    for i in range(np.shape(test_in)[0]):
        best, activations[i,:] = best_net.somfwd(test_in[i,:])

    print(best , np.shape(activations))
    print("Test data with Best Network SOM from training:")
    p1 = pcn.pcn(activations, test_tgt)
    p1.pcntrain(activations, test_tgt,0.25,100)
    p1.confmat(activations, test_tgt)

    # And now, using the test input to train SOM then use the SOM activations
    # Not sure if test data should be used to train the SOM as it should training data for learning
    # the SOM to be correct.
  #  testnet = som.som(best_net.x, best_net.y, test_in, usePCA=0)
  #  testnet.somtrain(test_in,100)

  #  activations = np.zeros(np.shape(test_in))
  #  activations = np.zeros((np.shape(test_in)[0],(best_net.x*best_net.y)))
  #  for i in range(np.shape(test_in)[0]):
  #      best, activations[i,:] = testnet.somfwd(test_in[i,:])

  #  print(best , np.shape(activations))
  #  print("Test data with training SOM with test data:")
  #  p1 = pcn.pcn(activations, test_tgt)
  #  p1.pcntrain(activations, test_tgt,0.25,100)
  #  p1.confmat(activations, test_tgt)
    
    
#==============================================================================
#==============================================================================
# Main process
#==============================================================================
#==============================================================================

parts = np.array([1,2,3])
part = [1] # change to which part 1 - MLP, 2 - MLP with GA, 3 - SOM
mask = np.in1d(parts,part)
bank = readPreProcessBank('bank-additional.csv', parts[mask])                 

# plt.ion()
bank_sp = np.shape(bank)

# Different type of normalisatin for MLP train and SOM
if parts[mask] == 1 or parts[mask] == 2:
    # Do the normalize on the data
    #bank[:,:bank_sp[1]-1] = bank[:,:bank_sp[1]-1]-bank[:,:bank_sp[1]-1].mean(axis=0)
    #bank[:,:bank_sp[1]-1] = bank[:,:bank_sp[1]-1]/bank[:,:bank_sp[1]-1].var(axis=0)

    bank[:,:bank_sp[1]-1] = bank[:,:bank_sp[1]-1]-bank[:,:bank_sp[1]-1].mean(axis=0) 
    imax = np.concatenate((bank.max(axis=0)*np.ones((1,bank_sp[1])),np.abs(bank.min(axis=0))*np.ones((1,bank_sp[1]))),axis=0).max(axis=0) 
    bank[:,:bank_sp[1]-1] = bank[:,:bank_sp[1]-1]/imax[:bank_sp[1]-1]
    
elif parts[mask] == 3:
    # Elucidain normialisatoin
#    normalisers = np.sqrt(np.sum(bank**2,axis=1)) * \
#                      np.ones((1,np.shape(bank)[0])) 
#    bank = np.transpose(np.transpose(bank)/normalisers)
# This normalisation did not look to keep the actual values in conversion (wasn't sure why 0.2/?
#    normalisers = np.sqrt(np.sum(bank[:,:bank_sp[1]-1]**2,axis=1)) * \
#                      np.ones((1,np.shape(bank[:,:bank_sp[1]-1])[0])) 
#    bank[:,:bank_sp[1]-1] = np.transpose(np.transpose(bank[:,:bank_sp[1]-1])/normalisers)
    # used this one to normalize
    bank[:,:bank_sp[1]-1] = bank[:,:bank_sp[1]-1]-bank[:,:bank_sp[1]-1].mean(axis=0)
    imax = np.concatenate((bank.max(axis=0)*np.ones((1,bank_sp[1])),bank.min(axis=0)*np.ones((1,bank_sp[1]))),axis=0).max(axis=0)
    bank[:,:bank_sp[1]-1] = bank[:,:bank_sp[1]-1]/imax[:bank_sp[1]-1]
    bank = getEqualTargets(bank)

    
#bank = getEqualTargets(bank)
order = np.arange(np.shape(bank)[0])
np.random.shuffle(order)
bank = bank[order,:]
#np.random.shuffle(bank)

test_blk = int(bank_sp[0] * 10//50)

cvaltrain_in =  bank[ test_blk:, :bank_sp[1]-1] #input features in the training cross validate
cvaltrian_tgt = bank[ test_blk:, bank_sp[1]-1:bank_sp[1]] #corresponding class labels for the cross validate

test_in =  bank[ :test_blk,  :bank_sp[1]-1] #input features for testing
test_tgt = bank[ :test_blk,  bank_sp[1]-1:bank_sp[1]] #corresponding class labels for the test set


if parts[mask] == 1:
    #-------------------------------------------------------------
    # MLP part 1
    mlpProcess(cvaltrain_in, cvaltrian_tgt, test_in, test_tgt)

    print("MLP kCross\n")
    #-------------------------------------------------------------

elif parts[mask] == 2:
    #-------------------------------------------------------------
    # MLP with GA part 2
    mlpProcessGA(cvaltrain_in, cvaltrian_tgt, test_in, test_tgt)
    
    print("MLP with GA\n")
    #-------------------------------------------------------------
    
elif parts[mask] == 3:
    #-------------------------------------------------------------
    # SOM classifiy and Perceptron with SOM input
    processSOMClassify(cvaltrain_in, cvaltrian_tgt, test_in, test_tgt)
    
    print("SOM classify and perceptron with SOM input\n")
    #-------------------------------------------------------------

    
print("Bank count", bank_sp[0], np.shape(bank), "test blk count", test_blk)
print("Cross size", np.shape(cvaltrain_in), "Test size", np.shape(test_in))
print(np.shape(bank))
print(np.shape(test_in))

plt.show()


