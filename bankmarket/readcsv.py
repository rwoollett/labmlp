
# Rodney Woollett 91109565
import pylab as pl
import numpy as np
import mlp

# get 50 / 50 of yes to no outpcomes
def get_fair_number_of_samples(bank_data, shape_of_data):
    # get fair number of samples from each result set
    yes_rows = np.where(bank_data[:, shape_of_data[1] - 1] == 1)
    no_rows = np.where(bank_data[:, shape_of_data[1] - 1] == 0)

    # How many rows to be selected from each class
    nselect = len(yes_rows[0]) if len(yes_rows[0]) < len(no_rows[0]) else len(no_rows[0])

    # Randomly shuffle indices
    np.random.shuffle(yes_rows[0])
    np.random.shuffle(no_rows[0])

    # get nselect # of rows from each set
    yes_set = bank_data[yes_rows[0][:nselect], :]
    no_set = bank_data[no_rows[0][:nselect], :]

    data_set = np.concatenate((yes_set, no_set), axis=0)
    np.random.shuffle(data_set)
    return data_set


# Read the dataset in (code from sheet)
# Convert categorized string into numeric values for calculation
job2Nm = {"admin.":0,"blue-collar":1,"entrepreneur":2,"housemaid":3,"management":4, \
          "retired":5,"self-employed":6,"services":7,"student":8,"technician":9, \
          "unemployed":10,"unknown":11}
marital = {"divorced":0,"married":1,"single":2,"unknown":3}
education = {"basic.4y":0,"basic.6y":1,"basic.9y":2,"high.school":3,"illiterate":4, \
           "professional.course":5, "university.degree":6, "unknown":7}
contact = {"cellular":0,"telephone":1}
month = {"jan":0, "feb":1, "mar":2, "apr":3, "may":4, "jun":5, "jul":6, "aug":7, \
         "sep":8, "oct":9, "nov":10, "dec":11}
dow = {"mon":0,"tue":1,"wed":2,"thu":3,"fri":4}
poutcome = {"failure":0,"nonexistent":1,"success":2}
yesOrNo = {'yes':1,'no':0}
yesOrNoOrUnknown = {'unknown': 2, 'yes':1,'no':0}

# Quoted string from reader are stripped of quotes - its all lowercase in file
convert2yn = lambda x: yesOrNo[x.strip('"')]
convert2ynk = lambda x: yesOrNoOrUnknown[x.strip('"')]
convert2job = lambda x: job2Nm[x.strip('"')]
convert2marital = lambda x: marital[x.strip('"')]
convert2education = lambda x: education[x.strip('"')]
convert2contact = lambda x: contact[x.strip('"')]
convert2month = lambda x: month[x.strip('"')]
convert2dow = lambda x: dow[x.strip('"')]
convert2poutcome = lambda x: poutcome[x.strip('"')]
convert2duration = lambda x: 1 # set as no change in all (removed)
 
# Read CSV file with loadtxt
bank = np.loadtxt( 'bank-additional-full.csv' , delimiter=';', skiprows=1, \
                   usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 ), \
                    converters = {1: convert2job, 2: convert2marital, \
                                  3: convert2education, 4: convert2ynk, \
                                  5: convert2ynk, 6: convert2ynk, \
                                  7: convert2contact, 8: convert2month, \
                                  9: convert2dow, 10: convert2duration, \
                                  14: convert2poutcome, \
                                  20: convert2yn}, encoding="ascii")

 
# Some checks                        
#bank = get_fair_number_of_samples(bank,np.shape(bank))
print(np.shape(bank))
bank_sp = np.shape(bank)
#print("Outcomes")
indices0 = np.where(bank[:,bank_sp[1]-1]]==0)
indices1 = np.where(bank[:,bank_sp[1]-1]]==1)


# Do the normalize on the data
#bank[:,:19] = bank[:,:19]-bank[:,:19].mean(axis=0)
#bank[:,:19] = bank[:,:19]/bank[:,:19].var(axis=0)
# Elucidain normialisatoin
normalisers = np.sqrt(np.sum(bank[:,:bank_sp[1]-1]**2,axis=1)) * \
                      np.ones((1,np.shape(bank[:,:bank_sp[1]-1])[0])) 
bank[:,:bank_sp[1]-1] = np.transpose(np.transpose(bank[:,:bank_sp[1]-1])/normalisers)


#print('pima after norm', pima[1,:8] )
train_in = bank[::2,:bank_sp[1]-1] #input features for training
test_in = bank[1::4,:bank_sp[1]-1] #input features in the test set
valid_in = bank[3::4,:bank_sp[1]-1] #input features in the validation set

train_tgt = bank[::2,bank_sp[1]-1:bank_sp[1]] #corresponding class labels for the training set
test_tgt = bank[1::4,bank_sp[1]-1:bank_sp[1]] #corresponding class labels for the test set
valid_tgt = bank[3::4,bank_sp[1]-1:bank_sp[1]] #corresponding class labels for the valid set


pl.ion()
colours = {1:'yo', 2:'go'}#, 3:'r*', 5:'ro'}#,10:'bo',20:'co',50:'ko'}
#colours = {1:'yo', 2:'go'}#, 3:'r*'}#, 5:'ro'}#,10:'bo',20:'co',50:'ko'}
#colours = {5:'ro',10:'bo'}#,20:'co',50:'ko'}#, 100:'go'}
#colours = {20:'co',50:'ko', 80:'go', 100:'ro', 150:'yo'}
for i in colours.keys(): #[1,2,5,10,20,50]:
    print("----- "+str(i))
    net = mlp.mlp(train_in,train_tgt,i,outtype='linear')
    net.earlystopping(train_in,train_tgt,valid_in,valid_tgt,0.1)
    #net.mlptrain(train_in,train_tgt,0.1,100)
    
    net.confmat(test_in,test_tgt)

#    inputs = np.concatenate((test_in,-np.ones((np.shape(test_in)[0],1))),axis=1)
#    outputs = net.mlpfwd(inputs)
   
    indicesn = np.where(net.error[:,0]>0)
    pl.plot(net.error[indicesn,0], net.error[indicesn,1],colours[i])

print(np.shape(bank))
print(np.shape(train_in))
print(np.shape(valid_in))
print(np.shape(test_in))
pl.show()
