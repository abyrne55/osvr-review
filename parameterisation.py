# Direct Port of constructParams.m
import numpy as np

def construct_admm_parameters(dataset, labelset, epsilon, bias, flag):
    datacells = dataset.T
    labelcells = labelset.T
    
    N = datacells.size
    T = np.zeros((N,1))
    
    num_pairs_max = 0;
    num_intensity = 0;
    
    # Collect Statistics of the Dataset
    for j in range(N):
        D = T[j] = datacells[j].size
        num_pairs_max += T[j]*(T[j]+1)/2
        num_intensity += 2*labelcells[j].size
        
    # initialize the components for OSVR problem
    # pre-allocate storage for A and e for efficiency
    # TODO: Bug? Seems to use last value assigned to D in for loop
    num_pairs_max = int(num_pairs_max.flat[0])
    #num_intensity = num_intensity.flat[0]
    
    A = np.zeros((num_intensity+num_pairs_max, D+bias))
    c = np.ones((num_intensity+num_pairs_max, 1))
    weight = np.ones((num_intensity+num_pairs_max, 1))
    idx_row_I = 0
    idx_row_P = num_intensity
    num_pairs = 0
    
    for n in range(N):
        data = datacells[n]
        label = labelcells[n]
        nframe = label.shape[0] #PN1i
        # index of apex frame

        peak = label[0].max(0)[1] #PN1i
        
        # all the indices with peak intensity
        idx = np.array([np.argmax(label[0])]).flatten()
        # select apx to be the median one of all the peak frames
        # TODO: Verify this line. Probably buggy
            
        import ipdb; ipdb.set_trace()
        apx = label[idx[int(max(0, np.ceil(len(idx) / 2.0))-1)]]

        # based on apex frame, create the ordinal set
        # number of ordinal pair
        pairs = np.zeros((T[n]*(T[n]+1)/2,2))
        dist = np.ones((T[n]*(T[n]+1)/2,1))
        count = 0
        #TODO PN1i conv probs needed
        for i in range(apx, 2, -1):
            pairs[count+1:count+i-1,1] = i
            pairs[count+1:count+i-1,2] = np.concatenate(np.arange(i - 1,1,- 1)).T
            dist[count+1:count+i-1] = np.concatenate(np.arange(1,i - 1)).T
            count += i-1
        
        if apx < T[n]:
            for i in np.arange(apx,T[n]).reshape(-1):
                pairs[count+1:count+T[n] - i,1]=i
                pairs[count+1:count+T[n] - i,2]=np.concatenate(np.arange(i + 1,T[n])).T
                dist[count+1:count+T[n] - i]=np.concatenate(np.arange(1,T[n] - i)).T
                count=count + T[n] - i
        
        pairs=pairs[1:count,:]
        dist=dist[1:count]
        num_pairs=num_pairs + count
        # compute objective function value and gradient of objective function
        dat=data[:,label[:,1]] # D*num_labels
        tij=data[:,pairs[:,1]] - data[:,pairs[:,2]] # D*num_pairs
        # assign values to parameters
        # TODO: PN1i probs needed for the next 10 lines
        A[idx_row_I + 1:idx_row_I + nframe,1:D]=dat.T
        A[idx_row_I + 1 + num_intensity / 2:idx_row_I + nframe + num_intensity / 2,1:D]=- dat.T
        A[idx_row_P + 1:idx_row_P + count,1:D]=- tij.T
        c[idx_row_I + 1:idx_row_I + nframe]=np.dot(- epsilon[1],np.ones(nframe,1)) - label[:,2]
        c[idx_row_I + 1 + num_intensity / 2:idx_row_I + nframe + num_intensity / 2]= np.dot(- epsilon[1],np.ones(nframe,1)) + label[:,2]
        c[idx_row_P + 1:idx_row_P + count]=epsilon[2]
        weight[idx_row_P + 1:idx_row_P + count]=1.0 / dist
        idx_row_I=idx_row_I + nframe
        idx_row_P=idx_row_P + count
    
    # truncate to the actual number of rows
    A=A[1:num_intensity + num_pairs,:]
    if bias: # augment A for including bias term
        A[1:num_intensity / 2,D + 1]=1
        A[1 + num_intensity / 2:num_intensity,D + 1]=- 1
    
    c=c[1:num_intensity + num_pairs,:]
    weight=weight[1:num_intensity + num_pairs]
    # unsupervisd flag to exclude all the rows associated with intensity lables
    # TODO: Below probably real buggy
    if flag:
        A=A[num_intensity + 1:,:]
        c=c[num_intensity + 1:,:]
        weight=weight[num_intensity + 1:,:]
        num_intensity=0
    
    return A,c,D,num_intensity,num_pairs,weight
        
        

            
        
        
    