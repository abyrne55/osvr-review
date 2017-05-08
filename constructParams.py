import numpy
from smop.core import *

# OSVR/constructParams.m

    
@function
def constructParams(dataset=None,labelset=None,epsilon=None,bias=None,flag=None,*args,**kwargs):
    varargin = constructParams.varargin
    nargin = constructParams.nargin

    # formalize model parameters used by admm solver for OSVR
    
#     if logical_not(iscell(dataset)):
#         datacells[1]=dataset
# # OSVR/constructParams.m:5
#         labelcells[1]=labelset
# # OSVR/constructParams.m:6
#     else:
#         datacells=copy(dataset)
# # OSVR/constructParams.m:8
#         labelcells=copy(labelset)
# # OSVR/constructParams.m:9

    datacells=copy(dataset)
    labelcells=copy(labelset)
    
    N=numel(datacells)
# OSVR/constructParams.m:11
    T=zeros(N,1)
# OSVR/constructParams.m:12
    num_pairs_max=0
# OSVR/constructParams.m:13
    num_intensity=0
# OSVR/constructParams.m:14
    # collect statistics of the dataset
    for n in arange(1,N).reshape(-1):
        D,T[n]=size(datacells[n],nargout=2)
# OSVR/constructParams.m:17
        num_pairs_max=num_pairs_max + dot(T[n],(T[n] + 1)) / 2
        num_pairs_max=int(num_pairs_max.flat[0])
# OSVR/constructParams.m:18
        num_intensity=num_intensity + dot(2,size(labelcells[n],1))
# OSVR/constructParams.m:19
    
    # initialize the components for OSVR problem
# pre-allocate storage for A and e for efficiency
    import ipdb; ipdb.set_trace()
    #A=zeros(num_intensity + num_pairs_max,D + bias)
    #A=numpy.zeros(122539,178)
    A=matlabarray(zeros(shape=(122539,178)))
# OSVR/constructParams.m:23
    c=ones(num_intensity + num_pairs_max,1)
# OSVR/constructParams.m:24
    weight=ones(num_intensity + num_pairs_max,1)
# OSVR/constructParams.m:25
    idx_row_I=0
# OSVR/constructParams.m:26
    idx_row_P=copy(num_intensity)
# OSVR/constructParams.m:27
    num_pairs=0
# OSVR/constructParams.m:28
    for n in arange(1,N).reshape(-1):
        data=datacells[n]
# OSVR/constructParams.m:30
        label=labelcells[n]
# OSVR/constructParams.m:31
        nframe=size(label,1)
# OSVR/constructParams.m:32
        peak=max(label[:,2])
# OSVR/constructParams.m:33
        idx=find(label[:,2] == peak)
# OSVR/constructParams.m:34
        apx=label[idx[max(1,ceil(length(idx) / 2))],1]
# OSVR/constructParams.m:35
        # based on apex frame, create the ordinal set
    # number of ordinal pair
        pairs=zeros(dot(T[n],(T[n] + 1)) / 2,2)
# OSVR/constructParams.m:38
        dist=ones(dot(T[n],(T[n] + 1)) / 2,1)
# OSVR/constructParams.m:39
        count=0
# OSVR/constructParams.m:40
        for i in arange(apx,2,- 1).reshape(-1):
            pairs[count + 1:count + i - 1,1]=i
# OSVR/constructParams.m:42
            pairs[count + 1:count + i - 1,2]=cat(arange(i - 1,1,- 1)).T
# OSVR/constructParams.m:43
            dist[count + 1:count + i - 1]=cat(arange(1,i - 1)).T
# OSVR/constructParams.m:44
            count=count + i - 1
# OSVR/constructParams.m:45
        if apx < T[n]:
            for i in arange(apx,T[n]).reshape(-1):
                pairs[count + 1:count + T[n] - i,1]=i
# OSVR/constructParams.m:49
                pairs[count + 1:count + T[n] - i,2]=cat(arange(i + 1,T[n])).T
# OSVR/constructParams.m:50
                #import ipdb; ipdb.set_trace()
                #PORT NOTE, check for accy, added ",1" to index on LHS
                dist[count + 1:count + T[n] - i,1]=cat(arange(1,T[n] - i)).T
# OSVR/constructParams.m:51
                count=count + T[n] - i
# OSVR/constructParams.m:52
        pairs=pairs[1:count,:]
# OSVR/constructParams.m:55
        #PORT NOTE, check for accy, added ",1" to index on RHS
        dist=dist[1:count,1]
# OSVR/constructParams.m:56
        num_pairs=num_pairs + count
# OSVR/constructParams.m:57
        dat=data[:,label[:,1]]
# OSVR/constructParams.m:59
        tij=data[:,pairs[:,1]] - data[:,pairs[:,2]]
# OSVR/constructParams.m:60
        # assign values to parameters
        A[idx_row_I + 1:idx_row_I + nframe,1:D]=dat.T
# OSVR/constructParams.m:62
        A[idx_row_I + 1 + num_intensity / 2:idx_row_I + nframe + num_intensity / 2,1:D]=- dat.T
# OSVR/constructParams.m:63
        A[idx_row_P + 1:idx_row_P + count,1:D]=- tij.T
# OSVR/constructParams.m:64
        c[idx_row_I + 1:idx_row_I + nframe]=dot(- epsilon[1],ones(nframe,1)) - label[:,2]
# OSVR/constructParams.m:65
        c[idx_row_I + 1 + num_intensity / 2:idx_row_I + nframe + num_intensity / 2]=dot(- epsilon[1],ones(nframe,1)) + label[:,2]
# OSVR/constructParams.m:66
        #PORT NOTE, check for accy, added ",1" to index on LHS and changed 2 to 1 on RHS
        c[idx_row_P + 1:idx_row_P + count,1]=epsilon[1]
# OSVR/constructParams.m:67
        import ipdb; ipdb.set_trace()
        weight[idx_row_P + 1:idx_row_P + count]=1.0 / dist
# OSVR/constructParams.m:69
        idx_row_I=idx_row_I + nframe
# OSVR/constructParams.m:70
        idx_row_P=idx_row_P + count
# OSVR/constructParams.m:71
    
    # truncate to the actual number of rows
    A=A[1:num_intensity + num_pairs,:]
# OSVR/constructParams.m:74
    if bias:
        A[1:num_intensity / 2,D + 1]=1
# OSVR/constructParams.m:76
        A[1 + num_intensity / 2:num_intensity,D + 1]=- 1
# OSVR/constructParams.m:77
    
    c=c[1:num_intensity + num_pairs,:]
# OSVR/constructParams.m:79
    weight=weight[1:num_intensity + num_pairs]
# OSVR/constructParams.m:80
    # unsupervisd flag to exclude all the rows associated with intensity lables
    if flag:
        A=A[num_intensity + 1:end(),:]
# OSVR/constructParams.m:83
        c=c[num_intensity + 1:end(),:]
# OSVR/constructParams.m:84
        weight=weight[num_intensity + 1:end(),:]
# OSVR/constructParams.m:85
        num_intensity=0
# OSVR/constructParams.m:86
    
    return A,c,D,num_intensity,num_pairs,weight
    
if __name__ == '__main__':
    pass
    