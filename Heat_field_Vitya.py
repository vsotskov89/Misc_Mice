import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def Sort(hmap):
    k = len(hmap[0])-1
    while k>0:
        ind = 0
        for j in range(k+1):
            if hmap[0][j]>hmap[0][ind]:
                ind = j        
        for i in range(2):
            m = hmap[i][ind]
            hmap[i][ind] = hmap[i][k]
            hmap[i][k]=m
        k-=1
    return hmap

def DrawHeatField(mu,std):
    x = np.linspace(0,360,500)
    y= norm.pdf(x,mu,std/2)
    plt.plot(x,y)
    plt.show()
#DrawHeatField(90,20)
#def resize(bigmap,N):


def MakeMap(mu_list, std_list, size, sort):

    c = np.array([mu_list,std_list])

    if sort:
        c = Sort(c)
        
    nmu_list = c[0]
    nstd_list = c[1]
    
    x = np.linspace(0,360,size)
    bigmap = np.zeros((len(nmu_list),size))
    
    for ix,mu in enumerate(nmu_list):
        if np.isnan(mu):
            bigmap[ix] = np.zeros(size)
        else:
            y = norm.pdf(x,nmu_list[ix],nstd_list[ix]/2)
            bigmap[ix] = y/max(y)
    return bigmap


def MakeMultiMap(cell_list, size):
    x = np.linspace(0,360,size)
    bigmap = np.zeros((len(cell_list),size))
    for i, cell in enumerate(cell_list):
        bigmap[i] = np.zeros(size)
        for pf in cell:
            bigmap[i] += norm.pdf(x, pf.mu, pf.std/2)
        bigmap[i] = bigmap[i]/max(bigmap[i])
    return bigmap
        
        
    
#mu_list = [225.0, 40.0, 45.0, 279.0, 144.0, 117.0, 234.0, 117.0, 284.0, 261.0, 297.0, 274.0, 148.0, 306.0, 207.0, 360.0, 126.0, 243.0, 284.0, 32.0, 45.0, 297.0, 238.0, 27.0, 32.0, 342.0]
#std_list = [54.0, 63.0, 54.0, 54.0, 18.0, 36.0, 36.0, 90.0, 45.0, 54.0, 18.0, 45.0, 27.0, 90.0, 36.0, 36.0, 18.0, 54.0, 63.0, 27.0, 36.0, 54.0, 45.0, 54.0, 45.0, 54.0]
#mu_list = [150,90,50]
#std_list = [30,60,10]


def DrawAllfields(mu_list, std_list, outfname, sort = True):
        
    fig = plt.figure(figsize=(10,10), dpi=300)
    ax = fig.add_subplot(111) 
    size = 360
    bigmap = MakeMap(mu_list, std_list, size, sort)
#    print(bigmap[0])
    ax.imshow(bigmap, cmap='jet',interpolation = 'nearest')
#    ax.colorbar()
    plt.savefig(outfname) 
    plt.close()
    
    
def DrawMultipleFields(cell_list, outfname):
    fig = plt.figure(figsize=(10,10), dpi=300)
    ax = fig.add_subplot(111) 
    size = 360
    bigmap = MakeMultiMap(cell_list, size)
    ax.imshow(bigmap, cmap='jet',interpolation = 'nearest')
    plt.savefig(outfname) 
    plt.close()