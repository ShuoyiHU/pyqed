from pyqed.dvr.dvr_1d import SineDVR
import numpy as np 
import scipy
from scipy.sparse.linalg import eigsh 

#import proplot as plt
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax.scipy.special import erf, erfc

pi = np.pi

class Gaussians:
    def __init__(self, alpha=1, x=0):
        self.alpha = alpha
        self.center = x
        return
    
class STO:
    def __init__(self, n, d, g):
        self.n = n      # the number of GTOs
        self.d = d      # coefficents
        self.g = g      # primitive GTOs
        return 
    
# class STO(STO):
#     pass

   
def sto_3g(center, zeta):

    scaling = zeta ** 2

    return STO(3, [0.444635, 0.535328, 0.154329],
               [Gaussians(scaling*0.109818, center),
                Gaussians(scaling*0.405771, center),
                Gaussians(scaling*2.22766, center)])

def sto_3g_hydrogen(center=0):

    return sto_3g(center, 1.24)

def overlap_1d(g1,g2):       #Gaussian wave package ~ (alpha/pi)*0.25 * exp(-0.5*alpha/(x-x0)**2)
    """
    overlap between two 1D GWPs
    """
    aj = g1.alpha
    ak = g2.alpha
    x = g1.center
    y = g2.center

    result = 0.0
    dq = y - x

    result = (aj*ak)**0.25 * np.sqrt(2./(aj+ak)) * np.exp(    \
            -aj*ak/(aj+ak) * (dq**2) )   
    return result       

def overlap_2d(gj, gk):
    """
    overlap between two GWPs defined by {a,x,p}
    """

    aj, qj = gj.alpha, gj.center
    ak, qk = gk.alpha, gk.center

    tmp = 1.0
    for d in range(2):
        #tmp *= overlap_1d(aj[d], qj[d], ak[d], qk[d])
        tmp *= overlap_1d(gj,gk)

    return tmp
        


#for zn, renormalize 2D sto-3g basis

def normalize(z):
    basis = sto_3g_hydrogen(0)
    a = [0.16885615680000002, 0.6239134896, 3.4252500160000006]
    sum = 0
    for i in range(basis.n):
        for j in range(basis.n):
            sum += np.exp(-a[i]*z**2)*np.exp(-a[j]*z**2)*basis.d[i]*basis.d[j]*overlap_2d(basis.g[i],basis.g[j])
    return np.sqrt(1./sum)
#print("normalize = ",normalize(0))   
'''
def sto_3g_2d(z):

    scaling = 1.24 ** 2
    N = normalize(z)

    return STO(3, [0.444635/z, 0.535328/z, 0.154329/z],
               [Gaussians(scaling*0.109818, 0),
                Gaussians(scaling*0.405771, 0),
                Gaussians(scaling*2.22766, 0)])
'''
def overlap_sto(b1,b2,z1,z2):
    sum = 0.
    N1 = normalize(z1)
    N2 = normalize(z2)
    a = [0.16885615680000002, 0.6239134896, 3.4252500160000006]
    for i in range(b1.n):
        for j in range(b2.n):
            sum += N1*N2*np.exp(-a[i]*z1**2)*np.exp(-a[j]*z2**2)*b1.d[i]*b2.d[j]*overlap_2d(b1.g[i],b2.g[j])
    return sum



def kin_1d(g1,g2):
    """
    kinetic energy matrix elements between two 1D GWPs
    """
    aj = g1.alpha
    ak = g2.alpha
    qj = g1.center
    qk = g2.center
    d0 = aj*ak/(aj+ak)  
    l = d0 * overlap_1d(g1,g2)     
    return l 

def kin_2d(gj, gk):
    """
    kinetic energy matrix elements between two multidimensional GWPs
    """

    aj, qj = gj.alpha, gj.center
    ak, qk = gk.alpha, gk.center

    ndim = 2
    
    # overlap for each dof 
    '''S = [overlap_1d(aj[d], qj[d], ak[d], qk[d]) \
         for d in range(ndim)]
        
    
    K = [kin_1d(aj[d], qj[d], ak[d], qk[d])\
         for d in range(ndim)]'''
    S = [overlap_1d(gj,gk) \
         for d in range(ndim)]
        
    
    K = [kin_1d(gj,gk)\
         for d in range(ndim)]

    res = 0
    for d in range(ndim):
        where = [True] * ndim
        where[d] = False
        res += K[d] * np.prod(S, where=where)

    return res

'''
def electron_nuclear_attraction(g1, g2, z):
    #i, q, r0 = g
    q = g1.alpha + g2.alpha
    b = z**2
    x = b * q



    return - jnp.sqrt(q/np.pi) * jnp.exp(x) * erfc(np.sqrt(x))
'''
def electron_nuclear_attraction(g1, g2, z):
    #i, q, r0 = g
    #q = g1.alpha + g2.alpha
    #b = z**2
    #x = b * q

    aj = g1.alpha
    ak = g2.alpha
    # print('xxxx', np.exp((aj+ak)*(z**2)) )
    # print('xxx', erfc(np.sqrt((aj+ak)*z**2)))
    
    x = np.sqrt((aj+ak)*(z**2))
    if abs(x) < 6:
        return  -2 * np.sqrt(aj*ak * np.pi/(aj+ak)) * np.exp(x**2) * erfc(x)
    else: 
        p = 0.47047
        a1 = 0.3480242
        a2 = -0.0958798
        a3 = 0.7478556
        t = 1/(1 + p * x)
        
        return  -2 * np.sqrt(aj*ak * np.pi/(aj+ak)) * (a1 * t + a2 * t**2 + a3 * t**3)
    
    #result = - jnp.sqrt(q/np.pi) * jnp.exp(x) * erfc(np.sqrt(x))
    # return result



def kin_sto(b1,b2,z1,z2):
    """
    
    Compute the kinetic energy oprator matrix elements between two GTOs
    
    .. math::
        
        K_{ij} = \langle g_i | -1\frac{1}{2} \nabla_x^2 + \nabla_y^2 |g_j\rangle 

    Parameters
    ----------
    b1 : TYPE
        DESCRIPTION.
    b2 : TYPE
        DESCRIPTION.
    z1 : TYPE
        DESCRIPTION.
    z2 : TYPE
        DESCRIPTION.

    Returns
    -------
    sum : TYPE
        DESCRIPTION.

    """
    sum = 0.
    N1 = normalize(z1)
    N2 = normalize(z2)
    
    #print(N1, N2)
    
    a = [0.16885615680000002, 0.6239134896, 3.4252500160000006]
    for i in range(b1.n):
        for j in range(b2.n):
            sum += N1*N2*np.exp(-a[i]*z1**2)*np.exp(-a[j]*z2**2)*b1.d[i]*b2.d[j]*kin_2d(b1.g[i],b2.g[j])
    return sum

def nuclear_attraction_sto(b1, b2, z):
    """
    

    Parameters
    ----------
    b1 : TYPE
        DESCRIPTION.
    b2 : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    V : TYPE
        DESCRIPTION.

    """
    V = 0. 
    N1 = normalize(z)
    # N2 = normalize(z2)
    a = [0.16885615680000002, 0.6239134896, 3.4252500160000006]

    for i in range(b1.n):
        for j in range(b2.n):

            tmp = electron_nuclear_attraction(b1.g[i], b2.g[j], z)
            # print('xx', i, j , tmp)
            
            V += N1**2 * np.exp(- (a[i] + a[j])* z**2) * b1.d[i] * b2.d[j] * tmp 
    return V





def energy(L, nz):

    #nz = 10
    #L =  2   # 2.57957
    xmin = -L
    xmax = L
    stepsize = (float(xmax)-float(xmin))/(float(nz)-1)
    z = np.array([xmin+stepsize*i for i in range (0, nz)])
    #print("z = ",z)
    #kinetic energy matrix XY

    T = np.zeros((nz, nz))
    for i in range (nz):
        b1 = sto_3g_hydrogen(0)
        b2 = sto_3g_hydrogen(0)
        T[i][i] = kin_sto(b1, b2, z[i], z[i])
    print("Txy",T)

    # attraction energy matrix 

    v = np.zeros(nz)
    b1 = sto_3g_hydrogen(0)
    b2 = sto_3g_hydrogen(0)
        
    for i in range(nz):

        v[i] = nuclear_attraction_sto(b1, b2, z[i])
    
    V = np.diag(v)
    
    print("V", v)

    # construct H'

    H_prime = T + V
    #print("H_prime", H_prime)

    dvr_z = SineDVR(npts=nz, xmin=-L, xmax=L) 
    # dvr_z = SincDVR(20, nz)
    z = dvr_z.x
    kz = dvr_z.t()
    #print("kz = ",kz)

    #overlap matrix

    S = np.zeros((nz,nz))
    basis = [0]*nz
    for i in range(nz):
        basis[i] = sto_3g_hydrogen(0)
    # construct S
    for i in range(nz):
        for j in range(nz):
            S[i][j] = overlap_sto(basis[i],basis[j],z[i],z[j])
    print("S = ",S)

    Tz = np.einsum('ij, ij -> ij', kz, S) # Kz * S
    #print("Tz = ",Tz)  
    H = Tz + H_prime
    print("H = ",H)


    E, U = eigsh(H, k=1, which='SA')
    #print("E = ", E)
    e = E[0]
    #return e, U
    return e


z = []
Energy = []
L = 6
nz = 65

E = energy(L, nz)
print(E)

# g1 = Gaussians()
# g2 = Gaussians()


# z = np.linspace(-5,5)

# e = np.zeros(len(z))
# for n in range(len(z)):
    
#     e[n] = electron_nuclear_attraction(g1, g2, z[n])
    

# fig, ax = plt.subplots()
# ax.plot(z, e, '-o')

    
    


# for nz in range(2,n):
#     Energy.append(energy(L, nz))
#     z.append(nz)

# #print("z = ",z)
# print("Energy = ", Energy)

# plt.plot(z, Energy, 'b.-', alpha=0.5, linewidth = 1, label = 'slice basis, L = {}'.format(L))

# plt.legend()
# plt.xlabel('nz')
# plt.ylabel("Energy(a.u.)")
# plt.ylim(-1,0.25)
# plt.savefig("save/slice_1.png")