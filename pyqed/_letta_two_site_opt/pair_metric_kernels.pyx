# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Packed reduced metric contractions. C-generated BLAS execution, no truncation."""
from libc.stdint cimport int64_t
from scipy.linalg.cython_blas cimport dgemm, zgemm

ctypedef fused scalar:
    double
    double complex

cdef void multiply(char ta, char tb, int m, int n, int k, scalar alpha,
                   scalar *a, int lda, scalar *b, int ldb, scalar *c, int ldc) noexcept nogil:
    cdef scalar zero = 0
    if scalar is double:
        dgemm(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &zero, c, &ldc)
    else:
        zgemm(&ta, &tb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &zero, c, &ldc)


def contract(const int64_t[:, ::1] routes, const double[::1] scales,
             scalar[::1] left, scalar[::1] right, scalar[::1] vector,
             scalar[::1] result, scalar[::1] workspace):
    cdef Py_ssize_t i, j
    cdef int64_t lo, ro, xo, yo
    cdef int l, r, a, b, m
    cdef char normal = 78, transpose = 84
    cdef scalar one = 1, scale
    with nogil:
        for i in range(routes.shape[0]):
            lo, ro, xo, yo = routes[i,0], routes[i,1], routes[i,2], routes[i,3]
            l, r, a, b, m = routes[i,4], routes[i,5], routes[i,6], routes[i,7], routes[i,8]
            if l == 0 or r == 0 or a == 0 or b == 0 or m == 0:
                continue
            scale = scales[i]
            if routes[i,9]:
                multiply(normal,normal,m*r,a,l,one,&vector[xo],m*r,&left[lo],l,&workspace[0],m*r)
                multiply(transpose,normal,b,a*m,r,one,&right[ro],r,&workspace[0],r,&result[yo],b)
            else:
                multiply(transpose,normal,b,l*m,r,one,&right[ro],r,&vector[xo],r,&workspace[0],b)
                multiply(normal,normal,m*b,a,l,one,&workspace[0],m*b,&left[lo],l,&result[yo],m*b)
            for j in range(<int64_t>a*m*b):
                result[yo+j]*=scale


def sector_action(const int64_t[:, ::1] routes, scalar[::1] coefficients,
                  scalar[::1] left, scalar[::1] right, scalar[::1] vector,
                  scalar[::1] result, scalar[::1] workspace):
    """Accumulate a complete reduced operator schedule in one C call."""
    cdef Py_ssize_t i, j, k
    cdef int64_t lo, ro, xo, yo, offset, delta
    cdef int l, r, a, b, ip, op, np_, no
    cdef char normal = 78, transpose = 84
    cdef scalar one = 1
    with nogil:
        for i in range(routes.shape[0]):
            lo,ro,xo,yo=routes[i,0],routes[i,1],routes[i,2],routes[i,3]
            l,r,a,b=routes[i,4],routes[i,5],routes[i,6],routes[i,7]
            ip,op,np_,no=routes[i,8],routes[i,9],routes[i,10],routes[i,11]
            if l==0 or r==0 or a==0 or b==0:
                continue
            for j in range(l):
                for k in range(r):
                    workspace[j*r+k]=vector[xo+(j*np_+ip)*r+k]
            offset=<int64_t>l*r
            if (<double>a*l*r+<double>a*r*b <= <double>l*r*b+<double>a*l*b):
                delta=offset+<int64_t>a*r
                multiply(normal,normal,r,a,l,one,&workspace[0],r,&left[lo],l,&workspace[offset],r)
                multiply(transpose,normal,b,a,r,one,&right[ro],r,&workspace[offset],r,&workspace[delta],b)
            else:
                delta=offset+<int64_t>l*b
                multiply(transpose,normal,b,l,r,one,&right[ro],r,&workspace[0],r,&workspace[offset],b)
                multiply(normal,normal,b,a,l,one,&workspace[offset],b,&left[lo],l,&workspace[delta],b)
            for j in range(a):
                for k in range(b):
                    result[yo+(j*no+op)*b+k]+=coefficients[i]*workspace[delta+j*b+k]


def sector_action_reuse(const int64_t[:, ::1] routes, scalar[::1] coefficients,
                  scalar[::1] left, scalar[::1] right, scalar[::1] vector,
                  scalar[::1] result, scalar[::1] workspace,
                  const int64_t[:, ::1] reuse_plan, scalar[::1] reuse,
                  unsigned char[::1] ready):
    """Accumulate a complete reduced operator schedule in one C call."""
    cdef Py_ssize_t i, j, k
    cdef int64_t lo, ro, xo, yo, offset, delta, slot, reused_offset
    cdef scalar *middle
    cdef int l, r, a, b, ip, op, np_, no
    cdef char normal = 78, transpose = 84
    cdef scalar one = 1
    with nogil:
        for i in range(routes.shape[0]):
            lo,ro,xo,yo=routes[i,0],routes[i,1],routes[i,2],routes[i,3]
            l,r,a,b=routes[i,4],routes[i,5],routes[i,6],routes[i,7]
            ip,op,np_,no=routes[i,8],routes[i,9],routes[i,10],routes[i,11]
            if l==0 or r==0 or a==0 or b==0:
                continue
            slot,reused_offset=reuse_plan[i,0],reuse_plan[i,1]
            offset=<int64_t>l*r
            if slot<0 or not ready[slot]:
                for j in range(l):
                    for k in range(r):
                        workspace[j*r+k]=vector[xo+(j*np_+ip)*r+k]
                middle=&workspace[offset] if slot<0 else &reuse[reused_offset]
                if (<double>a*l*r+<double>a*r*b <= <double>l*r*b+<double>a*l*b):
                    multiply(normal,normal,r,a,l,one,&workspace[0],r,&left[lo],l,middle,r)
                else:
                    multiply(transpose,normal,b,l,r,one,&right[ro],r,&workspace[0],r,middle,b)
                if slot>=0:ready[slot]=1
            else:
                middle=&reuse[reused_offset]
            if (<double>a*l*r+<double>a*r*b <= <double>l*r*b+<double>a*l*b):
                delta=offset+<int64_t>a*r
                multiply(transpose,normal,b,a,r,one,&right[ro],r,middle,r,&workspace[delta],b)
            else:
                delta=offset+<int64_t>l*b
                multiply(normal,normal,b,a,l,one,middle,b,&left[lo],l,&workspace[delta],b)
            for j in range(a):
                for k in range(b):
                    result[yo+(j*no+op)*b+k]+=coefficients[i]*workspace[delta+j*b+k]


def overlap_matrix(const scalar[:, ::1] left, const scalar[:, ::1] right,
                   const int64_t[:, ::1] routes, double weight,
                   scalar[:, :, :, ::1] output):
    """Assemble the exact scattered tensor-product factor in route order."""
    cdef Py_ssize_t i, a, b
    cdef int64_t col, l, p, r
    cdef scalar scaled
    with nogil:
        for i in range(routes.shape[0]):
            col,l,p,r=routes[i,0],routes[i,1],routes[i,2],routes[i,3]
            for a in range(left.shape[0]):
                scaled=weight*left[a,l]
                if scaled == 0:
                    continue
                for b in range(right.shape[0]):
                    if right[b,r] != 0:
                        output[a,p,b,col]+=scaled*right[b,r]


def sector_first_products(const int64_t[:, ::1] routes, const int64_t[::1] offsets,
                          const int64_t[:, ::1] special,
                          scalar[::1] left, scalar[::1] right, scalar[::1] vector,
                          scalar[::1] products, scalar[::1] workspace):
    """Produce unique first GEMMs into disjoint shared slots, once per vector."""
    cdef Py_ssize_t i,j,k
    cdef int64_t lo,ro,xo,target
    cdef int l,r,a,b,ip,np_,kind,row,col
    cdef bint first_left
    cdef scalar factor
    cdef char normal=78, transpose=84
    cdef scalar one=1
    with nogil:
        for i in range(routes.shape[0]):
            lo,ro,xo=routes[i,0],routes[i,1],routes[i,2]
            l,r,a,b=routes[i,4],routes[i,5],routes[i,6],routes[i,7]
            ip,np_=routes[i,8],routes[i,10]
            if l==0 or r==0 or a==0 or b==0:continue
            target=offsets[i];kind=special[i,0];row=special[i,1];col=special[i,2]
            first_left=<double>a*l*r+<double>a*r*b<=<double>l*r*b+<double>a*l*b
            if kind==1:
                factor=left[lo] if first_left else right[ro]
                for j in range(l):
                    for k in range(r):products[target+j*r+k]=factor*vector[xo+(j*np_+ip)*r+k]
                continue
            if kind==2:
                if first_left:
                    for j in range(<int64_t>a*r):products[target+j]=0
                    factor=left[lo+<int64_t>row*l+col]
                    for k in range(r):products[target+<int64_t>row*r+k]=factor*vector[xo+(col*np_+ip)*r+k]
                else:
                    for j in range(<int64_t>l*b):products[target+j]=0
                    factor=right[ro+<int64_t>row*r+col]
                    for j in range(l):products[target+j*b+row]=factor*vector[xo+(j*np_+ip)*r+col]
                continue
            for j in range(l):
                for k in range(r):workspace[j*r+k]=vector[xo+(j*np_+ip)*r+k]
            target=offsets[i]
            if <double>a*l*r+<double>a*r*b<=<double>l*r*b+<double>a*l*b:
                multiply(normal,normal,r,a,l,one,&workspace[0],r,&left[lo],l,&products[target],r)
            else:
                multiply(transpose,normal,b,l,r,one,&right[ro],r,&workspace[0],r,&products[target],b)


def sector_shared_action(const int64_t[:, ::1] routes, scalar[::1] coefficients,
                         const int64_t[:, ::1] tiles, const int64_t[::1] offsets,
                         scalar[::1] left, scalar[::1] right, scalar[::1] vector,
                         scalar[::1] products, scalar[::1] result, scalar[::1] workspace):
    """Consume prepared products; each output row tile has exactly one owner."""
    cdef Py_ssize_t i,j,k
    cdef int64_t lo,ro,xo,yo,offset,delta,slot
    cdef int l,r,a,b,ip,op,np_,no,start,stop,rows
    cdef char normal=78, transpose=84
    cdef scalar one=1
    cdef scalar *middle
    cdef bint first_left
    with nogil:
        for i in range(routes.shape[0]):
            lo,ro,xo,yo=routes[i,0],routes[i,1],routes[i,2],routes[i,3]
            l,r,a,b=routes[i,4],routes[i,5],routes[i,6],routes[i,7]
            ip,op,np_,no=routes[i,8],routes[i,9],routes[i,10],routes[i,11]
            start,stop=tiles[i,0],tiles[i,1];rows=stop-start
            if l==0 or r==0 or rows==0 or b==0:continue
            first_left=<double>a*l*r+<double>a*r*b<=<double>l*r*b+<double>a*l*b
            offset=<int64_t>l*r;slot=offsets[i]
            if slot<0:
                for j in range(l):
                    for k in range(r):workspace[j*r+k]=vector[xo+(j*np_+ip)*r+k]
                middle=&workspace[offset]
                if first_left:
                    multiply(normal,normal,r,a,l,one,&workspace[0],r,&left[lo],l,middle,r)
                else:
                    multiply(transpose,normal,b,l,r,one,&right[ro],r,&workspace[0],r,middle,b)
            else:middle=&products[slot]
            delta=offset+(<int64_t>a*r if first_left else <int64_t>l*b)
            if first_left:
                multiply(transpose,normal,b,rows,r,one,&right[ro],r,middle+<int64_t>start*r,r,&workspace[delta],b)
            else:
                multiply(normal,normal,b,rows,l,one,middle,b,&left[lo+<int64_t>start*l],l,&workspace[delta],b)
            for j in range(rows):
                for k in range(b):
                    result[yo+((start+j)*no+op)*b+k]+=coefficients[i]*workspace[delta+j*b+k]
