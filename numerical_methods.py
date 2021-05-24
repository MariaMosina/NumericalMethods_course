import numpy as np
import pandas as pd

def sqrt(x, eps, delta=1):
    p = (1+x)/2 
    p1 = (p+x/p)/2
    while abs(p-p1) > eps/delta:
        p = p1
        p1 = (p+x/p)/2
    return p1

def factorial(x):
    f=1
    for i in range(1, x+1): f*=i
    return f

def ch(x, eps):
    rem = 1
    k=1
    res = rem
    while rem > eps/3:
        rem = x**(2*k)/factorial(2*k)
        res += rem
        k+=1
    return res

def cos(x, eps):
    if abs(x) < m.pi/2:
        rem = 1
        k=1
        res = rem
        while abs(rem) > eps/3:
            rem = (-1)**k * x**(2*k)/factorial(2*k)
            res += rem
            k+=1
        return res
    else:
        print("Аргумент х выходит за пределы необходимого диапазона")
        

def determinant(matrix, mul):
    width = len(matrix)
    if width == 1:
        return mul * matrix[0][0]
    else:
        sign = -1
        s = 0
        for i in range(width):
            m = []
            for j in range(1, width):
                buff = []
                for k in range(width):
                    if k != i:
                        buff.append(matrix[j][k])
                m.append(buff)
            sign *= -1
            s += mul * determinant(m, sign * matrix[0][i])
    return s


# In[6]:


class Array:
    
    def __init__(self, a):
        self.array = np.array(a, dtype = 'float')
        self.shape = self.shape_f()

    def shape_f(self):
        return self.array.shape
    
    def T(self):
        res = np.zeros((self.shape[1], self.shape[0]))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[j][i] = self.array[i][j]
        return Array(res)
        

    def plus(self, other):
        res = np.array(self.array)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[i][j] += other.array[i][j]
        return Array(res)

    def pointwise_multiply(self, other):
        res = np.array(self.array)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                res[i][j] *= other.array[i][j]
        return Array(res)
    

    def matrix_multiply(self, other):
        res = np.zeros((self.shape[0], other.shape[1]))
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                for k in range(len(self.array[i])):
                    res[i][j] += self.array[i][k] * other.array[k][j]
        return Array(res)
    def det(self):
        return determinant(self.array, 1)
        
    
    # Operators
    def __add__(self, other):
        return self.plus(other)
    
    def __sub__(self, other):
        return self.plus(Array(-(other.array)))
    
    def __mul__(self, other):
        return self.pointwise_multiply(other)
    
    def __matmul__(self, other):
        return self.matrix_multiply(other)
    
    def __getitem__(self, ij):
        (i, j) = ij
        return self.array[i][j]


# In[7]:


def v_norm_inf(v):
    return max(v.array)
def v_norm_1(v):
    return sum(abs(v.array))
def v_norm_2(v):
    return np.sqrt((v.T() @ v).array[0][0])

def m_norm_inf(A):
    return max(sum(abs(A.array)))
def m_norm_1(A):
    return m_norm_inf(A.T())
def m_norm_2(A):
    M = A.T() @ A
    return sqrt(max(np.linalg.eigh(M.array)[0]))
def m_norm_f(A):
    return sum(sum((A*A).array))


# In[8]:


def simple_iteration(A, b, eps):
    if not A.det() :
        print("Матрица А особая")
        return 0
    v_n_norm = 0
    b = b.T()
    n = A.shape[0]
    mu = 1/m_norms[m_n_norm](A)
    Mu = Array(np.ones(A.shape) * mu)
    B = Array(np.eye(n)) - Mu*A
    flag = 0
    if m_norms[0](B)>1 and m_norms[1](B)>1:
        b = A.T() @ b
        A = A.T() @ A
        mu = 1/m_norms[m_n_norm](A) 
        Mu = Array(np.ones(A.shape) * mu)
        B = Array(np.eye(n)) - Mu*A
        if m_norms[0](B)>1 and m_norms[1](B)>1:
            flag = 1
            if m_norms[0](B)>1: v_n_norm = 1
            else: v_n_norm = 0
        
    Mu = Array(np.ones(b.shape) * mu)
    c = Mu * b
    x_old = c
    x = B @ x_old + c
    
    B_norm = m_norms[m_n_norm](B)/(1-m_norms[m_n_norm](B))
    iterations = 1
    if flag:
        while v_norms[v_n_norm](A @ x - b) > eps:
            x_old = x
            x = B @ x_old + c
            iterations +=1
    else:
        while B_norm * v_norms[v_n_norm](x-x_old) > eps:
            x_old = x
            x = B @ x_old + c
            iterations +=1
    return x, iterations


# In[9]:


def seidel(A, b, eps):
    if not A.det() :
        print("Матрица А особая")
        return 0
    n = A.shape[0]
    b = b.T()
    
    M = [abs(A[i, i]) > (sum(abs(A.array[i]))-abs(A[i, i])) for i in range(n)]
    if sum(M) != n:
        b = A.T() @ b
        A = A.T() @ A
    C = Array(A.array)
    C = Array([C.array[i]*(-1/A[i, i]) for i in range(n)]) + Array(np.eye(n))
    d = Array([[b[i, 0]/A[i, i] for i in range(n)]]).T()
    x_old = Array(d.array)
    x = Array(x_old.array)
    for i in range(n):
        x_old = Array(x.array)
        x.array[i] = (Array([C.array[i]]) @ x_old)[0, 0] + d[i, 0]
    iterations = 1
    while v_norms[v_n_norm](A @ x - b) > eps:
        for i in range(n):
            x_old = Array(x.array)
            x.array[i] = (Array([C.array[i]]) @ x_old)[0, 0] + d[i, 0]
        iterations += 1
    return x, iterations


# In[10]:


def gauss(A, b):
    if not A.det() :
        print("Матрица А особая")
        return 0
    A = Array(A.array)
    n = A.shape[0]
    b = b.T()
    for i in range(n):
        T = A.T()
        m = np.argmax(abs(T.array[i][i:]))
        A.array[i], A.array[i+m] = np.array(A.array[i+m]), np.array(A.array[i])
        b.array[i], b.array[i+m] = np.array(b.array[i+m]), np.array(b.array[i])
        a = A[i, i]
        A.array[i] = A.array[i]*(1/a)
        b.array[i] = b.array[i]*(1/a)
        for j in range(i+1, n):
            b.array[j] = np.array(b.array[j] - A[j, i] * b.array[i])
            A.array[j] = np.array(A.array[j] - A[j, i] * A.array[i])
    x = Array(np.zeros(b.shape))
    x.array[n-1] = b.array[n-1]
    for i in range(n-1):
        tmp = Array(-x.array)
        x.array[n-2-i] = (Array([A.array[n-2-i]]) @ tmp)[0,0] + b.array[n-2-i]
    return x


# In[11]:


def householder(A, b):
    if not A.det() :
        print("Матрица А особая")
        return 0
    A = Array(A.array)
    n = A.shape[0]
    b = b.T()
    
    Q = Array(np.eye(n))
    Ri = Array(A.array)
    R = [Ri]
    
    for i in range(n-1):
        Q_old = Array(np.eye(n))
        R_old = Array(Ri.array[i:, i:])
        zi = Array(np.zeros((n-i, 1)))
        zi.array[0][0] = 1
        yi = Array([R_old.array[:, 0]]).T()
        alpha = v_norm_2(yi)
        alp = Array([np.ones(zi.shape) * alpha]).T()
        ro = v_norm_2(yi-alp*zi)
        wi = yi - alp*zi
        wi = Array([np.ones(wi.shape)* (1/ro)]).T() * wi
        Qi = Array(np.eye(n-i)) - wi @ wi.T() - wi @ wi.T()
        Q_old.array[i:, i:] = Qi.array
        Ri.array[i:, i:] = (Qi @ R_old).array
        Q = Q_old @ Q
        R.append(R[i])
    y = Q @ b
    x = gauss(R[-1], y.T())
    
    return x

