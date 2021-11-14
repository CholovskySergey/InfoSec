import random
import numpy as np

import sys
sys.setrecursionlimit(5000) #for generalized euclidean alg


def factorization(num):
    ec = num -1
    if ec % 2 != 0:
        return -1, -1
    k = 0
    while ec % 2 == 0:
        ec >>= 1
        k += 1
    k -= 1
    d = (num - 1) // ( 2 ** k)
    # print(d, k)
    assert (num - 1) == d * (2**k)
    return d, k

print(factorization(60))


def miller_rabin_test(n, r):
    if n == 2 or n == 3:
        return True
    m, s = factorization(n)
    if m == -1:
        return False
    # print(f'factorization{m, s}')
    for j in range(r):
        a = random.randrange(2, n-2)
        # print(f'round {j} number {a}')

        b = pow(a, m, n)
        # print(b)
        if b != 1 and b != (n - 1):
            i = 1
            while i < s and b != n-1 :
                b = (b ** 2 ) % n
                # print(b)
                if b == 1:
                    return False
                i += 1
            if b != n - 1:
                return False

    return True

# k = 0
# for i in range(1000):
#     k += miller_rabin_test(13, 10)
# print(k / 1000)
# print(miller_rabin_test(19, 10))
# print(random.randrange(2,12))

def generate_prime(max_n, r):
    p = random.randrange(5, max_n)
    i = 1
    # print(f'max guess {max_n}')
    while not miller_rabin_test(p, r):
        i += 1
        p = random.randrange(5, max_n)

    return p, i

def generate_prime_quick(max_n, r):
    p = random.randrange(5, max_n, step=2)
    N = min(max_n, 10000)
    i = 1
    # print(f'max guess {max_n}')
    primitives_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                       31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                        101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
                       163, 167, 173, 179, 181, 191, 193, 197, 199,    211,    223,
                       227,    229,
                        233,    239,    241,    251,    257,    263,    269,    271,    277,    281,
                        283,    293,    307,    311,    313 ,   317,    331,    337,    347,    349,
                        353,    359,    367,    373,    379,    383,    389,    397,    401,    409,
                        419,    421,    431,    433,    439,    443,    449,    457,    461,    463,
                        467,    479,    487,    491,    499,    503,    509,    521,    523,    541,
                        547,    557,    563,    569,    571,    577,    587,    593,    599,    601,
                        607,    613,    617,    619,    631,    641,    643,    647,    653,    659,
                        661,   673,    677,    683,    691,    701,    709,    719,    727,    733,
                        739,   743,    751,    757,    761,    769,    773,    787,    797,    809,
                        811,    821,    823,    827,    829,    839,    853,    857,    859,    863,
                        877,    881,    883,    887,    907,   911,    919,    929,    937,    941,
                        947,    953,    967,    971,    977,    983,    991,    997,   1009,   1013 ]
    while True:
        if miller_rabin_test(p, r):
            break
        i += 1
        guess = None
        delta = 2
        flag = True
        while flag:
            p = random.randrange(5, max_n, step=2)
            flag = False
            for g in primitives_list:
                if p % g == 0:
                    flag = True
                    break

    # print(f'tries before guess:{i}')
    return p, i

# print(generate_prime(10000,10))
print(miller_rabin_test(9,4))

def eucledean_alg(a, b):
    r = []
    arr = [a,b]
    while r!=0 :
        r = arr[-2] % arr[-1]
        arr.append(r)
    return arr[-2]

# print(eucledean_alg(2,3))
# print(eucledean_alg(11,3))
# print(eucledean_alg(4,6))
# print(eucledean_alg(40,6))
# print(eucledean_alg(40,30))
# print(eucledean_alg(30,40))

def generate_key(max_n=100, r = 10, prime_generator=generate_prime):
    p, _ = prime_generator(max_n, r)
    q, _ = prime_generator(max_n, r)
    print(f'primes: {p,q}')
    N = p * q
    fi = (p - 1) * (q - 1)
    e = random.randrange(2, fi)
    while eucledean_alg(e, fi) != 1:
        e = random.randrange(2, fi)

    d = extended_euclid(e, fi)[1] % fi
    return N,e, N,d, fi, p,q

def extended_euclid(a, b):
        if (b == 0):
            return a, 1, 0
        d, x, y = extended_euclid(b, a % b)
        return d, y, x - (a // b) * y

# def calculate_reversed(e, fi):

# print()
# N,e, N,d, fi, p,q = generate_key()
# print((e*d) % fi)
# print(eucledean_alg(e, fi))

def encrypt(m,N, e):
    # return (m ** e) % N
    return pow(m,e,N)

def chineese_decrypt(m, p, q, e):
    m_p = pow(m,e,p)
    m_q = pow(m,e,q)
    M = p * q
    M1 = q
    M2 = p
    M1_r = extended_euclid(M1, p)[1]
    M2_r = extended_euclid(M2, q)[1]
    # print((M1_r + M1) % p == 0)
    return (m_p*M1_r*M1 + m_q*M2*M2_r) % N

# m = 10
# print(f'modulus {N} public {e} secret {d}')
# print(eucledean_alg(m,N))
# c = encrypt(m, N,e)
# q = encrypt(c, N,d)
# Q = encrypt(q, N,e)
# print(eucledean_alg(c,N))
# print(m,c,q, Q)

def padding(x,k):
    # print(x,k,-11)
    return x + [1]*(k -len(x))

def truncate(x,k):
    return x[:k]

def xor(X,Y):
    return [x^y for x,y in zip(X,Y)]

def OAEP(n, k0, k1, G, H):
    def enc(m, r, k0, k1, G, H):
        p = m + [0] * k1
        X = xor(p, G(r))
        Y = xor(r, H(X))
        # print('X')
        # print(X)
        # print('Y')
        # print(Y)
        return X + Y

    def dec(arr, n, k0, k1, G, H):
        X = arr[:n-k0]
        # print('DEC')
        # print(arr)
        # print('X')
        # print(X)
        Y = arr[-k0:]
        # print('Y')
        # print(Y)
        r = xor(Y, H(X))
        E = xor(X, G(r))
        return E[:n-k1 -k0]

    OAEP_enc = lambda m,r: enc(m, r, k0, k1, G, H)
    OAEP_dec = lambda arr: dec(arr,n, k0, k1, G, H)
    return OAEP_enc, OAEP_dec

#
n = 10
k0 = 2
k1 = 2

m = [1,2,3,4,5,6]
r =[11, -5]
G = lambda x: padding(x, n-k1)
H = lambda x: truncate(x, k0) #

enc, dec = OAEP(n,k0,k1,G,H)

CR = enc(m, r)

DR = dec(CR)


def arr_to_int(arr):
    x = 0
    for i, num in enumerate(arr):
        x += (num % 256) << (8 * i)
    return x

def int_to_arr(num, n_bytes):
    i = 0
    arr = []
    while i < n_bytes:
        arr.append((num>>8*i)%256)
        i += 1
    return arr


# very syntetic example of OAEP using
def OAEP_RSA_encryption(n, k0, k1, N, e, key_len, enc):
    msg = [6] * (n-k0-k1)
    r = [3] * k1
    CR = enc(msg, r)
    num = arr_to_int(CR)
    rse = encrypt(num, N, e)
    return rse

def OAEP_RSA_decryption(msg, n, k0, k1, N, d, key_len, dec):
    rsd = encrypt(msg, N, d)
    to_dec = int_to_arr(rsd, n)
    DR = dec(to_dec)
    return DR

import time
import tqdm
times = []
times_per_byte = []
key_lenths = []
generation_times = []
encryption_times = []
decryption_times = []
chineese_decryption_times = []
OAEP_enc_dec_times = []

print('experiment')
for i in (range(128, 1024,8)):
    print(i)
    key_lenths.append(i)
    m = 2**i
    msg = [1] * 8
    num = arr_to_int(msg) % m



    # num = 2*(m+1) - 1
    start = time.time()
    N, e, N, d, fi, p,q = generate_key(max_n=m, r = 30, prime_generator=generate_prime)
    end = time.time()
    dt = end - start
    generation_times.append(dt)

    start = time.time()
    for j in range(1000):
        c = encrypt(num, N, e)
    end = time.time()
    dt = end - start
    encryption_times.append(dt)

    start = time.time()
    for j in range(1000):
        Q = encrypt(c, N, d)
    end = time.time()
    dt = end - start
    decryption_times.append(dt)

    start = time.time()
    for j in range(1000):
        Q = chineese_decrypt(c, p,q, d)
    end = time.time()
    dt = end - start
    chineese_decryption_times.append(dt)

    if Q != num:
        print('error')
        print(p*q == N)
    end = time.time()


    key_len = i
    n = 8
    k0 = 2
    k1 = 2
    G = lambda x: padding(x, n - k1)
    H = lambda x: truncate(x, k0)
    oaep_enc, oaep_dec = OAEP(n, k0, k1, G, H)
    start = time.time()
    for j in range(1000):
        crypted = OAEP_RSA_encryption(n, k0, k1, N, e, key_len, oaep_enc)
        decrypted = OAEP_RSA_decryption(crypted, n, k0, k1, N, d, key_len, oaep_dec)
    # print(decrypted)
    # print(len(decrypted), n)
    end = time.time()
    dt = end - start
    OAEP_enc_dec_times.append(dt)



with open('results4.txt','w', encoding='utf-8') as f:
    for i,gen_time,enc_time,dec_time,cdec_time, oaep_time in zip(
                                                                key_lenths, generation_times,
                                                                encryption_times, decryption_times,
                                                                chineese_decryption_times, OAEP_enc_dec_times):
        f.write('-----------------------------------------\n')
        f.write(f'Довжина простого числа: {i} біт\n')
        f.write(f'Довжина повідомлення : {1000*i} біт\n')
        f.write(f' витрачено часу на генерацію ключа {gen_time} с\n')
        f.write(f' витрачено часу на шифрування {enc_time} с\n')
        f.write(f' витрачено часу на дешифрування звичайним методом {dec_time} с\n')
        f.write(f' витрачено часу на дешифрування за допомогою КТЛ {cdec_time} с\n')
        f.write(f'корисна довжина OAEP-повідомлення : {1000*(i - 4)} біт\n')
        f.write(f' витрачено часу на розшифрування та дешифрування за допомогою OAEP  {oaep_time} с\n')
