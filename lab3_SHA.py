def padding(msg, ext_length):
    int_len = len(msg)
    filler1 = [2**7]
    filler2 = [0] * (55 - int_len)
    filler3 = []
    for i in range(7, -1, -1):
        n = (ext_length >> (8 * i)) % 256
        filler3.append(n)
    ext_msg = msg + filler1 + filler2 + filler3
    return ext_msg


def split_msg(msg):
    block_len = 512
    blocks = [split_block(msg[i:i + block_len]) for i in range(0, len(msg), block_len // 8)]
    return blocks

import sys

def split_block(block):
    M = [0,0,0,0] * 4
    for i in range(16):
        # print( block[4*i], block[4*i + 1], block[4*i + 2], block[4*i + 3])
        # print(block[4*i:4*i + 4])
        M[i] = ((block[4*i] %256)<< 24) +( (block[4*i + 1] %256) << 16) +( (block[4*i+2] %256) << 8) +(block[4*i+3] %256)
        # print(f'len M {sys.getsizeof(M[i])}')
        # print((block[4*i + 3]))
        # print(M[i])
    return M

H0 = [0x6a09e667,
      0xbb67ae85,
      0x3c6ef372,
      0xa54ff53a,
      0x510e527f,
      0x9b05688c,
      0x1f83d9ab,
      0x5be0cd19
      ]

K_256 = [ 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
          0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
          0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
          0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
          0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
          0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
          0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
          0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 ]

BASE = 2**32

def Ch(x,y,z):
    return (x & y) ^ (~x & z)

def MAJ(x,y,z):
    return (x & y) ^ (x & z) ^ (y & z)

def SIGMA_256(x):
    return ROTR(x,2) ^ ROTR(x,13) ^ ROTR(x,22)

def SIGMA1_256(x):
    return ROTR(x,6) ^ ROTR(x,11) ^ ROTR(x,25)

def sigma_256(x):
    return ROTR(x,7) ^ ROTR(x,18) ^ SHR(x,3)

def sigma1_256(x):
    return ROTR(x,17) ^ ROTR(x,19) ^ SHR(x,10)

w = 32
def ROTR(x, n):
    return (x >> n) | (x << (w -n))

def SHR(x,n):
    return  x >> n

def prepare_schedule(block):
    W = [0 for i in range(64)]
    # print(len(W))
    for i in range(16):
        W[i] = block[i]
    for i in range(16,64):
        W[i] = (sigma1_256(W[i-2]) + W[i-7] + sigma_256(W[i-15]) + W[i-16] ) % BASE
    # print("-_-")
    # print(W)
    # print('____')
    return W


def compute_hash(blocks):
    H = [0x6a09e667,
      0xbb67ae85,
      0x3c6ef372,
      0xa54ff53a,
      0x510e527f,
      0x9b05688c,
      0x1f83d9ab,
      0x5be0cd19
      ]

    N = len(blocks)
    # print(N)
    # print('*')
    for i in range(N):
        W = prepare_schedule(blocks[i])
        a = H[0]
        b = H[1]
        c = H[2]
        d = H[3]
        e = H[4]
        f = H[5]
        g = H[6]
        h = H[7]

        for t in range(64):
            T1 = (h + SIGMA1_256(e) + Ch(e,f,g) + W[t] + K_256[t]) % BASE
            # print(W[t] % BASE)
            T2 = SIGMA_256(a) + MAJ(a,b,c)
            # print(f'{t}T1{T1}, T2{T2}')
            # print(f'{t} {h},  {SIGMA1_256(e)}, {Ch(e,f,g)}, {W[t] %10}')
            # print(f'sum {(W[t] ) % BASE} ')
            h,g,f = g,f,e
            e = (d + T1) % BASE
            d,c,b = c,b,a
            a = (T1 + T2) % BASE
            # print(a,b,c,d,e,f,g,h)

        H[0] = (a + H[0]) % BASE
        H[1] = (b + H[1]) % BASE
        H[2] = (c + H[2]) % BASE
        H[3] = (d + H[3]) % BASE
        H[4] = (e + H[4]) % BASE
        H[5] = (f + H[5]) % BASE
        H[6] = (g + H[6]) % BASE
        H[7] = (h + H[7]) % BASE

    return H


#
import numpy as np
# default = bytearray(b'abc')

# default = [1,1,1,3,1,1]
# # print(default)
#
# ext = padding(default, 4)
#
# prepared_msg = split_msg(ext)
#
# hash = compute_hash(prepared_msg)
# hash1 = [600636658, 3102750418, 2792937945, 1622647743, 755610774, 894257966, 1888627754, 4142282149]
# print(hash)
# print(hash == hash1)

import string
import random
import time
l1 = 12
l2 = 14
N=0

n_of_collisions = 0
collisions_tries =[]
start = time.time()
np.random.seed(10)

while n_of_collisions < 10:
    guess1 = np.random.randint(256, size=l1).tolist()
    guess2 = np.random.randint(256, size=l2).tolist()
    msg1 = padding(guess1, l1*8)
    # print(len(msg1))
    to_hash1 = split_msg(msg1)
    # to_hash1 = split_msg(guess1)
    h1 = compute_hash(to_hash1)
    msg2 = padding(guess2, l2*8)
    to_hash2 = split_msg(msg2)
    # to_hash2 = split_msg(guess2)
    h2 = compute_hash(to_hash2)
    N += 1
    if N % 1000 == 0:
        print(N)
    # print(h1[0] % 256, h2[0] % 256)
    # break
    if h1[0] % 256 == h2[0] % 256:
        n_of_collisions += 1
        collisions_tries.append(N)
        N = 0
        # print("Hurrah!")
        # print(N)
        # # print(guess1)
        # # print(guess2)
        # print(h1 )
        # print(h2)
        # print(H0)
        # print(to_hash1)
        # print(to_hash2)

end = time.time()
print('___')
print(n_of_collisions)
print(end - start)
print(np.mean(collisions_tries))
print(np.std(collisions_tries)**0.5)
print(collisions_tries)
