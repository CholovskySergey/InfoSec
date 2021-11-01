def padding(msg, ext_length):
        int_len = len(msg)
        filler1 = [2 ** 7]
        filler2 = [0] * (115 - int_len)
        filler3 = []
        for i in range(0, 12, 1):
            n = (ext_length >> (8 * i)) % 256
            filler3.append(n)
        ext_msg = msg + filler1 + filler2 + filler3
        return ext_msg

msg = [1,2,3]
inp =[ 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,0x0E, 0x0F,
       0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
        0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E,
       0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F ]
# print(len(inp)*8)
inp = [int(byte) for byte in inp]
# print(len(inp)*8)
padded = padding(inp, len(inp)*8)
print(len(padded)*8)
# print([hex(byte) for byte in padded])

def build_state_matrix(inp):
    G = [[] for i in range(8)]
    for i in range(len(inp)):
        G[i % 8].append(inp[i])

    return G

G = build_state_matrix(padded)
print(len(G))
print(len(G[0]))
# for g in G:
#     print(g)

pi = [[168, 67, 95, 6, 107, 117, 108, 89, 113, 223, 135, 149, 23, 240, 216, 9,
                    109, 243, 29, 203, 201, 77, 44, 175, 121, 224, 151, 253, 111, 75, 69, 57,
                    62, 221, 163, 79, 180, 182, 154, 14, 31, 191, 21, 225, 73, 210, 147, 198,
                    146, 114, 158, 97, 209, 99, 250, 238, 244, 25, 213, 173, 88, 164, 187, 161,
                    220, 242, 131, 55, 66, 228, 122, 50, 156, 204, 171, 74, 143, 110, 4, 39,
                    46, 231, 226, 90, 150, 22, 35, 43, 194, 101, 102, 15, 188, 169, 71, 65,
                    52, 72, 252, 183, 106, 136, 165, 83, 134, 249, 91, 219, 56, 123, 195, 30,
                    34, 51, 36, 40, 54, 199, 178, 59, 142, 119, 186, 245, 20, 159, 8, 85,
                    155, 76, 254, 96, 92, 218, 24, 70, 205, 125, 33, 176, 63, 27, 137, 255,
                    235, 132, 105, 58, 157, 215, 211, 112, 103, 64, 181, 222, 93, 48, 145, 177,
                    120, 17, 1, 229, 0, 104, 152, 160, 197, 2, 166, 116, 45, 11, 162, 118,
                    179, 190, 206, 189, 174, 233, 138, 49, 28, 236, 241, 153, 148, 170, 246, 38,
                    47, 239, 232, 140, 53, 3, 212, 127, 251, 5, 193, 94, 144, 32, 61, 130,
                    247, 234, 10, 13, 126, 248, 80, 26, 196, 7, 87, 184, 60, 98, 227, 200,
                    172, 82, 100, 16, 208, 217, 19, 12, 18, 41, 81, 185, 207, 214, 115, 141,
                    129, 84, 192, 237, 78, 68, 167, 42, 133, 37, 230, 202, 124, 139, 86, 128],
                    [206, 187, 235, 146, 234, 203, 19, 193, 233, 58, 214, 178, 210, 144, 23, 248,
                     66, 21, 86, 180, 101, 28, 136, 67, 197, 92, 54, 186, 245, 87, 103, 141,
                    49, 246, 100, 88, 158, 244, 34, 170, 117, 15, 2, 177, 223, 109, 115, 77,
                    124, 38, 46, 247, 8, 93, 68, 62, 159, 20, 200, 174, 84, 16, 216, 188,
                    26, 107, 105, 243, 189, 51, 171, 250, 209, 155, 104, 78, 22, 149, 145, 238,
                    76, 99, 142, 91, 204, 60, 25, 161, 129, 73, 123, 217, 111, 55, 96, 202,
                    231, 43, 72, 253, 150, 69, 252, 65, 18, 13, 121, 229, 137, 140, 227, 32,
                    48, 220, 183, 108, 74, 181, 63, 151, 212, 98, 45, 6, 164, 165, 131, 95,
                    42, 218, 201, 0, 126, 162, 85, 191, 17, 213, 156, 207, 14, 10, 61, 81,
                    125, 147, 27, 254, 196, 71, 9, 134, 11, 143, 157, 106, 7, 185, 176, 152,
                    24, 50, 113, 75, 239, 59, 112, 160, 228, 64, 255, 195, 169, 230, 120, 249,
                    139, 70, 128, 30, 56, 225, 184, 168, 224, 12, 35, 118, 29, 37, 36, 5,
                    241, 110, 148, 40, 154, 132, 232, 163, 79, 119, 211, 133, 226, 82, 242, 130,
                    80, 122, 47, 116, 83, 179, 97, 175, 57, 53, 222, 205, 31, 153, 172, 173,
                    114, 44, 221, 208, 135, 190, 94, 166, 236, 4, 198, 3, 52, 251, 219, 89,
                    182, 194, 1, 240, 90, 237, 167, 102, 33, 127, 138, 39, 199, 192, 41, 215],
                    [147, 217, 154, 181, 152, 34, 69, 252, 186, 106, 223, 2, 159, 220, 81, 89,
                    74, 23, 43, 194, 148, 244, 187, 163, 98, 228, 113, 212, 205, 112, 22, 225,
                    73, 60, 192, 216, 92, 155, 173, 133, 83, 161, 122, 200, 45, 224, 209, 114,
                    166, 44, 196, 227, 118, 120, 183, 180, 9, 59, 14, 65, 76, 222, 178, 144,
                    37, 165, 215, 3, 17, 0, 195, 46, 146, 239, 78, 18, 157, 125, 203, 53,
                    16, 213, 79, 158, 77, 169, 85, 198, 208, 123, 24, 151, 211, 54, 230, 72,
                    86, 129, 143, 119, 204, 156, 185, 226, 172, 184, 47, 21, 164, 124, 218, 56,
                    30, 11, 5, 214, 20, 110, 108, 126, 102, 253, 177, 229, 96, 175, 94, 51,
                    135, 201, 240, 93, 109, 63, 136, 141, 199, 247, 29, 233, 236, 237, 128, 41,
                    39, 207, 153, 168, 80, 15, 55, 36, 40, 48, 149, 210, 62, 91, 64, 131,
                    179, 105, 87, 31, 7, 28, 138, 188, 32, 235, 206, 142, 171, 238, 49, 162,
                    115, 249, 202, 58, 26, 251, 13, 193, 254, 250, 242, 111, 189, 150, 221, 67,
                    82, 182, 8, 243, 174, 190, 25, 137, 50, 38, 176, 234, 75, 100, 132, 130,
                    107, 245, 121, 191, 1, 95, 117, 99, 27, 35, 61, 104, 42, 101, 232, 145,
                    246, 255, 19, 88, 241, 71, 10, 127, 197, 167, 231, 97, 90, 6, 70, 68,
                    66, 4, 160, 219, 57, 134, 84, 170, 140, 52, 33, 139, 248, 12, 116, 103],
                    [104, 141, 202, 77, 115, 75, 78, 42, 212, 82, 38, 179, 84, 30, 25, 31,
                    34, 3, 70, 61, 45, 74, 83, 131, 19, 138, 183, 213, 37, 121, 245, 189,
                    88, 47, 13, 2, 237, 81, 158, 17, 242, 62, 85, 94, 209, 22, 60, 102,
                    112, 93, 243, 69, 64, 204, 232, 148, 86, 8, 206, 26, 58, 210, 225, 223,
                    181, 56, 110, 14, 229, 244, 249, 134, 233, 79, 214, 133, 35, 207, 50, 153,
                    49, 20, 174, 238, 200, 72, 211, 48, 161, 146, 65, 177, 24, 196, 44, 113,
                    114, 68, 21, 253, 55, 190, 95, 170, 155, 136, 216, 171, 137, 156, 250, 96,
                    234, 188, 98, 12, 36, 166, 168, 236, 103, 32, 219, 124, 40, 221, 172, 91,
                    52, 126, 16, 241, 123, 143, 99, 160, 5, 154, 67, 119, 33, 191, 39, 9,
                    195, 159, 182, 215, 41, 194, 235, 192, 164, 139, 140, 29, 251, 255, 193, 178,
                    151, 46, 248, 101, 246, 117, 7, 4, 73, 51, 228, 217, 185, 208, 66, 199,
                    108, 144, 0, 142, 111, 80, 1, 197, 218, 71, 63, 205, 105, 162, 226, 122,
                    167, 198, 147, 15, 10, 6, 230, 43, 150, 163, 28, 175, 106, 18, 132, 57,
                    231, 176, 130, 247, 254, 157, 135, 92, 129, 53, 222, 180, 165, 252, 128, 239,
                    203, 187, 107, 118, 186, 90, 125, 120, 11, 149, 227, 173, 116, 152, 59, 54,
                    100, 109, 220, 240, 89, 169, 76, 23, 127, 145, 184, 201, 87, 27, 224, 97]]

v = [0x01, 0x01, 0x05, 0x01, 0x08, 0x06, 0x07, 0x04]

def T_l_xor(state):
    for round in range(10):
        state = k_nu_l(state, round)
        state = sbox(state)
        state = tau(state)
        state = psi(state)

    return state

def T_l_plus(state):
    for round in range(9):
        state = etha_nu_l(state, round)
        state = sbox(state)
        state = tau(state)
        state = psi(state)

    return state

def mult_field(x, y):
    p = 0
    while x:
        if (x & 1):
            p ^= y
        if y & 0x80:
            y = (y << 1) ^ 0x11D
        else:
            y <<= 1
            #             print(y)
        x >>= 1
    return p


def mul_by_02(num):
        #     base = 0x
    if num < 128:
        return num << 1
    else:
        return ((num << 1)) ^ 0x11D


def psi(state):
    res = []
    # for i in range(2):
    for i in range(8):
        r = []
        for j in range(len(state[0])):
            g = [state[i][j] for i in range(8)]
            r.append(scalar_mult(g, rightshiftvector(v, i)))
        res.append(r)
    return res


def scalar_mult( x, y):
        res = 0
        for i in range(len(x)):

#             res ^= self.multtable[x[i]][y[i]]
            res ^= mult_field(x[i],y[i])
        return res

def sbox(state):
    for i in range(8):
        for j in range(8):
            state[i][j] = pi[i%4][state[i][j]]
    return state


def modadd(x, y):
        return x + y % 2**64

def rightshiftvector(x, i):
    l = len(x)
    i = i % l
    return x[l-i:] + x[:l-i]

def tau(state):
    for i in range(len(state)-1):
        state[i] = rightshiftvector(state[i],i)
    if len(state[0]) == 16:
        state[len(state)-1] = rightshiftvector(state[len(state)-1], 11)
    else:
        state[len(state)-1] = rightshiftvector(state[len(state)-1], 7)

    return state
def k_nu_l(state, round):
    def generate_omega(j, v):
        omega = [((j%256) << 4) ^ v] + [0] * 7
        # print(j,omega)
        return omega
    for j in range(len(state[0])):
        omega = generate_omega(j,round)
        for i in range(8):
            # print(state[i][j], omega[i], state[i][j]^omega[i])
            state[i][j] ^= omega[i]
            # print(state[i][j])
    return state

def etha_nu_l(state, round, c=16):
    def generate_dzeta(j,c, round):
        # print(f'*{j, c-1-j,(c -1 - j) << 4}')
        dzeta =  [0xF3] + [0xF0] * 6  + [((c -1 - j) << 4)^(round)]
        return  dzeta

    def int2list(x):
        return [(x >> i) %256 for i in [56, 48, 40, 32, 24, 16, 8, 0]]

    #Represent list of bytes as 64-bit number
    def list2int(x):
        l = [56, 48, 40, 32, 24, 16, 8, 0]
        return sum([x[i] << l[i] for i in range(8)])

    for j in range(c):
        column = [state[i][j] for i in range(8)]
        dzeta = generate_dzeta(j,c, round)
        column = int2list((list2int(column)+list2int(dzeta)) % 2**64)
        for i in range(8):
            state[i][j] = column[i]
    return state
# 02   00    00 00 00 00 00 00 00 00 00

inp_string = '000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F'
inp_string = '2602D1E580D3126B9B293B34018A690EBF72F15BBDB2A1811A50B37E8F01D33D8AFDAA4350CFC93381F67A9B52273EC3AA3EC141D6E37358747938EF0A1C18EB'
inp = []
for i in range(0, len(inp_string), 2):
    s = int('0x'+inp_string[i:i+2], 16)
    inp.append(s)
state = build_state_matrix(inp)
print(len(state))
print(len(state[0]))

# out_string = []

# l = len(out_string)
# print(out_string[:l//2])
# print(out_string[l//2:])
test_string_final = '20A066016C8DAA5AA2ACA450D21F2796FBDC2E0CC452AF0AAF67E27A0755CB32718C2C7909201D3E7A3F256234C80B70D51AE3936DB26CF56E1F1BA8A0A7E1C0 '
test_string = test_string_final
test_string = '19F3C1D671C403DD8E1A2C25F27A5A70B263E24CAEA392D30D41A46F80F2C37F7DEE9B3441C0BA6574E76B8C43182FE59D2FB232C7D4646A676A29E0FB0C09ED'
test_string = [test_string[i:i+2] for i in range(0,len(test_string),2)]
# state = k_nu_l(state, 0)
# state = sbox(state)
# state = tau(state)
# state = psi(state)

# print(state)
# print('rez')
# state = T_l_xor(state)
state = etha_nu_l(state, round=1, c=8)
# print(len(state), len(state[0]))
# for i in range(8):
#     for j in range(8):
#         print(j,i,hex(state[j][i]),test_string[8*i+j])

def cupyna_hash(blocks, block_l):
    def init_hash(block_l):
        if block_l == 512:
            return [2**510]+[0] * 63
        if block_l == 1024:
            return [2**1023]+[0] * 128

    def to_list(state):
        n = []
        for i in range(8):
            for j in range(len(state[0])):
                n.append(state[i][j])
        return n

    def xor(L1,L2):
        L3 = [l1^l2 for l1,l2 in zip(L1,L2)]
        return L3
    ho = init_hash(block_l)
    H = [ho]
    for i in range(len(blocks)):
        m = build_state_matrix(blocks[i])
        n = build_state_matrix(xor(H[i-1], blocks[i]))
        q = xor(to_list(m),to_list(n))
        H.append(xor(q,H[i-1]))

    h_result = xor(to_list(T_l_xor(build_state_matrix(H[-1]))), H[-1])
    return h_result



# print(padded, len(padded))
msg = inp
padded = padding(msg, len(inp)*8)
print(len(padded))
cup = cupyna_hash([padded], 1024)
# print(len(cup*8))
print(cup)
import string
import random
import numpy as np
import time
N = 0
n_of_collisions = 0
collisions_tries =[]
start = time.time()
np.random.seed(10)
while n_of_collisions < 10:
    guess1 = np.random.randint(256, size=64).tolist()
    guess2 = np.random.randint(256, size=64).tolist()
    msg1 = [padding(guess1, 64 * 8)]
    msg2 = [padding(guess2, 64 * 8)]
    h1 = cupyna_hash(msg1, 512)
    h2 = cupyna_hash(msg2, 512)
    N += 1
    if N % 1000 == 0:
        print(N)
    # print(h1[0] % 256, h2[0] % 256)
    # break
    if h1[0] % 256 == h2[0] % 256:
        # print("Hurrah!")
        # print(N)
        n_of_collisions += 1
        collisions_tries.append(N)
        N = 0
        # print(guess1)
        # print(guess2)
        # print(h1 )
        # print(h2)
        # print(to_hash1)
        # print(to_hash2)

end = time.time()
print('___')
print(n_of_collisions)
print(end - start)
print(np.mean(collisions_tries))
print(np.std(collisions_tries)**0.5)