

import struct
import numpy as np


def binary(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode() # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value # Instance of bytes



def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode() # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value # Instance of str


str1 = to_bytes(4.5)
#x = np.frombuffer( str1, np.float32 )
#print('x',x)
print('str1', str1)
#print(oct(2.3))

#def float_to_bin(num):
 #   return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)

# returns binary string
def float_to_bin(num):
  return str(format(struct.unpack('!I', struct.pack('!f', num))[0], '032b'))

# returns float
def bin_to_float(binary):
  return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]

#def bin_to_float(binary):
 #   return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]

y = float_to_bin(123.123)
z = bin_to_float(y)
print('float 2 bin',y)
print('bin 2 float',z)

y = float_to_bin(-123.123)
z = bin_to_float(y)
print('float 2 bin',y)
print('bin 2 float',z)
#1100 0010 1111 0110 0011 1110 1111 1010
#0100 0010 0000 0100 0111 1101 1111 0100
#float_to_bin(bin_to_float(float_to_bin(123.123))) == float_to_bin(123.123)
#1100 0010 1111 0110 0011 1110 1111 1010

#exit()

#print (binary(1))
int32bits = np.float32(-0.30555432).view(np.int32).item()    


int32bits2 = np.float32(3.03).view(np.uint32).item() 
print('{:032b}'.format(int32bits))
print('{:016b}'.format(int32bits2))
print('{:032b}'.format(int32bits2))
print('{:08b}'.format(int32bits2))

int16bits = np.float16(3.03).view(np.int16).item() 
print('int16','{:016b}'.format(int16bits))

int16bits = np.float16(20000.03).view(np.int16).item() 
print('int16','{:016b}'.format(int16bits))
print(type(int16bits))
int16bits = np.float16(-30000.0).view(np.int16).item() 
print('int16','{:016b}'.format(int16bits))
print(type(int16bits))
int16bits = np.float16(30000.0).view(np.int16).item() 
print('int16','{:016b}'.format(int16bits))
print(type(int16bits))

int32bits = np.float32(-9.0).view(np.int32).item() 
print('int32','{:08b}'.format(int32bits))
print(type(int32bits))
str1 = '{:32b}'.format(int32bits)
#str2 = '{:32f}'.format(str1)
#print('str1',str1, str2)
#folat16bits = np.int16(-3.03).view(np.int16).item() 
#print('int16','{:016b}'.format(int16bits))
#print(float(int16bits))
#1000 0010 0010 0000 0000 0000 0000 000
#-111 1101 1110 0000 0000 0000 0000 000
hexit = ''
for s in str1:
    #s = str1[i]
    if s == '1':
        hexit = hexit+"\x01"
    else:
        hexit = hexit+"\x00"
        
print('hexit',hexit)        
#np.fromstring( "\x00\x00\x00\x00\x00\x00\x00\x00", np.float32 )
x = np.frombuffer( hexit.encode(), np.float32 )
print('x',x)







