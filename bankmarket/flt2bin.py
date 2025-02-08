
#https://stackoverflow.com/questions/8751653/how-to-convert-a-binary-string-into-a-float-value

from codecs import decode
import struct


def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]


def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.
        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    print('n',n)
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]


def float_to_bin(value):  # For testing.
    """ Convert float to 64-bit binary string. """
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return '{:064b}'.format(d)

#for f in 0.0, 1.0, -14.0, 12.546, 3.141593:
for f in 0, -4.567653, -40.0, 1.0, -14.0, 12.546, 3.141593:
        print('Test value: %f' % f)
        binary = float_to_bin(f)
        print(' float_to_bin: %r' % binary)
        floating_point = bin_to_float(binary)  # Round trip.
        print(' bin_to_float: %f\n' % floating_point)

#1100 0000 0001 0010 0100 0101 0100 0110 1101 0011 1111 1001 1110 0111 1011 1000
        
#1000 1110 0101 1110 0000 0100 0100 0100 0001 1101 1101 0110 1011 0100 1000 01000
        
sf='0100000000001001001000011111101110000010110000101011110101111111'
floating_point = bin_to_float(sf)  # Round trip.
print(' bin_to_float: %f\n' % floating_point)
