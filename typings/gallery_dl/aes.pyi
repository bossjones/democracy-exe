"""
This type stub file was generated by pyright.
"""

from Cryptodome.Cipher import AES as Cryptodome_AES

if Cryptodome_AES:
    def aes_cbc_decrypt_bytes(data, key, iv):
        """Decrypt bytes with AES-CBC using pycryptodome"""
        ...
    
    def aes_gcm_decrypt_and_verify_bytes(data, key, tag, nonce):
        """Decrypt bytes with AES-GCM using pycryptodome"""
        ...
    
else:
    def aes_cbc_decrypt_bytes(data, key, iv): # -> bytes:
        """Decrypt bytes with AES-CBC using native implementation"""
        ...
    
    def aes_gcm_decrypt_and_verify_bytes(data, key, tag, nonce): # -> bytes:
        """Decrypt bytes with AES-GCM using native implementation"""
        ...
    
bytes_to_intlist = list
def intlist_to_bytes(xs): # -> bytes:
    ...

def unpad_pkcs7(data):
    ...

BLOCK_SIZE_BYTES = ...
def aes_ecb_encrypt(data, key, iv=...): # -> list[Any]:
    """
    Encrypt with aes in ECB mode

    @param {int[]} data        cleartext
    @param {int[]} key         16/24/32-Byte cipher key
    @param {int[]} iv          Unused for this mode
    @returns {int[]}           encrypted data
    """
    ...

def aes_ecb_decrypt(data, key, iv=...): # -> list[Any]:
    """
    Decrypt with aes in ECB mode

    @param {int[]} data        cleartext
    @param {int[]} key         16/24/32-Byte cipher key
    @param {int[]} iv          Unused for this mode
    @returns {int[]}           decrypted data
    """
    ...

def aes_ctr_decrypt(data, key, iv): # -> list[Any]:
    """
    Decrypt with aes in counter mode

    @param {int[]} data        cipher
    @param {int[]} key         16/24/32-Byte cipher key
    @param {int[]} iv          16-Byte initialization vector
    @returns {int[]}           decrypted data
    """
    ...

def aes_ctr_encrypt(data, key, iv): # -> list[Any]:
    """
    Encrypt with aes in counter mode

    @param {int[]} data        cleartext
    @param {int[]} key         16/24/32-Byte cipher key
    @param {int[]} iv          16-Byte initialization vector
    @returns {int[]}           encrypted data
    """
    ...

def aes_cbc_decrypt(data, key, iv): # -> list[Any]:
    """
    Decrypt with aes in CBC mode

    @param {int[]} data        cipher
    @param {int[]} key         16/24/32-Byte cipher key
    @param {int[]} iv          16-Byte IV
    @returns {int[]}           decrypted data
    """
    ...

def aes_cbc_encrypt(data, key, iv): # -> list[Any]:
    """
    Encrypt with aes in CBC mode. Using PKCS#7 padding

    @param {int[]} data        cleartext
    @param {int[]} key         16/24/32-Byte cipher key
    @param {int[]} iv          16-Byte IV
    @returns {int[]}           encrypted data
    """
    ...

def aes_gcm_decrypt_and_verify(data, key, tag, nonce): # -> list[Any]:
    """
    Decrypt with aes in GBM mode and checks authenticity using tag

    @param {int[]} data        cipher
    @param {int[]} key         16-Byte cipher key
    @param {int[]} tag         authentication tag
    @param {int[]} nonce       IV (recommended 12-Byte)
    @returns {int[]}           decrypted data
    """
    ...

def aes_encrypt(data, expanded_key): # -> list[Any]:
    """
    Encrypt one block with aes

    @param {int[]} data          16-Byte state
    @param {int[]} expanded_key  176/208/240-Byte expanded key
    @returns {int[]}             16-Byte cipher
    """
    ...

def aes_decrypt(data, expanded_key): # -> list[Any]:
    """
    Decrypt one block with aes

    @param {int[]} data          16-Byte cipher
    @param {int[]} expanded_key  176/208/240-Byte expanded key
    @returns {int[]}             16-Byte state
    """
    ...

def aes_decrypt_text(data, password, key_size_bytes): # -> bytes:
    """
    Decrypt text
    - The first 8 Bytes of decoded 'data' are the 8 high Bytes of the counter
    - The cipher key is retrieved by encrypting the first 16 Byte of 'password'
      with the first 'key_size_bytes' Bytes from 'password'
      (if necessary filled with 0's)
    - Mode of operation is 'counter'

    @param {str} data                    Base64 encoded string
    @param {str,unicode} password        Password (will be encoded with utf-8)
    @param {int} key_size_bytes          Possible values: 16 for 128-Bit,
                                                          24 for 192-Bit, or
                                                          32 for 256-Bit
    @returns {str}                       Decrypted data
    """
    ...

RCON = ...
SBOX = ...
SBOX_INV = ...
MIX_COLUMN_MATRIX = ...
MIX_COLUMN_MATRIX_INV = ...
RIJNDAEL_EXP_TABLE = ...
RIJNDAEL_LOG_TABLE = ...
def key_expansion(data):
    """
    Generate key schedule

    @param {int[]} data  16/24/32-Byte cipher key
    @returns {int[]}     176/208/240-Byte expanded key
    """
    ...

def iter_vector(iv): # -> Generator[Any, Any, NoReturn]:
    ...

def sub_bytes(data): # -> list[Any]:
    ...

def sub_bytes_inv(data): # -> list[Any]:
    ...

def rotate(data):
    ...

def key_schedule_core(data, rcon_iteration): # -> list[Any]:
    ...

def xor(data1, data2): # -> list[Any]:
    ...

def iter_mix_columns(data, matrix): # -> Generator[int | Any, Any, None]:
    ...

def shift_rows(data): # -> list[Any]:
    ...

def shift_rows_inv(data): # -> list[Any]:
    ...

def shift_block(data): # -> list[Any]:
    ...

def inc(data):
    ...

def block_product(block_x, block_y): # -> list[int] | list[Any]:
    ...

def ghash(subkey, data): # -> list[int] | list[Any]:
    ...
