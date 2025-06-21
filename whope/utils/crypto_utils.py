import base64
import os

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    CipherContext,
    algorithms,
    modes,
)


def decrypt_with_private_key(encrypted_data_b64: bytes, private_key_bytes: bytes) -> bytes:
    # this function is used to decrypt the client's symmetric key with the server's private key
    from whope.settings import PRIVATE_KEY_PASSWORD

    encrypted_data: bytes = base64.b64decode(encrypted_data_b64)
    private_key: rsa.RSAPrivateKey = serialization.load_pem_private_key(private_key_bytes, password=PRIVATE_KEY_PASSWORD)
    decrypted_data: bytes = private_key.decrypt(encrypted_data, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return decrypted_data


def encrypt_with_symmetric_key(data: str, symmetric_key: bytes) -> str:
    iv: bytes = os.urandom(16)

    cipher: Cipher = Cipher(algorithms.AES(symmetric_key), modes.GCM(iv))
    encryptor: CipherContext = cipher.encryptor()

    ciphertext: bytes = encryptor.update(data.encode()) + encryptor.finalize()
    tag: bytes = encryptor.tag
    combined: bytes = iv + ciphertext + tag
    encoded: str = base64.b64encode(combined).decode("utf-8")

    return encoded


def decrypt_with_symmetric_key(encrypted_data: str, symmetric_key: bytes) -> str:
    combined: bytes = base64.b64decode(encrypted_data.encode("utf-8"))
    iv: bytes = combined[:16]
    tag: bytes = combined[-16:]
    cipher_text: bytes = combined[16:-16]

    cipher = Cipher(algorithms.AES(symmetric_key), modes.GCM(iv, tag))
    decryptor = cipher.decryptor()

    decrypted_data: bytes = decryptor.update(cipher_text) + decryptor.finalize()

    return decrypted_data.decode("utf-8")
