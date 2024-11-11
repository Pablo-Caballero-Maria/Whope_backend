import base64
import os
from typing import Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    CipherContext,
    algorithms,
    modes,
)


def generate_asymmetric_keys(password: bytes) -> Tuple[bytes, bytes]:
    private_key: rsa.RSAPrivateKey = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    public_key: rsa.RSAPublicKey = private_key.public_key()

    private_key_bytes: bytes = private_key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.BestAvailableEncryption(password))

    public_key_bytes: bytes = public_key.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo)

    return private_key_bytes, public_key_bytes


def decrypt_with_private_key(encrypted_data_b64: bytes, private_key_bytes: bytes) -> bytes:
    # this function is used to decrypt the client's symmetric key with the server's private key
    from whope.settings import PRIVATE_KEY_PASSWORD

    encrypted_data: bytes = base64.b64decode(encrypted_data_b64)
    private_key: rsa.RSAPrivateKey = serialization.load_pem_private_key(private_key_bytes, password=PRIVATE_KEY_PASSWORD, backend=default_backend())
    decrypted_data: bytes = private_key.decrypt(encrypted_data, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return decrypted_data


def encrypt_with_symmetric_key(data: str, symmetric_key: bytes) -> str:
    # Generamos un IV (Vector de Inicialización) de 16 bytes de todo ceros
    iv: bytes = os.urandom(16)

    # Creamos un objeto de cifra AES-GCM
    cipher: Cipher = Cipher(algorithms.AES(symmetric_key), modes.GCM(iv), backend=default_backend())
    encryptor: CipherContext = cipher.encryptor()

    # Encriptamos los datos
    ciphertext: bytes = encryptor.update(data.encode()) + encryptor.finalize()
    tag: bytes = encryptor.tag
    combined: bytes = iv + ciphertext + tag
    encoded: str = base64.b64encode(combined).decode("utf-8")

    return encoded


def decrypt_with_symmetric_key(encrypted_data: bytes, symmetric_key: bytes) -> str:
    # El primer paso es extraer el IV (16 bytes) y el tag de autenticación
    combined: bytes = base64.b64decode(encrypted_data)
    iv: bytes = combined[:16]  # El IV es de 16 bytes
    tag: bytes = combined[-16:]  # El tag de autenticación de 16 bytes (tamaño de AES-GCM)

    # Los datos encriptados están entre el IV y el tag
    cipher_text: bytes = combined[16:-16]

    # Creamos el cifrador AES-GCM con el IV y la clave simétrica
    cipher = Cipher(algorithms.AES(symmetric_key), modes.GCM(iv, tag), backend=default_backend())
    decryptor = cipher.decryptor()

    # Desencriptamos los datos
    decrypted_data: bytes = decryptor.update(cipher_text) + decryptor.finalize()

    # Devolvemos los datos desencriptados como un string
    return decrypted_data.decode("utf-8")
