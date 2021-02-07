# -*- coding: utf-8 -*-


import socket


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 1234))
client_socket.send('今天天氣很好'.encode('utf-8'))
data = client_socket.recv(1024)
print('[client-socket] recv: {}'.format(data.decode('utf-8')))
client_socket.close()