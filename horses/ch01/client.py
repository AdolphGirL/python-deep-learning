# -*- coding: utf-8 -*-


import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 6688))

data = '這是測試文字'
data = data.encode('utf-8')
print('[client.py] client send data byte len: {}'.format(len(data)))
client.sendall(data)

rec_data = client.recv(1024)
print('[client.py] get server return type: {}'.format(type(rec_data)))
print('[client.py] get server return value: {}'.format(rec_data.decode('utf-8')))

client.close()
print('[client.py] client closed ... ')