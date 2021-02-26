# -*- coding: utf-8 -*-


import socket

# IPv4 TCP
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 6688))
server.listen(5)
print('[server.py] socket name: {}'.format(server.getsockname()))
print('[server.py] socket name: {}，waiting for connect...'.format(server.getsockname()))

# 一但有客戶端連接accept才會觸發
connect, (host, port) = server.accept()

peer_name = connect.getpeername()
sock_name = connect.getsockname()

print('[server.py] the client: {}:{}，has connected.'.format(host, port))
print('[server.py] peer name and socket name is: {}'.format(peer_name, sock_name))

# 讀取數據
data = connect.recv(1024)
print('[server.py] recv data type: {}，len: {}'.format(type(data), len(data)))
print('[server.py] recv data value : {}'.format(data.decode('utf-8')))

# 內部循環調用send，直到資料發送完畢
connect.sendall(b'your words has received.')

server.close()
print('[server.py] server closed ... ')
