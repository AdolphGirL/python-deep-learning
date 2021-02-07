# -*- coding: utf-8 -*-


import socket


# AF_INET: IPV4 SOCK_STREAM: TCP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# server_socket.bind(socket.gethostname(), 1234)
server_socket.bind(('127.0.0.1', 1234))
server_socket.listen(5)
while True:
    cnn, address_info = server_socket.accept()
    print('[server-socket] connect from: {}'.format(address_info))

    # 讀取1024個bytes
    data = cnn.recv(1024)
    print('[server-socket] recv from: {}'.format(data.decode('utf-8')))
    cnn.send(data)
    cnn.close()

server_socket.close()

