import socket

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect(('192.168.1.7', 12345))

clients_input = '13'
soc.send(clients_input.encode('utf8'))
result_bytes = soc.recv(4096)
result_string = result_bytes.decode('utf8')

print(f'Result from server is {result_string}')
