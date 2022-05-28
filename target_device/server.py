import sys
import socket
from threading import Thread
import traceback
import errno
import select
import time


def client_thread(conn, ip, port, MAX_BUFFER_SIZE=4096):
    while True:
        input_from_client_bytes = conn.recv(MAX_BUFFER_SIZE)
        if input_from_client_bytes is None:
            break
        buffer_size = sys.getsizeof(input_from_client_bytes)
        if buffer_size >= MAX_BUFFER_SIZE:
            print(f'The length of input is probably too long: {buffer_size}')

        input_from_client = input_from_client_bytes.decode('utf8').rstrip()
        print(input_from_client)

        res = input_from_client + ' - accepted'
        vysl = res.encode('utf8')
        try:
            conn.sendall(vysl)
        except IOError as ie:
            if ie.errno == errno.EPIPE:
                print('Client socket closed, exiting')
                break
    conn.close()
    print('Connection to ' + ip + ':' + port + ' terminated')


def start_server():
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print('Socket created')

    try:
        soc.bind(('', 12345))
        print('Socket bind complete')
    except socket.error as msg:
        print('Bind failed. Error : ' + str(sys.exc_info()))
        sys.exit()

    soc.listen()
    print('Socket now listening')

    while True:
        r, _, _ = select.select((soc,), (), (), 20)
        for s in r:
            conn, addr = soc.accept()
            ip, port = str(addr[0]), str(addr[1])
            print('Accepting connection from ' + ip + ':' + port)
            try:
                Thread(target=client_thread, args=(conn, ip, port)).start()
            except EOFError:
                print('Keyboard interrupt received. Stop.')
                traceback.print_exc()
                soc.close()
                sys.exit(0)


if __name__ == '__main__':
    start_server()
