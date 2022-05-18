import sys
import socket
from threading import Thread
import traceback
import logging

logger - logging.getLogger('logger_name')


def process(input_string):
    logger.info(input_string)
    return input_string + ' Accepted!'


def client_thread(conn, ip, port, MAX_BUFFER_SIZE=4096):
    input_from_client_bytes = conn.recv(MAX_BUFFER_SIZE)

    buffer_size = sys.getsizeof(input_from_client_bytes)
    if buffer_size >= MAX_BUFFER_SIZE:
        logger.info(f'The length of input is probably too long: {buffer_size}')

    input_from_client = input_from_client_bytes.decode('utf8').rstrip()

    res = process(input_from_client)
    logger.info(f'Result of processing {input_from_client} is: {res}')

    vysl = res.encode('utf8')
    conn.sendall(vysl)
    conn.close()
    logger.info('Connection to ' + ip + ':' + port + ' terminated')


def start_server():
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    logger.info('Socket created')

    try:
        soc.bind(('', 12345))
        logger.info('Socket bind complete')
    except socket.error as msg:
        logger.info('Bind failed. Error : ' + str(sys.exc_info()))
        sys.exit()

    soc.listen(10)
    logger.info('Socket now listening')

    while True:
        conn, addr = soc.accept()
        ip, port = str(addr[0]), str(addr[1])
        logger.info('Accepting connection from ' + ip + ':' + port)
        try:
            Thread(target=client_thread, args=(conn, ip, port)).start()
        except EOFError:
            logger.info('EOF interrupt received. Stop.')
            traceback.print_exc()
            sys.exit(0)

    soc.close()


start_server()
