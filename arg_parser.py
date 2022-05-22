import getopt
import sys
from auth import Auth


def set_params(options: str, long_options: list):
    ip_input = ''
    auth_object = None
    try:
        arguments, values = getopt.getopt(sys.argv[1:], options, long_options)
        for opt, arg in arguments:
            if opt in ('-h', '--help'):
                print('IP - IP address of the target system accepting TCP connections.')
            elif opt == '--IP':
                ip_input = str(arg)
                print(f'Received target IP: {ip_input}')
                break
            elif opt == '--hass':
                auth_object = Auth('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9'
                                   '.eyJpc3MiOiIzNTc0ZThiNDIwYzc0MDBiYmN'
                                   'hMWQyNjU0MTBjNWMwNSIsImlhdCI6MTY1MDM'
                                   'yNjE5MiwiZXhwIjoxOTY1Njg2MTkyfQ.7FpT'
                                   'HLidC4XUnPQ0Y_gMFKLinxTfzjeby4JHwQ44lSw'
                                   )
                print(f'Received Home Assistant IP')
                break
    except getopt.error as err:
        print(str(err))
    return ip_input, auth_object
