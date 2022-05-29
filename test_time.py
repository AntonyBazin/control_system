from datetime import datetime
import json
import socket

from auth import Auth

auth = Auth(f'http://192.168.1.127:8123/api/states/sensor.gestures',
            'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9'
            '.eyJpc3MiOiIzNTc0ZThiNDIwYzc0MDBiYmN'
            'hMWQyNjU0MTBjNWMwNSIsImlhdCI6MTY1MDM'
            'yNjE5MiwiZXhwIjoxOTY1Njg2MTkyfQ.7FpT'
            'HLidC4XUnPQ0Y_gMFKLinxTfzjeby4JHwQ44lSw',
            )

resp = auth.request('POST',
                    headers={'content-type': 'application/json'},
                    data=json.dumps({'state': '0', 'attributes': {'friendly_name': 'Recognized gestures',
                                                                  'gesture_name': 'No gesture'}}))
times = []
for i in range(50):
    time = datetime.now()
    resp = auth.request('POST',
                        headers={'content-type': 'application/json'},
                        data=json.dumps({'state': f'{i}', 'attributes': {'friendly_name': 'Recognized gestures',
                                                                         'gesture_name': 'Test gesture'}}))
    resp_time = datetime.now()
    # print(time, resp_time, resp_time - time)
    times.append(resp_time - time)
    # print(time - (datetime.strptime(json.loads(resp.content.decode('utf8'))['last_changed'],
    # '%Y-%m-%dT%H:%M:%S.%f%z')).replace(tzinfo=None))
print('HASS:')
print([i.microseconds for i in times])

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect(('192.168.1.127', 12345))

tcp_times = []
for i in range(50):
    time = datetime.now()

    soc.send('0'.encode('utf8'))
    result_bytes = soc.recv(4096)

    resp_time = datetime.now()
    tcp_times.append(resp_time - time)

print('TCP:')
print([i.microseconds for i in tcp_times])
