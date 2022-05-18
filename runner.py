import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import *
import transforms as t
import matplotlib.pyplot as plt
import json
import socket
import utils as u
import getopt
import sys

options = 'h'
long_options = ['help', 'IP=']


def set_params():
    ip_input = ''
    try:
        arguments, values = getopt.getopt(sys.argv[1:], options, long_options)
        for opt, arg in arguments:
            if opt in ('-h', '--help'):
                print('IP - IP address of the target system accepting TCP connections.')
            elif opt == '--IP':
                ip_input = str(arg)
        print(f'Set target IP={ip_input}')
    except getopt.error as err:
        print(str(err))
    return ip_input


soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connected = False
ip = set_params()
try:
    soc.connect((ip, 12345))
    connected = True
except OSError as oe:
    print(f'Could not connect to target ip: {oe}')

with open('./configs.json') as data_file:
    config = json.load(data_file)

label_dict = pd.read_csv(config['full_labels_csv'], header=None)
ges = label_dict[0].tolist()


seq_len = 16
value = 0
imgs = []
pred = 8
top_3 = [9, 8, 7]
out = np.zeros(10)
print('Loading model')

curr_folder = './'
model = u.CombinedModel(batch_size=1, seq_lenght=16)
loaded_dict = torch.load(curr_folder + 'model_40_10.ckp')
model.load_state_dict(loaded_dict)
model = model.cuda()
model.eval()

std, mean = [0.2674, 0.2676, 0.2648], [0.4377, 0.4047, 0.3925]
transform = Compose([
    t.CenterCrop((96, 96)),
    t.ToTensor(),
    t.Normalize(std=std, mean=mean),
])

print('Starting prediction')

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 30)

n = 0
hist = []
mean_hist = []
setup = True
plt.ion()
fig, ax = plt.subplots()
eval_samples = 5
num_classes = 27
timeout = 64
frames_for_detection = 16

score_energy = torch.zeros((eval_samples, num_classes))

while True:
    ret, frame = cam.read()
    resized_frame = cv2.resize(frame, (160, 120))
    pre_img = Image.fromarray(resized_frame.astype('uint8'), 'RGB')
    img = transform(pre_img)

    if n % 4 == 0:
        imgs.append(torch.unsqueeze(img, 0))

    if len(imgs) == frames_for_detection:
        data = torch.cat(imgs).cuda()
        out = model(data.unsqueeze(0)).squeeze().data.cpu().numpy()
        if len(hist) > 500:
            mean_hist = mean_hist[1:]
            hist = hist[1:]
        hist.append(out)
        score_energy = torch.tensor(np.array(hist[-eval_samples:]))
        curr_mean = torch.mean(score_energy, dim=0)
        mean_hist.append(curr_mean.cpu().numpy())
        value, indice = torch.topk(curr_mean, k=1)
        indices = np.argmax(out)
        _, top_3 = torch.topk(curr_mean, k=3)
        if timeout > 0:
            timeout -= 1
        if value.item() > -1.7 and indices < 25 and not timeout:
            print('Gesture:', ges[indices], '\t\t\t\t\t\t Value: {:.2f}'.format(value.item()))
            if connected:
                soc.send(str(indices).encode('utf8'))
                result_bytes = soc.recv(4096)
                result_string = result_bytes.decode('utf8')
                print(f'Received from target: {result_string}')
            timeout = frames_for_detection
        pred = indices
        imgs = imgs[1:]

    n += 1
    bg = np.full((480, 1200, 3), 15, np.uint8)
    bg[:480, :640] = frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    if value > -1.7:
        cv2.putText(bg, ges[pred], (40, 40), font, 1, (0, 0, 0), 2)
    cv2.rectangle(bg, (128, 48), (640 - 128, 480 - 48), (0, 255, 0), 3)
    for i, top in enumerate(reversed(top_3)):
        cv2.putText(bg, ges[top], (700, 200 - 70 * i), font, 1, (255, 255, 255), 1)
        cv2.putText(bg, str(round(np.exp(out[top]), 3)), (750, 230 - 70 * i), font, 1, (255, 255, 255), 1)

    cv2.imshow('preview', bg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if connected:
    soc.close()
cam.release()
cv2.destroyAllWindows()
