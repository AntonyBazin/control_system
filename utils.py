import torch
import torch.nn as nn
import math
from torchvision.transforms import *
import numbers
import random
from PIL import Image
import os
import glob
import csv
from collections import namedtuple


class FullModel(nn.Module):

    def __init__(self, batch_size, seq_lenght=8):
        super(FullModel, self).__init__()

        class CNN2D(nn.Module):
            def __init__(self, batch_size=batch_size, image_size=96, seq_lenght=8, in_channels=3):
                super(CNN2D, self).__init__()
                self.conv1 = self._create_conv_layer(in_channels=in_channels, out_channels=16)
                self.conv2 = self._create_conv_layer(in_channels=16, out_channels=32)
                self.conv3 = self._create_conv_layer_pool(in_channels=32, out_channels=64)
                self.conv4 = self._create_conv_layer_pool(in_channels=64, out_channels=128)
                self.conv5 = self._create_conv_layer_pool(in_channels=128, out_channels=256)

            def forward(self, x):
                batch_size, frames, channels, width, height = x.shape
                x = x.view(-1, channels, width, height)
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.conv5(x)
                return x

            def _create_conv_layer(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )

            def _create_conv_layer_pool(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1),
                                        pool=(2, 2)):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(pool)
                )

        class CNN3D(nn.Module):
            def __init__(self, batch_size=batch_size, image_size=96, seq_lenght=8):
                super(CNN3D, self).__init__()
                self.conv1 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(1, 1, 1))
                self.conv2 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(2, 2, 2))
                self.conv3 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(2, 1, 1))
                self.conv4 = self._create_conv_layer_pool(in_channels=256, out_channels=256, pool=(2, 2, 2))

            def forward(self, x):
                batch_size, channels, frames, width, height = x.shape
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                return x

            def _create_conv_layer(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
                return nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(),
                )

            def _create_conv_layer_pool(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                                        pool=(1, 2, 2)):
                return nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool3d(pool)
                )

        class Combiner(nn.Module):

            def __init__(self, in_features):
                super(Combiner, self).__init__()
                self.linear1 = self._create_linear_layer(in_features, in_features // 2)
                self.linear2 = self._create_linear_layer(in_features // 2, 1024)
                self.linear3 = self._create_linear_layer(1024, 27)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x

            def _create_linear_layer(self, in_features, out_features, p=0.6):
                return nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.Dropout(p=p)
                )

        self.rgb2d = CNN2D(batch_size)
        self.rgb3d = CNN3D(batch_size)
        self.combiner = Combiner(4608)

        self.batch_size = batch_size
        self.seq_lenght = seq_lenght
        self.steps = 0
        self.steps = 0
        self.epochs = 0
        self.best_validation_loss = math.inf

    def forward(self, x):
        self.batch_size = x.shape[0]
        x = self.rgb2d(x)
        batch_and_frames, channels, dim1, dim2 = x.shape
        x = x.view(self.batch_size, -1, channels, dim1, dim2).permute(0, 2, 1, 3, 4)
        x = self.rgb3d(x)
        x = x.view(self.batch_size, -1)
        x = self.combiner(x)

        if self.training:
            self.steps += 1

        return x


class CombinedModel(nn.Module):
    def __init__(self, batch_size, seq_lenght=8):
        super().__init__()

        class Combiner(nn.Module):

            def __init__(self, in_features):
                super(Combiner, self).__init__()
                self.linear1 = self._create_linear_layer(in_features, 32)
                self.LSTM = nn.LSTM(32, 32, 1)
                self.linear2 = self._create_linear_layer(32, 27)
                self.logsoftmax = nn.LogSoftmax(-1)

            def forward(self, x):
                x = self.linear1(x).unsqueeze(0)
                x, _ = self.LSTM(x)
                x = self.linear2(x)
                x = self.logsoftmax(x)
                return x

            def _create_linear_layer(self, in_features, out_features, p=0.6):
                return nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.Dropout(p=p)
                )

        self.fm = FullModel(batch_size, seq_lenght)
        self.combiner = Combiner(4608)
        self.fc = nn.Linear(27, 27)
        self.batch_size = batch_size
        self.seq_lenght = seq_lenght
        self.steps = 0
        self.steps = 0
        self.epochs = 0
        self.best_validation_loss = math.inf

    def forward(self, x):
        self.batch_size = x.shape[0]
        x = self.fm.rgb2d(x)
        batch_and_frames, channels, dim1, dim2 = x.shape
        x = x.view(self.batch_size, -1, channels, dim1, dim2).permute(0, 2, 1, 3, 4)
        x = self.fm.rgb3d(x)
        x = x.view(self.batch_size, -1)
        x = self.combiner(x)

        if self.training:
            self.steps += 1

        return x


def calculate_loss_and_accuracy(validation_loader, model, criterion, stop_at=1200, print_every=99999):
    correct = 0
    total = 0
    steps = 0
    total_loss = 0
    sz = len(validation_loader)

    for images, labels in validation_loader:

        if total % print_every == 0 and total > 0:
            accuracy = 100 * correct / total
            print(accuracy)

        if total >= stop_at:
            break
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)

        loss = criterion(outputs.squeeze(), labels)
        total_loss += loss.item()

        p, predicted = torch.max(outputs.squeeze().data, 1)

        predicted = predicted.squeeze()

        total += labels.size(0)
        steps += 1
        correct += (predicted == labels).sum().item()

        del outputs, loss, p, predicted

    accuracy = 100 * correct / total
    return total_loss / steps, accuracy


class GroupToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img_group):
        return [ToTensor()(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        return [CenterCrop(self.size)(img) for img in img_group]


class GroupResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        return [Resize(self.size)(img) for img in img_group]


class GroupResizeFit(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        print(img_group[0].size)
        return [Resize(self.size)(img) for img in img_group]


class GroupExpand(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        tw, th = self.size
        out_images = list()
        if (w >= tw and h >= th):
            assert img_group[0].size == self.size
            return img_group
        for img in img_group:
            new_im = Image.new("RGB", (tw, th))
            new_im.paste(img, ((tw - w) // 2, (th - w) // 2))
            out_images.append(new_im)
        assert out_images[0].size == self.size
        return out_images


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size  # 120, 160
        th, tw = self.size  # 100, 140
        out_images = list()
        if (w - tw) < 0:
            print('W < TW')
            for img in img_group:
                new_im = Image.new("RGB", (tw, th))
                new_im.paste(img_group[0], ((tw - w) // 2, (th - w) // 2))
                out_images.append(new_im)
            return out_images

        x1 = random.randint(0, (w - tw))
        y1 = random.randint(0, abs(h - th))
        for img in img_group:
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupRandomRotation(object):
    def __init__(self, max):
        self.max = max

    def __call__(self, img_group):
        angle = random.randint(-self.max, self.max)
        return [functional.rotate(img, angle) for img in img_group]


class GroupNormalize(object):
    def __init__(self, given_mean, std):
        self.mean = given_mean
        self.std = std

    def __call__(self, tensor_list):
        for t, m, s in zip(tensor_list, self.mean, self.std):
            t.sub_(m).div_(s)

        return tensor_list


class GroupUnormalize(object):
    def __init__(self, given_mean, std):
        self.mean = given_mean
        self.std = std

    def __call__(self, tensor_list):
        for t, m, s in zip(tensor_list, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor_list


ListDataJpeg = namedtuple('ListDataJpeg', ['id', 'label', 'path'])
ListDataGulp = namedtuple('ListDataGulp', ['id', 'label'])


class JpegDataset(object):

    def __init__(self, csv_path_input, csv_path_labels, data_root):
        self.csv_data = self.read_csv_input(csv_path_input, data_root)
        self.classes = self.read_csv_labels(csv_path_labels)
        self.classes_dict = self.get_two_way_dict(self.classes)

    def read_csv_input(self, csv_path, data_root):
        csv_data = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                item = ListDataJpeg(row[-2],
                                    row[1],
                                    os.path.join(data_root, row[0])
                                    )
                csv_data.append(item)
        return csv_data[1:]

    def read_csv_labels(self, csv_path):
        classes = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                classes.append(row[0])
        return classes

    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']


def default_loader(path):
    return Image.open(path).convert('RGB')


class VideoFolder(torch.utils.data.Dataset):

    def __init__(self, root, csv_file_input, csv_file_labels, clip_size,
                 nclips, step_size, is_val, transform=None,
                 loader=default_loader):
        self.dataset_object = JpegDataset(
            csv_file_input, csv_file_labels, root)

        self.csv_data = self.dataset_object.csv_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform = transform
        self.loader = loader

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val

    def __getitem__(self, index):
        item = self.csv_data[index]
        try:
            img_paths = self.get_frame_names(item.path)
        except Exception:
            print(item, index)

        imgs = []
        for img_path in img_paths:
            img = self.loader(img_path)
            # img = self.transform(img)
            imgs.append(img)
        target_idx = self.classes_dict[item.label]

        # added_line
        imgs = self.transform(imgs)
        # format data to torch
        data = torch.stack(imgs)
        return (data, target_idx)

    def __len__(self):
        return len(self.csv_data)

    def get_frame_names(self, path):
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
        frame_names = list(sorted(frame_names))
        num_frames = len(frame_names)

        if not num_frames:
            print(frame_names, num_frames_necessary, num_frames)
            print(glob.glob(os.path.join(path, "*" + ext)))
            print(path)

        # set number of necessary frames
        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames

        # pick frames
        offset = 0
        if num_frames_necessary > num_frames and frame_names:
            # pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * \
                           (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset
            diff = (num_frames - num_frames_necessary)
            # Temporal augmentation
            if not self.is_val:
                # offset = np.random.randint(0, diff)
                offset = diff // 2
        frame_names = frame_names[offset:num_frames_necessary + offset:self.step_size]
        return frame_names


print('Utility script is ready')
