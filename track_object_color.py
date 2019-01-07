import cv2
import numpy as np
from skimage import data, io, color, transform, exposure
import os
from matplotlib import pyplot as plt
from threading import Thread
from queue import Queue
import json
from pprint import pprint
import time
import operator
import copy

class FileVideoStream:
    def __init__(self, path, queueSize = 1024):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize = queueSize)

    def start(self):
        t = Thread(target = self.update, args = ())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                    return

                self.Q.put(frame)
    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True

def only_people(file, data):
    loading = False

    fvs = cv2.VideoCapture(file)

    i = 0

    success, frame = fvs.read()

    cropped_frame = {}

    for i in range(20):
#     while success:

        if i in data:
            people = len(data[i])
            for j in range(0, people):
                tmp_points = copy.deepcopy(data[i][j])
                x_points = copy.deepcopy(data[i][j])
                y_points = {}

                for point in tmp_points:
                    if x_points[point][0] == 0:
                        del x_points[point]

                    if tmp_points[point][1] != 0:
                        y_points[point] = tmp_points[point][1]


                minY = min(y_points.items(), key = operator.itemgetter(1))[1]
                maxY = max(y_points.items(), key = operator.itemgetter(1))[1]


                maxX, GY = max(x_points.items(), key = operator.itemgetter(1))[1]
                minX, GY = min(x_points.items(), key = operator.itemgetter(1))[1]

                minY = int(minY)
                maxY = int(maxY)
                minX = int(minX)
                maxX = int(maxX)

                person_frame = np.zeros(((maxY - minY), (maxX - minX), 3))

                for y in range(minY, maxY):
                    for x in range(minX, maxX):
                        person_frame[y - minY][x - minX] = frame[y][x]

                hsv = cv2.cvtColor(person_frame, cv2.COLOR_BGR2HSV)

                plt.imshow(person_frame)
                plt.show()

        i += 1

        success, frame = fvs.read()

def read_jsons(folder):
    jsons = [json for json in os.listdir(folder) if json.endswith(".json")]

    return jsons

def data_points(jsons, json_folder_dir):
    data_points = []

    for i in range(0, len(jsons)):
        data_points.append([])
        with open((json_folder_dir + jsons[i])) as f:
            data = json.load(f)

        n_people = len(data['people'])

        for j in range(0, n_people):
            data_points[i].append([])
            datas = data['people'][j]['pose_keypoints_2d']

            for k in range(0, len(datas)):
                data_points[i][j].append(datas[k])

    return data_points

def organise_data(data):
    total = len(data)
    people_body_point = {}

    for i in range(0, total):
        people = len(data[i])
        frame_points = {}
        if people != 0:
            for j in range(0, people):
                points = data[i][j]
                n_points = len(points)
                body_points = []
                for k in range(0, n_points - 1, 3):
                    item = int(k/3)
                    body_points.append((points[0 + k], points[1 + k]))

                frame_points[j] = body_points

            organised = data_classification(frame_points)
            people_body_point[i] = organised

    return people_body_point


def data_classification(data):
    body_label = {0:"Nose", 1:"Neck", 2:"RShoulder", 3:"RElbow", 4:"RWrist", 5:"LShoulder", 6:"LElbow", 7:"LWrist", 8:"MidHip", 9:"RHip", 10:"RKnee", 11:"RAnkle", 12:"LHip", 13:"LKnee",14:"LAnkle", 15:"REye", 16:"LEye", 17:"REar", 18:"LEar", 19:"LBigToe", 20:"LSmallToe", 21:"LHeel", 22:"RBigToe", 23:"RSmallToe", 24:"RHeel"}
    parts = {}
    people = {}

    for person in range(0, len(data)):
        points = data[person]
        for i in range(0, len(points)):
            parts[body_label[i]] = points[i]

        people[person] = parts
        parts = {}

    return people



json_folder = 'DS3'
json_folder_slash = 'DS3/'

jsons = read_jsons(json_folder)
data_points1 = data_points(jsons, json_folder_slash)

people_body_part_points = organise_data(data_points1)

only_people('DS2.MOV', people_body_part_points)
