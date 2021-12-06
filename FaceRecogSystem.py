# This file is for Face Recognition System.
# Version: 2.0
# @Author: Malay Bhavsar

# Importing the modules.
from os import path, makedirs, walk
import cv2
from numpy import array, shape
from sys import stdout

# Creating the class.


class leo_frs:
    def __init__(self, path="./"):
        self.path = path
        self.face = cv2.CascadeClassifier(
            "./data/cascade_data/haarcascade_frontalface_default.xml")
        self.name_ls = []
        self.file_ls = []
        self.max_confi = 0

    def __create_dir(self, name):
        # This function is used to create folder.
        if path.exists(f"{self.path}/{name}"):
            stdout.write("[LOG]: Folder Already Exists\n")
        else:
            makedirs(f"{self.path}/{name}")
            stdout.write(
                f"[LOG]: Folder named:'{name}' Created @ path:'{self.path}'\n")

    def __get_files(self, paths):
        # This function returns the files in that path except for system files.
        for root, subdir, files in walk(paths):
            for subd in subdir:
                if len(subd) != 0:
                    self.name_ls.append(subd)
            stdout.write(f"[LOG]: Reading File: {root}/{subdir}\n")
            for file_name in files:
                if file_name[0] != ".":
                    self.name_ls.append(path.basename(root))
                    img = cv2.imread("/".join([root, file_name]))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.file_ls.append(gray)
        self.name_ls = array(self.name_ls).ravel()
        self.name_ls = self.name_ls.tolist()

    def __get_name_ls(self, path):
        for root, subdir, files in walk(path):
            for subd in subdir:
                if len(subd) != 0:
                    self.name_ls.append(subd)
        self.name_ls = array(self.name_ls).ravel()
        self.name_ls = self.name_ls.tolist()

    def __get_faces(self, img):
        self.faces = self.face.detectMultiScale(img, 1.1, 5)

    def __draw_rect(self, x, y, w, h):
        self.frame = cv2.rectangle(
            self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def __put_text(self, label, confidence, x, y):
        # This function is used to put text over the box.
        if(label == -1):
            name = "Unknown"
            cv2.putText(self.frame, f"{name}",
                        (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        else:
            name = self.name_ls[label]
            cv2.putText(self.frame, f"{name} - {round((100 - confidence),2)}%",
                        (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    def __put_count_down(self, number):
        # This function is used for captureing the frame.
        cv2.putText(self.frame, f"{200-number}",
                    (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4)

    def __get_num_dir(self, path):
        # This function calculates number of folders.
        self.num_dir = 0
        for _, dirnames, filenames in walk(path):
            self.num_dir += len(dirnames)

    def import_img(self, path, name):
        self.__get_files(path)
        self.__create_dir("data/raw_data/"+name)
        num_photo = 0
        for file in self.file_ls:
            self.__get_faces(file)
            if len(self.faces) != 0:
                if len(self.faces) == 1:
                    for face in self.faces:
                        (x, y, w, h) = face
                        file = file[y:y+h, x:x+w]
                        stdout.write(
                            "[SUCCESS] Photo imported: " + str(num_photo) + "\n")
                        cv2.imwrite(
                            f"./data/raw_data/{name}/frame[{num_photo}].jpeg", file)
                        num_photo += 1
                else:
                    stdout.write("[ERROR]: Photo got more than one face.\n")
            else:
                stdout.write("[ERROR]: Photo got no face\n")

    def capture_start(self, num, name):
        # This function is used for capturing the raw frames.
        self.video = cv2.VideoCapture(num)
        path_save = f"data/raw_data/{name.lower()}"
        self.__create_dir(path_save)
        frame_count = 0
        frame_break = 1
        max_frame_capture = 200
        while 1:
            ret, self.frame = self.video.read()
            if ret == True and shape(self.frame) != ():
                self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.__get_faces(self.gray)
                if len(self.faces) != 0:
                    for face in self.faces:
                        (x, y, w, h) = face
                        self.__draw_rect(x, y, w, h)
                    if frame_count % frame_break == 0:
                        self.gray = self.gray[y:y+h, x:x+w]
                        stdout.write(
                            "[SUCCESS] Frame captured: " + str(int(frame_count/frame_break)) + "\n")
                        self.__put_count_down(int(frame_count/frame_break))
                        cv2.imwrite(
                            f"{path_save}/frame[{int(frame_count/frame_break)}].jpeg", self.gray)
                    if frame_count >= max_frame_capture:
                        break
                    frame_count += 1
                cv2.imshow("Live Cam For Capture", self.frame)
                if cv2.waitKey(1) == ord("q"):
                    break
        self.video.release()
        cv2.destroyAllWindows()

    def train(self):
        self.__get_files("./data/raw_data")
        self.__get_num_dir("./data/raw_data")
        label = []
        for name in self.name_ls:
            label.append(self.name_ls.index(name))
        stdout.write("[LOG]: Training Started\n")
        self.face_identify = cv2.face.LBPHFaceRecognizer_create()
        self.face_identify.train(self.file_ls, array(label[self.num_dir:]))
        stdout.write("[LOG]: Writing output file\n")
        self.face_identify.write("./data/data.yml")
        stdout.write("[SUCCESS]: Model trained\n")

    def predict(self, num):
        # This function is used for predicting the faces.
        self.__get_name_ls("./data/raw_data")
        self.face_identify = cv2.face.LBPHFaceRecognizer_create()
        stdout.write("[LOG]: Loading the trained data\n")
        self.face_identify.read("./data/data.yml")
        stdout.write("[SUCCESS]: Trained Data loaded\n")
        stdout.write("[LOG]: Starting the camera output\n")
        self.video = cv2.VideoCapture(num)
        while (True):
            ret, self.frame = self.video.read()
            if ret == True and shape(self.frame) != ():
                try:
                    self.__get_faces(self.frame)
                    if len(self.faces) != 0:
                        for face in self.faces:
                            (x, y, w, h) = face
                            self.__draw_rect(x, y, w, h)
                            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                            label, confidence = self.face_identify.predict(
                                gray[y:y+h, x:x+w])
                            if confidence <= 50:
                                self.__put_text(label, confidence, x, y)
                            else:
                                self.__put_text(-1, confidence, x, y)
                except:
                    self.__put_text(-1, confidence, x, y)
                cv2.imshow("Identify me! ( Q to exit )", self.frame)
                if cv2.waitKey(1) == ord("q"):
                    break
        self.video.release()
        cv2.destroyAllWindows()
        stdout.write("[LOG]: Camera Closed\n")


### Test ###
pt = leo_frs()

# Uncomment the following code and run multiple times with multiple people.(one people one time)
#pt.capture_start(0, "Malay")

# Uncomment the following code give path and name of the person to import its image.
#pt.import_img("./path", "Enter name")

# Uncomment the following code after you capturing some peoples photo.
# pt.train()

# Uncomment the following code after training.(Comment both of above line)
pt.predict(0)
