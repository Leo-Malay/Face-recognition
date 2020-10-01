# This file is for Face Recognition System.
# Version: 1.0
# @Author: Malay Bhavsar

# Importing the modules.
import os
import cv2
import numpy as np

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
        if os.path.exists(f"{self.path}/{name}"):
            print("[LOG]: Folder Already Exists")
        else:
            os.makedirs(f"{self.path}/{name}")
            print(f"[LOG]: Folder named:'{name}' Created @ path:'{self.path}'")

    def __get_files(self, path):
        # This function returns the files in that path except for system files.
        for root, subdir, files in os.walk(path):
            for subd in subdir:
                if len(subd) != 0:
                    self.name_ls.append(subd)
            print(f"[LOG]: Reading File: {root}/{subdir}")
            for file_name in files:
                if file_name[0] == ".":
                    pass
                self.name_ls.append(os.path.basename(root))
                img = cv2.imread("/".join([root, file_name]))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.file_ls.append(gray)
        self.name_ls = np.array(self.name_ls).ravel()
        self.name_ls = self.name_ls.tolist()

    def __get_name_ls(self, path):
        for root, subdir, files in os.walk(path):
            for subd in subdir:
                if len(subd) != 0:
                    self.name_ls.append(subd)
        self.name_ls = np.array(self.name_ls).ravel()
        self.name_ls = self.name_ls.tolist()

    def __get_faces(self, img):
        self.faces = self.face.detectMultiScale(img, 1.1, 5)

    def __draw_rect(self, x, y, w, h):
        self.frame = cv2.rectangle(
            self.frame, (x, y), (x+w, y+h), (0,255, 0), 2)

    def __put_text(self, label, confidence, x, y):
        # This function is used to put text over the box.
        cv2.putText(self.frame, f"{self.name_ls[label]} - {round((100 - confidence),2)}%",
                    (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)

    def __put_count_down(self, number):
        # This function is used for captureing the frame.
        cv2.putText(self.frame, f"{50-number}",
                    (10,40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4)

    def __get_num_dir(self, path):
        # This function calculates number of folders.
        self.num_dir = 0
        for _, dirnames, filenames in os.walk(path):
            self.num_dir += len(dirnames)

    def import_img(self,path,name):
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
                        print(
                            f"[SUCCESS] Photo imported: {num_photo}")
                        cv2.imwrite(
                            f"./data/raw_data/{name}/frame[{num_photo}].jpeg", file)
                        num_photo +=1
                else:
                    print("[ERROR]: Photo got more than one face.")
            else:
                print(f"[ERROR]: Photo got no face")           

    def capture_start(self, num, name):
        # This function is used for capturing the raw frames.
        self.video = cv2.VideoCapture(num)
        path_save = f"data/raw_data/{name.lower()}"
        self.__create_dir(path_save)
        frame_count = 0
        while (True):
            ret, self.frame = self.video.read()
            if ret == True:
                self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.__get_faces(self.gray)
                if len(self.faces) != 0:
                    for face in self.faces:
                        (x, y, w, h) = face
                        self.__draw_rect(x, y, w, h)
                    if frame_count % 3 == 0:
                        self.gray = self.gray[y:y+h, x:x+w]
                        print(
                            f"[SUCCESS] Frame captured: {int(frame_count/3)}")
                        self.__put_count_down(int(frame_count/3))
                        cv2.imwrite(
                            f"{path_save}/frame[{int(frame_count/3)}].jpeg", self.gray)
                    if frame_count >= 150:
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
        print("[LOG]: Training Started")
        self.face_identify = cv2.face.LBPHFaceRecognizer_create()
        self.face_identify.train(self.file_ls, np.array(label[self.num_dir:]))
        print("[LOG]: Writing output file")
        self.face_identify.write("./data/data.yml")
        print("[SUCCESS]: Model trained")

    def predict(self, num):
        # This function is used for predicting the faces.
        self.__get_name_ls("./data/raw_data")
        self.face_identify = cv2.face.LBPHFaceRecognizer_create()
        print("[LOG]: Loading the trained data")
        self.face_identify.read("./data/data.yml")
        print("[SUCCESS]: Trained Data loaded")
        self.video = cv2.VideoCapture(num)
        print("[LOG]: Starting the camera output")
        while (True):
            ret, self.frame = self.video.read()
            if ret == True:
                try:
                    self.__get_faces(self.frame)
                    if len(self.faces) != 0:
                        for face in self.faces:
                            (x, y, w, h) = face
                            self.__draw_rect(x, y, w, h)
                            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                            label, confidence = self.face_identify.predict(
                                gray[y:y+h, x:x+w])
                            if confidence <= 100:
                                self.__put_text(label, confidence, x, y)
                            print(f"Confidence: {confidence}")
                except:
                    print("[LOG]: No face")
                cv2.imshow("Identify me! ( Q to exit )", self.frame)
                if cv2.waitKey(1) == ord("q"):
                    break
        self.video.release()
        cv2.destroyAllWindows()
        print("[LOG]: Camera Closed")


### Test ###
#pt = leo_frs()

# Uncomment the following code and run multiple times with multiple people.(one people one time)
#pt.capture_start(0, "xyz")

# Uncomment the following code give path and name of the person to import its image.
#pt.import_img("./path","Enter name")

# Uncomment the following code after you capturing some peoples photo.
#pt.train()

# Uncomment the following code after training.(Comment both of above line)
#pt.predict(0)
