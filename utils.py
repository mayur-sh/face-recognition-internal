import face_recognition as fr
import pysondb
import numpy as np
import cv2
import filetype
import os

class FaceRec:
    def __init__(self, db_path):
        self.db = pysondb.getDb(db_path)
        self.known_face_encodings = [ np.array(d['face_encoding']) for d in self.db.getAll() ]
        self.known_face_names = [ d['name'] for d in self.db.getAll() ]

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which fr uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in self.known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / 0.25
        return face_locations.astype(int), face_names
    
    def get_known_face_names(self):
        return self.known_face_names
    
    def load_image_encodings(self, image_folder_path):
        # Getting the image folder and filtering the images and making a dictionary of names and paths

        names_paths = {}

        for f in os.listdir(f'{image_folder_path}'):
            if filetype.guess( f"{image_folder_path}/" + f ).mime.split("/")[0] == 'image':
                names_paths[f.replace("."+filetype.guess_extension(f"{image_folder_path}/" + f), '' )] = f"{image_folder_path}/" + f

        # Reading those images with fr module
        # dbbo : database_object
        dbo = {}
        for name in names_paths:
            # name = 'Chakra'
            if name not in self.known_face_names:
                path = names_paths[name]
                img = fr.load_image_file(path)
                fe = fr.face_encodings(img, num_jitters=10, model="large")

                dbo["name"] = name
                for n,arr in enumerate(fe):
                    fe[n] = list(arr)
                dbo["face_encoding"] = fe[0]

                self.db.add(dbo)
                print(f"Embeddings for {name} loaded in DB.. Reload the Database to detect the loaded faces..")

        