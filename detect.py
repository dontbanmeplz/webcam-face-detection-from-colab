import face_recognition
import cv2
import numpy as np
import time
import os
import glob

class detect:
  def __init__():
    self.known_face_encodings = []
    self.known_face_names = []
    for face in glob.glob(os.path.join(os.getcwd(), "faces", "*.npy")):
    	    temp = np.load(face)
    	    self.known_face_encodings.append(temp)
    	    self.known_face_names.append(os.path.basename(face[:-4]))
    def detect(frame):
      rgb_frame = frame[:, :, ::-1]

      face_locations = face_recognition.face_locations(rgb_frame)
      face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
      name = "Unknown"

      for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
          # See if the face is a match for the known face(s)
          matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

          face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
          best_match_index = np.argmin(face_distances)
          if matches[best_match_index]:
              name = self.known_face_names[best_match_index]

          cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

          cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
          font = cv2.FONT_HERSHEY_DUPLEX
          cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
      return frame 
