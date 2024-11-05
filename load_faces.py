from utils import FaceRec


face_rec = FaceRec("db/database.json")

face_rec.load_image_encodings("images/")