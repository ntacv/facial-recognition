from deepface import DeepFace

model_name = "Facenet"

result = DeepFace.verify(
  img1_path = "img1.png",
  img2_path = "img1.png",
  model_name = model_name
)
