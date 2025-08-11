from fastapi import FastAPI,UploadFile,File
from PIL import Image
import io
import torch
from torchvision.transforms import ToTensor,Resize
from model import MyModel

app = FastAPI()

model = MyModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

@app.get('/')
def root():
    return 'welcome to my project - image classification!'

@app.post("/predict_model")
async def predict(file: UploadFile = File(...)):
    #accept image form user
    image_bytes = await file.read()
    #read and conver the same size with train
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224,224))
    class_name = ["Car","Bike"]
    #covert to tensor
    tensor = ToTensor()
    image = tensor(image)
    image = image.unsqueeze(0)
    prediction = model(image)
    prediction = int(torch.argmax(prediction))
    prediction = class_name[prediction]
    return {"predict": prediction}
#python -m uvicorn API:app --reload to run


