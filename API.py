from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root():
    return 'welcome to my project - image classification!'

@app.post("/predict_model")
def deloy():
    return


