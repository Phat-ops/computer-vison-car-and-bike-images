FROM python:3.13
LABEL tac_gia="phat06"

WORKDIR /phat06

COPY model.py /phat06/model.py
COPY model.pkl /phat06/model.pkl
COPY API.py /phat06/API.py
COPY requirements.txt /phat06/requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8888

CMD ["uvicorn", "API:app","--host","0.0.0.0","--port","8888" ]