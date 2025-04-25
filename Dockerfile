FROM python:3.9-slim

RUN pip install pandas scikit-learn matplotlib joblib

COPY . .
CMD ["python","app.py"]
