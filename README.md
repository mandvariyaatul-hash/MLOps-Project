# MLOps-Project

- [Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

### Pre-requisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [AWS CLI configured](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

### How to run projects
1. Clone project and run below commnads
```
pip install -r requirements.txt
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
python main.py

2. Create Docker file
docker build --no-cache --platform linux/arm64 -t diabetes-mlops:v1 . 

3. Run docker file
docker run -d \
  -p 8000:8000 \
  -p 5000:5000 \
  -v "$(pwd)/drift_reports:/app/drift_reports" \
  -v "$(pwd)/prediction_logs.csv:/app/prediction_logs.csv" \
  --name diabetes_app diabetes-mlops:v1


### Troubleshoot

If any port is open, close it by using below command.
```
lsof -ti:5000 | xargs kill -9 2>/dev/null
```

### Use it with Public IP
ec2-3-84-213-161.compute-1.amazonaws.com
