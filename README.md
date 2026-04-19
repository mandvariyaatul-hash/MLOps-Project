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


### Troubleshoot

If any port is open, close it by using below command.
```
lsof -ti:5000 | xargs kill -9 2>/dev/null
```