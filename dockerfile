From okwrtdsh/anaconda3:latest
RUN pip install msgpack
RUN pip install pydotplus
RUN apt update -y
RUN apt-get update -y
RUN apt install graphviz -y
RUN pip install graphviz
RUN apt-get install spyder3