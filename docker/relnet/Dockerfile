FROM victordarvariu/rabbitconda
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y build-essential cmake sqlite3 libsqlite3-dev zip unzip \
    libboost-dev libboost-system-dev libboost-thread-dev libboost-serialization-dev \
    libboost-python-dev libboost-regex-dev libboost-numpy-dev

RUN apt-get update && apt-get install -y graphviz libgraphviz-dev graphviz-dev pkg-config

WORKDIR /relnet
ADD ./environment.yml /relnet/environment.yml
RUN conda env update --file environment.yml

ARG RDKIT_VERSION=Release_2017_09_2

RUN cd /usr/lib/ \
    && wget --quiet https://github.com/rdkit/rdkit/archive/$RDKIT_VERSION.tar.gz \
    && tar -xzvf $RDKIT_VERSION.tar.gz -C ./ \
    && rm -rf $RDKIT_VERSION.tar.gz

ENV RDBASE=/usr/lib/rdkit-$RDKIT_VERSION
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RDBASE:$RDBASE/lib:/usr/lib/x86_64-linux-gnu

RUN cd $RDBASE \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j4 \
    && make install

RUN chmod -R 755 $RDBASE

RUN cd /usr/lib \
    && git clone https://github.com/VictorDarvariu/pytorch_structure2vec/ \
    && cd pytorch_structure2vec \
    && cd s2v_lib \
    && make -j4

RUN chmod -R 777 /usr/lib/pytorch_structure2vec

ENV PYTHONPATH="/usr/lib/pytorch_structure2vec/s2v_lib:${PYTHONPATH}"

CMD ["/bin/bash"]