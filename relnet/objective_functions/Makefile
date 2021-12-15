# location of the Python header files
PYTHON_INCLUDE = /opt/conda/envs/ucfadar-relnet/include/python3.6m

# location of the Boost Python include files and library
BOOST_INC = /usr/include/boost
BOOST_LIB = /usr/lib/x86_64-linux-gnu

# compile mesh classes
TARGET = objective_functions_ext

$(TARGET).so: $(TARGET).o
	g++ -shared -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python3-py36 -lboost_numpy3-py36 -lboost_system -o $(TARGET).so

$(TARGET).o: $(TARGET).cpp
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -O3 -c $(TARGET).cpp

clean:
	rm -f *.o
	rm -f *.so
	rm -f core.*
	rm -rf __pycache__