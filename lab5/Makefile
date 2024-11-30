CXX        = nvcc
LD         = $(CXX)

LIBS_PATH = -L/usr/lib64/mpich/lib
LIBS =  -lmpi -lopa -lmpl -lrt -lpthread
INCLUDE_PATH = -I/usr/include/mpich-x86_64/
FLAGS = -g
TARGET = "/home/sloniki/mephi_gst-main/lab5/compute.cu"
OBIN = "/home/sloniki/mephi_gst-main/lab5/main"

all: $(TARGET)

$(TARGET):
	$(LD) $(INCLUDE_PATH) $(FLAGS) $(TARGET) -o $(OBIN) $(LIBS_PATH) $(LIBS)