###############################################################################
# Makefile Template
# Andrew Tu
#
###############################################################################


###############################################################################
# Important Variables
###############################################################################
SRC_DIR := src
RELEASE_DIR := release
DEBUG_DIR := debug

# Compilers
CXX := g++
CXXFLAGS := -std=c++11

NVCC := nvcc
NVCC_FLAGS := -std=c++11 

RELEASE_FLAGS := -o3
DEBUG_FLAGS := -g3 


#############
# OpenCV Lib
#############
#opencvLocation = /usr/local/opencv
opencvLIB= -L/shared/apps/opencv-3.0.0-beta/INSTALL/lib
opencvINC= -I/shared/apps/opencv-3.0.0-beta/INSTALL/include
OPENCV_LINK =  -lopencv_core \
			   -lopencv_highgui \
			   -lopencv_imgproc \
			   -lopencv_photo \
			   -lopencv_video \
			   -lopencv_imgcodecs \
			   -lopencv_features2d

INCLUDES := -I include $(opencvINC)
LIBS := $(opencvLIB) 
CXX_LIBS := -fopenmp
NVCC_LIBS := -Xcompiler -fopenmp

###############################################################################
# Sources List + Object List Creation
###############################################################################

EXECUTABLE := main

# A list of source files for the executable we want to build
EXEC_SRCS := \
  $(SRC_DIR)/main.cpp \
  $(SRC_DIR)/HarrisCorner.cpp \
  $(SRC_DIR)/utils.cpp


# An self generating list of objects for each of the objects needed in the 
# executable
RELEASE_EXEC_OBJS += $(patsubst $(SRC_DIR)/%.cpp, $(RELEASE_DIR)/%.o, \
	     $(filter $(SRC_DIR)/%.cpp, $(EXEC_SRCS)))
RELEASE_EXEC_OBJS += $(patsubst $(SRC_DIR)/%.cu, $(RELEASE_DIR)/%.o, \
	     $(filter $(SRC_DIR)/%.cu, $(EXEC_SRCS)))

DEBUG_EXEC_OBJS += $(patsubst $(SRC_DIR)/%.cpp, $(DEBUG_DIR)/%.o, \
	     $(filter $(SRC_DIR)/%.cpp, $(EXEC_SRCS)))
DEBUG_EXEC_OBJS += $(patsubst $(SRC_DIR)/%.cu, $(DEBUG_DIR)/%.o, \
	     $(filter $(SRC_DIR)/%.cu, $(EXEC_SRCS)))

.PHONY: \
  all \
  release \
  debug \
  clean 

###############################################################################
# Build Commands 
###############################################################################
all : \
  release

release :\
 $(RELEASE_DIR)/$(EXECUTABLE)

debug :\
  $(DEBUG_DIR)/$(EXECUTABLE)

# The build sequence for the particular executable we want to build. 
$(RELEASE_DIR)/$(EXECUTABLE): $(RELEASE_EXEC_OBJS)
	@echo $(RELEASE_EXEC_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) $(LIBS) $(INCLUDES) $(GENCODE_FLAGS) $(OPENCV_LINK) -link $^ -o $@ 

$(DEBUG_DIR)/$(EXECUTABLE): $(DEBUG_EXEC_OBJS)
	@echo $(DEBUG_EXEC_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_LIBS) $(LIBS) $(INCLUDES) $(GENCODE_FLAGS) $(OPENCV_LINK) -link $^ -o $@ 

clean :
	@rm -rf $(DEBUG_DIR) $(RELEASE_DIR) 

###############################################################################
# Dependencies 
###############################################################################

# Build the .o file for each src file
$(RELEASE_DIR)/%.o : $(SRC_DIR)/%.cpp | release_directory
	@echo $(RELEASE_EXEC_OBJS)
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) $(INCLUDES) $(CXX_LIBS) $(LIBS) -c $< -o $@
$(RELEASE_DIR)/%.o : $(SRC_DIR)/%.cu | release_directory
	@echo $(RELEASE_EXEC_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(NVCC_LIBS) $(LIBS) -c $< -o $@

$(DEBUG_DIR)/%.o : $(SRC_DIR)/%.cpp | debug_directory
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $(INCLUDES) $(CXX_LIBS) $(LIBS) -c $< -o $@
$(DEBUG_DIR)/%.o : $(SRC_DIR)/%.cu | debug_directory
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(NVCC_LIBS) $(LIBS) -c $< -o $@

# Autogen dependencies
$(RELEASE_DIR)/%.d : $(SRC_DIR)/%.cpp 
	@echo "Running Autogen..."
	@set -e;\
	rm -f $@;\
	$(CXX) -MM $(CXXFLAGS) $(INCLUDES) $(CXX_LIBS) $(LIBS) $< > $@.$$$$;\
	sed 's,\($*\)\.o[ :]*,$(RELEASE_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

$(RELEASE_DIR)/%.d : $(SRC_DIR)/%.cu
	@echo "Running Autogen..."
	@set -e;\
	rm -f $@;\
	$(NVCC) -MM $(NVCC_FLAGS) $(NVCC_LIBS) $(LIBS) $< > $@.$$$$;\
	sed 's,\($*\)\.o[ :]*,$(RELEASE_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

$(DEBUG_DIR)/%.d : $(SRC_DIR)/%.cpp
	@echo "Running Autogen..."
	@set -e;\
	rm -f $@;\
	$(CXX) -MM $(CXXFLAGS) $(INCLUDES) $(CXX_LIBS) $(LIBS) $< > $@.$$$$;\
	sed 's,\($*\)\.o[ :]*,$(DEBUG_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

$(DEBUG_DIR)/%.d : $(SRC_DIR)/%.cu
	@echo "Running Autogen..."
	@set -e;\
	rm -f $@;\
	$(NVCC) -MM $(NVCC_FLAGS) $(INCLUDES) $(NVCC_LIBS) $(LIBS) $< > $@.$$$$;\
	sed 's,\($*\)\.o[ :]*,$(DEBUG_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

# Create the build directories
release_directory :
	@echo "Make Release Dir"
	@mkdir -p $(RELEASE_DIR)

debug_directory :
	@mkdir -p $(DEBUG_DIR)

-include $(OBJS:.o=.d)


