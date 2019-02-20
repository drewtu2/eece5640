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

RELEASE_FLAGS := -o3
DEBUG_FLAGS := -g3

INCLUDES := -I include
LIBS := -lpthread

###############################################################################
# Sources List + Object List Creation
###############################################################################

EXECUTABLE := dining_philosophers

# A list of source files for the executable we want to build
EXEC_SRCS := \
  $(SRC_DIR)/main.cpp\
  $(SRC_DIR)/Philosopher.cpp\
  $(SRC_DIR)/SmartPhilosopher.cpp\
  $(SRC_DIR)/SmartPhilosopherMiddleFork.cpp\
  $(SRC_DIR)/table.cpp

# An self generating list of objects for each of the objects needed in the 
# executable
RELEASE_EXEC_OBJS += $(patsubst $(SRC_DIR)/%.cpp, $(RELEASE_DIR)/%.o, \
	     $(filter $(SRC_DIR)/%.cpp, $(EXEC_SRCS)))
DEBUG_EXEC_OBJS += $(patsubst $(SRC_DIR)/%.cpp, $(DEBUG_DIR)/%.o, \
	     $(filter $(SRC_DIR)/%.cpp, $(EXEC_SRCS)))

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
#  debug  

release :\
 $(RELEASE_DIR)/$(EXECUTABLE)

debug :\
  $(DEBUG_DIR)/$(EXECUTABLE)

# The build sequence for the particular executable we want to build. 
$(RELEASE_DIR)/$(EXECUTABLE): $(RELEASE_EXEC_OBJS)
	@echo $(RELEASE_EXEC_OBJS)
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) $(INCLUDES) $(LIBS) -o $@ $^

$(DEBUG_DIR)/$(EXECUTABLE): $(DEBUG_EXEC_OBJS)
	@echo $(DEBUG_EXEC_OBJS)
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $(INCLUDES) $(LIBS) -o $@ $^

clean :
	@rm -rf $(DEBUG_DIR) $(RELEASE_DIR) 

###############################################################################
# Dependencies 
###############################################################################

# Build the .o file for each src file
$(RELEASE_DIR)/%.o : $(SRC_DIR)/%.cpp | release_directory
	@echo $(RELEASE_EXEC_OBJS)
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) $(INCLUDES) -c $< -o $@

$(DEBUG_DIR)/%.o : $(SRC_DIR)/%.cpp | debug_directory
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) $(INCLUDES) -c $< -o $@

# Autogen dependencies
$(RELEASE_DIR)/%.d : $(SRC_DIR)/%.cpp
	@echo "Running Autogen..."
	@set -e;\
	rm -f $@;\
	$(CXX) -MM $(CXXFLAGS) $(INCLUDES) $< > $@.$$$$;\
	sed 's,\($*\)\.o[ :]*,$(RELEASE_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

$(DEBUG_DIR)/%.d : $(SRC_DIR)/%.cpp
	@echo "Running Autogen..."
	@set -e;\
	rm -f $@;\
	$(CXX) -MM $(CXXFLAGS) $(INCLUDES) $< > $@.$$$$;\
	sed 's,\($*\)\.o[ :]*,$(DEBUG_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

# Create the build directories
release_directory :
	@echo "Make Release Dir"
	@mkdir -p $(RELEASE_DIR)

debug_directory :
	@mkdir -p $(DEBUG_DIR)

-include $(OBJS:.o=.d)


