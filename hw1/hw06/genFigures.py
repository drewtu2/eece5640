import subprocess
import pickle
import statistics
#import matplotlib.pyplot as plt

def runTest(testName, numProcs, dataFile, outputFile):
    procRunTimes = []
    timingFile = str(numProcs) + "results.result"
    numSamples = "5000000"

    # If Linux
    timeCmd = "/usr/bin/time"
    # If MAC - Note: c code doesn't run on Mac...
    #timeCmd = "/usr/local/bin/gtime"

    for x in range(0, 3):

        # Generate New Data...
        genData = subprocess.Popen(["./tools/gen-input", numSamples, dataFile], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)  
         
        print("Generated new data")
        output,err = genData.communicate()
        print("Gen Data STDOUT: " + output.decode('ascii'))
        print("Gen Data STDERR: " + err.decode('ascii'))

        # Run command and time it...
        print("Running: " + testName + " with " + str(numProcs) + " procs...")
        p1 = subprocess.Popen([timeCmd, "-f", "%e", "-o", timingFile,
                "-a", testName, numProcs, dataFile, outputFile], stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE)  
        output,err = p1.communicate()
        print("Cmd STDOUT: " + output.decode('ascii'))
        print("Cmd STDERR: " + err.decode('ascii'))

        # Print status...
        print("Finished run " + str(x) + " of " + numProcs);

        
    with open(timingFile) as f:
        content = f.readlines()
        for line in content:
            try:
                procRunTimes.append(float(line.rstrip()))
            except ValueError as e:
                print("error ", e, " with ", line)
        return statistics.median(procRunTimes);        

def runAllTests():
    timings = {};
    timings[1] = (runTest("./tssort", "1", "test.data", "output.data"))
    timings[2] = (runTest("./tssort", "2", "test.data", "output.data"))
    timings[4] = (runTest("./tssort", "4", "test.data", "output.data"))
    timings[8] = (runTest("./tssort", "8", "test.data", "output.data"))
    print(timings)

    pickle.dump( timings, open( "save.p", "wb" ) )

def generateGraph():
    timings = pickle.load(open("save.p", "rb"))
    x = [1, 2, 4, 8]
    y = [timings[1], timings[2], timings[4], timings[8]]
    plt.bar(x, y)
    plt.title("Effect of Threading on Sorting 50 Million Numbers Using Sample Sort")
    plt.ylabel("Time (s)")
    plt.xlabel("Number of threads")
    plt.show()

#generateGraph()

runAllTests();

