from threading import Thread
import VideoPlay

class ProcessThread(Thread):
    def __init__(self, tm):
        Thread.__init__(self)
        self.tm = tm

    def run(self):
        while self.tm.hasFiles():
            currFile = self.tm.getFile()
            VideoPlay.main(currFile)


class ThreadMeister():
    def __init__(self, numThreads, files):
        self.numThreads = numThreads
        self.threads = []
        self.fileIndex = 0
        self.max = len(files)
        self.files = files
        self.isLocked = False

    def hasFiles(self):
        return self.fileIndex < self.max

    def getFile(self):
        while self.isLocked:
            pass
        
        self.isLocked = True
        toReturn = self.files[self.fileIndex]
        self.fileIndex += 1

        self.isLocked = False
        print("Now processing file " + str(self.fileIndex) + " of " + str(self.max))
        return toReturn

    def start(self):
        for i in range(self.numThreads):
            self.threads.append(ProcessThread(self))
        for thread in self.threads:
            thread.start()

with open("videos/movList.txt", "r") as f:
    files = [x.strip() for x in f.readlines()]
    t = ThreadMeister(20, files)
    t.start() #pray for this to work
