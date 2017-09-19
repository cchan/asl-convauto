import numpy as np
import cv2, os


threshold = 55
skin_boundaries = [
        ([75,40,60], [216, 231, 255])
    ]

d = dict()

def getCat(cat):
    return d[cat]

def removeBackground(frame):
    mask = model.apply(frame)
    n = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, n, iterations=1)
    return cv2.bitwise_and(frame, frame, mask=mask)

def blur(frame, n):
    return cv2.GaussianBlur(frame, (n, n), 0)

def detectEdges(frame):
    x = cv2.Sobel(frame, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(frame, cv2.CV_16S, 0, 1)
    sobel = np.hypot(x, y)
    sobel[sobel > 255] = 255
    return sobel

def contours(frame, edge):
    i, c, h = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    l1 = []
    for i, t in enumerate(h[0]):
        if t[3] == -1:
            t = np.insert(t, 0, [i])
            l1.append(t)

    realContours = []
    minSize = edge.size * 0.02
    for t in l1:
        contour = c[t[0]]
        area = cv2.contourArea(contour)
        if area > minSize:
            realContours.append([contour, area])
#            cv2.drawContours(frame, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)
    return [i[0] for i in realContours]

def removeFace(frame):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    tuples = []
    for (x,y,w,h) in faces:
        tuples.append((x,y,w,h))
    
    return tuples

def convexHull(contour, frame):
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return

    for i in xrange(defects.shape[0]):
	s,e,f,d = defects[i,0]
	start = tuple(contour[s][0])
	end = tuple(contour[e][0])
	far = tuple(contour[f][0])
        if d > 450:
            pass
            #cv2.line(frame,start,end,[0,255,0],2)
    	    #cv2.circle(frame,far,5,[0,0,255],-1)

def main(url):
    global d
    #with open("WordList.txt", "r") as f:
    #    for i in f.readlines():
    #        k, v = i.strip().split(" ")
    #        d[k] = v

    
    cap = cv2.VideoCapture('videos/' + url)
    frames = []
    #r = os.system("mkdir" + " 'convauto/alex/data/" + getCat(url) + "'")
    #if (r != 0):
    #    print("error making directory for " + getCat(url))
    #    return
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            frame = frame[:300]
        else:
            break
        
        #cv2.imshow('frame', frame)

        #excluded = removeFace(frame)


        for lower, upper in skin_boundaries:
            lower = np.array(lower, np.uint8)
            upper = np.array(upper, np.uint8)
                
            mask = cv2.inRange(frame, lower, upper)
            frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        blurred = blur(frame, 11)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(grey)

        edge = np.max(np.array([detectEdges(blurred[:,:, 0]), detectEdges(blurred[:,:, 1]), detectEdges(blurred[:,:, 2])]), axis=0)
        mean = np.mean(edge)
        edge[edge <= mean] = 0

        e_8u = np.asarray(edge, np.uint8)
        realContours = contours(frame, e_8u)

        for i in realContours:
            convexHull(i, frame)

        mask = edge.copy()
        mask[mask > 0] = 0
        cv2.fillPoly(mask, realContours, 255)
        mask = np.logical_not(mask)
        
        frame[mask] = 0

        cv2.fastNlMeansDenoising(frame, frame, 5)
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #print("~/convauto/alex/data/" + getCat(url) + "/" + url[:url.find(".mov")] + ".bmp")
        #cv2.imwrite("~/convauto/alex/data/" + getCat(url) + "/" + url[:url.find(".mov")] + ".bmp", frame)

        #with open("csv/" + url[:url.index(".")] + ".csv", "a+") as f:
         #   np.savetxt(f, frame, delimiter=",", fmt="%d")
         #   f.write("\n")

    cap.release()

