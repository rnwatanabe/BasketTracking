import cv2
import numpy as np

from tools.plot_tools import plt_plot

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


def collage(frames, direction=1, plot=False):
    sift = cv2.SIFT_create()

    if direction == 1:
        current_mosaic = frames[0]
    else:
        current_mosaic = frames[-1]

    for i in range(len(frames) - 1):

        # FINDING FEATURES
        kp1 = sift.detect(current_mosaic)
        kp1, des1 = sift.compute(current_mosaic, kp1)
        kp2 = sift.detect(frames[i * direction + direction])
        kp2, des2 = sift.compute(frames[i * direction + direction], kp2)

        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # Finding an homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        result = cv2.warpPerspective(frames[i * direction + direction],
                                     M,
                                     (current_mosaic.shape[1] + frames[i * direction + direction].shape[1],
                                      frames[i * direction + direction].shape[0] + 50))

        result[:current_mosaic.shape[0], :current_mosaic.shape[1]] = current_mosaic
        current_mosaic = result

        # removing black part of the collage
        for j in range(len(current_mosaic[0])):
            if np.sum(current_mosaic[:, j]) == 0:
                current_mosaic = current_mosaic[:, :j - 50]
                break

        if plot:
            plt_plot(current_mosaic)

    return current_mosaic


def add_frame(frame, pano, pano_enhanced, plot=False):
    sift = cv2.xfeatures2d.SIFT_create()  # sift instance

    # FINDING FEATURES
    kp1 = sift.detect(pano)
    kp1, des1 = sift.compute(pano, kp1)
    kp2 = sift.detect(frame)
    kp2, des2 = sift.compute(frame, kp2)

    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    print(f"Number of good correspondences: {len(good)}")
    if len(good) < 70: return pano

    # Finding an homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(frame,
                                 M,
                                 (pano.shape[1],
                                  pano.shape[0]))

    if plot: plt_plot(result, "Warped new image")

    avg_pano = np.where(result < 100, pano_enhanced,
                        np.uint8(np.average(np.array([pano_enhanced, result]), axis=0, weights=[1, 0.7])))

    if plot: plt_plot(avg_pano, "AVG new image")

    return avg_pano


def binarize_erode_dilate(img, thresh=130, plot=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if plot: plt_plot(gray, "Panorama after gray", cmap="gray")
    th, img_otsu = cv2.threshold(gray, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)

    if plot: plt_plot(img_otsu, "Panorama after Otsu", cmap="gray")

    kernel = np.array([[0, 0, 0],
                       [1, 1, 1],
                       [0, 0, 0]], np.uint8)
    img_otsu = cv2.erode(img_otsu, kernel, iterations=20)
    img_otsu = cv2.dilate(img_otsu, kernel, iterations=20)

    if plot: plt_plot(img_otsu, "After Erosion-Dilation", cmap="gray")
    return img_otsu

def encontrar_centros(img, corners,  plot=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = hsv[:,:,1]
    hsv[:,:,2] = hsv[:,:,1]
    th, img_otsu = cv2.threshold(hsv, thresh=60, maxval=255, type=cv2.THRESH_BINARY)
    
    if plot: plt_plot(img_otsu, "Panorama after Thres", cmap="gray")

    kernel = np.array([[0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0]], np.uint8)
    img_otsu = cv2.erode(img_otsu, kernel, iterations=20)
    img_otsu = cv2.dilate(img_otsu, kernel, iterations=20)
    # img_otsu = cv2.erode(img_otsu, kernel, iterations=10)
    
    kernel = np.array([[0, 0, 0],
                       [1, 1, 1],
                       [0, 0, 0]], np.uint8)
    img_otsu = cv2.erode(img_otsu, kernel, iterations=20)
    img_otsu = cv2.dilate(img_otsu, kernel, iterations=20)
    
    centro = corners.mean(axis=0)
    circle_radius = int(0.17*img_otsu.shape[0])
    print(circle_radius)
    cv2.circle(img_otsu, (int(centro[0]), int(centro[1])), circle_radius, (0,0,0), -1)
    if plot: plt_plot(img_otsu, "After Erosion-Dilation", cmap="gray")
    img_otsu[:,:int(img_otsu.shape[1]//2*0.8)] = 0
    img_otsu[:,int(img_otsu.shape[1]//2*1.2):] = 0

    mask = np.zeros(img_otsu.shape, dtype=np.uint8)
    cnts = cv2.findContours(img_otsu[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    areas = np.zeros(len(cnts))
    
    for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])
        areas[i] = area
        
    areas_ind = np.argsort(areas)[-2:]
    contours_court = []
    for i in areas_ind:
        print(areas[i])
        cv2.drawContours(mask, [cnts[i]], -1, (36, 255, 12), -1)
        contours_court.append(cnts[i])
    img_otsu = mask
    
    if plot: plt_plot(img_otsu, "After corner1", cmap="gray")
    hull = []
    for i in range(2):
        hull.append(cv2.convexHull(contours_court[i]))
        cv2.drawContours(img_otsu, [hull[i]], -1, (255,0,0), -1)
    if plot: plt_plot(img_otsu, "After Hull", cmap="gray")
    cantos = []
    for i in range(2):
        x, y, w, h = cv2.boundingRect(hull[i])   
        corners = np.zeros((4,2))
        corners[0] = np.array([x, y+h])
        corners[1] = np.array([x, y])
        corners[2] = np.array([x+w, y])
        corners[3] = np.array([x+w, y+h])
        approx = corners.copy()
        cantos.append(order_corners(corners))
        print(approx)
        cv2.rectangle(img_otsu, (x,y),(x+w, y+h), 100, 5)
    
    if plot: plt_plot(img_otsu, "After Rectangle Fit", cmap="gray")
    centro0 = cantos[0].mean(axis=0)
    centro1 = cantos[1].mean(axis=0)
    centros = []
    if centro0[0] < centro1[0]:
        centros.append((cantos[0][2]+cantos[1][1])/2)
        centros.append((cantos[0][3]+cantos[1][0])/2)
    else:
        centros.append((cantos[0][1]+cantos[1][2])/2)
        centros.append((cantos[0][0]+cantos[1][3])/2)
    centros = np.array(centros).astype(int)
    cv2.circle(img, (centros[0][0], centros[0][1]), 15, (255,0,0),-1)
    cv2.circle(img, (centros[1][0], centros[1][1]), 15, (255,0,0),-1)
    if plot: plt_plot(img, "Centres", cmap="gray")
    
    return centros

def order_corners(corners):
    center = np.mean(corners, axis=0)
    cantos = np.zeros((4,2))
    for i in range(4):
        if corners[i][0]<center[0] and corners[i][1]<center[1]:
            tl_court = corners[i]
        if corners[i][0]<center[0] and corners[i][1]>center[1]:
            bl_court = corners[i]
        if corners[i][0]>center[0] and corners[i][1]<center[1]:
            tr_court = corners[i]
        if corners[i][0]>center[0] and corners[i][1]>center[1]:
            br_court = corners[i]
    cantos[0] = bl_court
    cantos[1] = tl_court
    cantos[2] = tr_court
    cantos[3] = br_court
    return cantos


def rectangularize_court(pano, plot=False):
    # BLOB FILTERING & BLOB DETECTION

    # adding a little frame to enable detection
    # of blobs that touch the borders
    pano[-4: -1] = pano[0:3] = 0
    pano[:, 0:3] = pano[:, -4:-1] = 0

    mask = np.zeros(pano.shape, dtype=np.uint8)
    cnts = cv2.findContours(pano, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_court = []

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    threshold_area = 100000
    for c in cnts:
        area = cv2.contourArea(c)
        if area > threshold_area:
            cv2.drawContours(mask, [c], -1, (36, 255, 12), -1)
            contours_court.append(c)

    pano = mask
    if plot: plt_plot(pano, "After Blob Detection", cmap="gray")

    # pano = 255 - pano
    contours_court = contours_court[0]
    simple_court = np.zeros(pano.shape)

    # convex hull
    hull = cv2.convexHull(contours_court)
    cv2.drawContours(pano, [hull], 0, 100, 2)
    if plot: plt_plot(pano, "After ConvexHull", cmap="gray",
                      additional_points=hull.reshape((-1, 2)))

    # fitting a poly to the hull
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    corners = approx.reshape(-1, 2)
    approx = corners.copy()
    corners = order_corners(corners)
    cv2.drawContours(pano, [approx], 0, 100, 5)
    cv2.drawContours(simple_court, [approx], 0, 255, 3)

    if plot:
        plt_plot(pano, "After Rectangular Fitting", cmap="gray")
        plt_plot(simple_court, "Rectangularized Court", cmap="gray")
        print("simplified contour has", len(approx), "points")

    return simple_court, corners


def homography(rect, image, plot=False):
    bl, tl, tr, br = rect
    rect = np.array([tl, tr, br, bl], dtype="float32")

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB)) + 700

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    if plot: plt_plot(warped)
    return warped, M


def rectify(pano_enhanced, corners, plot=False, game_path='.'):
    # TODO: adapt this in a way that works in any setting

    centros = encontrar_centros(pano_enhanced, corners, plot=plot)
    center_line = int((centros[0][0]+centros[1][0])/2)
    panoL = pano_enhanced[:, :center_line]
    panoR = pano_enhanced[:, center_line:]
    cornersL = np.array([corners[0], corners[1], centros[0], centros[1]])
    cornersR = np.array(
        [[centros[1][0] - center_line, centros[1][1]],
         [centros[0][0] - center_line, centros[0][1]],
         [corners[2][0] - center_line, corners[2][1]],
         [corners[3][0] - center_line, corners[3][1]]
         ])
    
    h, M = homography(corners, pano_enhanced)
    np.save(game_path+'/Rectify1.npy', M)
    h1, ML = homography(cornersL, panoL)
    np.save(game_path+'/RectifyL.npy', ML)

    h2, MR = homography(cornersR, panoR)
    np.save(game_path+'/RectifyR.npy', MR)

    # rectified = np.hstack((h1, cv2.resize(h2, (int((h2.shape[0] / h1.shape[0]) * h1.shape[1]), h1.shape[0]))))
    rectified = np.hstack((h1, cv2.resize(h2, (h1.shape[1], h1.shape[0]))))
    cv2.imwrite(game_path+"/rectified.png", rectified)
    if plot: plt_plot(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
    return rectified

