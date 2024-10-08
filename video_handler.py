from player_detection import *
import skvideo.io
from datetime import datetime

TOPCUT = 320

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)



class VideoHandler:
    def __init__(self, pano, video, map_2d, game_path, start, topcut, width, height):#ball_detector, feet_detector, map_2d, game_path, start, topcut, width, height):
        start_time=datetime.strptime(start, '%H:%M:%S')
        origin = '00:00:00'
        origin_time=datetime.strptime(origin,"%H:%M:%S") #origin 
        fps = video.get(cv2.CAP_PROP_FPS)
        start_trim_frame=fps*(start_time-origin_time).total_seconds()  #get the start frame
        self.M1 = np.load(game_path+"/Rectify1.npy")
        self.sift = cv2.SIFT_create()
        self.pano = pano
        self.video = video
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_trim_frame-1)
        self.kp1, self.des1 = self.sift.compute(pano, self.sift.detect(pano))
        # self.feet_detector = feet_detector
        # self.ball_detector = ball_detector
        self.map_2d = map_2d
        self.game_path = game_path
        self.topcut = topcut
        self.width = width
        self.height = height

    def run_detectors(self):
        # writer = skvideo.io.FFmpegWriter(self.game_path+"/demo2.mp4")
        time_index = 0
        while self.video.isOpened():
            ok, frame = self.video.read()
            if not ok:
                break
            else:
                if 0 <= time_index <= 1:

                    # print("\r Computing DEMO: " + str(int(100 * time_index / 200)) + "%",
                    #       flush=True, end='')

                    frame = frame[self.topcut:, :]
                    M, src_pts = self.get_homography(frame, self.des1, self.kp1)
                    for i in range(len(src_pts)):
                        cv2.circle(frame, (int(src_pts[i][0][0]), int(src_pts[i][0][1])), 5, (255,0,0),-1)
                    # frame, self.map_2d, map_2d_text = self.feet_detector.get_players_pos(M, self.M1, frame, time_index,
                    #                                                                      self.map_2d)
                    # frame, ball_map_2d = self.ball_detector.ball_tracker(M, self.M1, frame, self.map_2d.copy(),
                    #                                                      map_2d_text, time_index)
                    # vis = np.vstack((frame, cv2.resize(map_2d_text, (frame.shape[1], frame.shape[1] // 2))))

                    # cv2.imshow("Tracking", vis)
                    # plt_plot(vis)
                    warped = cv2.warpPerspective(frame, M, (self.width, self.height))
                    cv2.imshow("Frame", warped)  #display frame
                    # writer.writeFrame(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

                    k = cv2.waitKey(1) & 0xff
                    if k == 27:
                        break
            time_index += 1
        self.video.release()
        # writer.close()
        cv2.destroyAllWindows()

    def get_homography(self, frame, des1, kp1):
        kp2 = self.sift.detect(frame)
        kp2, des2 = self.sift.compute(frame, kp2)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return M, src_pts
