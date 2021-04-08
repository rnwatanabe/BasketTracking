from ball_detect_track import *
from plot_tools import *
from feet_detect import *
import skvideo.io

TOPCUT = 320

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


class VideoHandler:
    def __init__(self, pano, video, ball_detector, feet_detector):
        self.M1 = np.load("Rectify1.npy")
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.pano = pano
        self.video = video
        self.kp1, self.des1 = self.sift.compute(pano, self.sift.detect(pano))
        self.feet_detector = feet_detector
        self.ball_detector = ball_detector

    def run_detectors(self):
        writer = skvideo.io.FFmpegWriter("demo.mp4")
        time_index = 0
        while self.video.isOpened():
            ok, frame = self.video.read()
            if not ok:
                break
            else:
                if 0 <= time_index < 20:
                    frame = frame[TOPCUT:, :]
                    M = self.get_homography(frame, self.des1, self.kp1)
                    self.feet_detector.get_players_pos(M, self.M1, frame, time_index, writer)
                    self.ball_detector.map2d = self.feet_detector.map2d_updated
                    self.ball_detector.ball_tracker(M, self.M1, frame, writer)

                    k = cv2.waitKey(1) & 0xff
                    if k == 27:
                        break

            time_index += 1
        self.video.release()
        writer.close()
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
        return M
