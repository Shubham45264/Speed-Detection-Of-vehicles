import cv2
import math

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.f = {}
        self.sp = {}
        self.data = []
        self.frame_points = []
        self.limit_val = 50
        self.prev_frame_centers = {} #store center point of previous frame

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.f[self.id_count-1] = 0
                self.sp[self.id_count-1] = 0

            #speed calculation
            for id in self.center_points.keys():
                if id in self.prev_frame_centers:
                    dist = math.hypot(self.center_points[id][0] - self.prev_frame_centers[id][0], self.center_points[id][1] - self.prev_frame_centers[id][1])
                    self.sp[id] = round(dist, 2)
                else:
                    self.sp[id] = 0

        self.prev_frame_centers = self.center_points.copy()

        new_objects_bbs_ids = []
        for obj_bb_id in objects_bbs_ids:
            if obj_bb_id[4] in self.center_points:
                new_objects_bbs_ids.append(obj_bb_id)

        return new_objects_bbs_ids

    def getsp(self, id):
        if id in self.sp:
            return self.sp[id]
        else:
            return 0

    def limit(self):
        return self.limit_val

    def capture(self, frame, x, y, h, w, s, id):
        cv2.imwrite(f"vehicle_{id}.jpg", frame[y:y+h,x:x+w])

    def dataset(self):
        return list(self.center_points.keys()), list(self.sp.values())

    def end(self):
        print("Tracking Ended")

    def datavis(self, ids, spds):
        print("Data Visualization:")
        for i in range(len(ids)):
            print(f"ID: {ids[i]}, Speed: {spds[i]}")