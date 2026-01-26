import numpy as np
import open3d as o3d
import cv2

import utils
from glassesTools import drawing, ocv, pose, transforms
from glassesTools.validation.dynamic import _get_position

class Detector:
    def __init__(self,
                 plane_distance_cm,
                 fiducial_phis_deg,
                 fiducial_rhos,
                 target_locations,
                 min_radius_threshold   = 5,    # pixels
                 min_radius_hough       = 3,    # pixels
                 max_radius             = 40,   # pixels
                 edge_cut_fac           = .1,
                 blackout_rect: tuple[int,int,int,int]|None = None
                 ):
        fiducials_array = [utils.pol2cart(rho, phi/180*np.pi) for rho,phi in zip(fiducial_rhos,fiducial_phis_deg)]
        # convert to cm (position in world)
        fiducials_array = [_get_position(p, plane_distance_cm, 'deg') for p in fiducials_array]
        self.world_points = np.hstack((np.array(fiducials_array), np.zeros((len(fiducials_array),1))))
        self.world_points[:,1] *= -1    # NB: PsychoPy's y axis is reversed
        self.world_points_pc = o3d.geometry.PointCloud()
        self.world_points_pc.points = o3d.utility.Vector3dVector(self.world_points)

        self.targets = {p:list(_get_position(target_locations[p], plane_distance_cm, 'deg')) for p in target_locations}

        # some detection config and state variables
        self.min_radius_threshold = min_radius_threshold
        self.min_radius_hough     = min_radius_hough
        self.max_radius           = max_radius
        self.edge_cut_fac         = edge_cut_fac
        self.blackout_rect        = blackout_rect

        self.initial_Tform = None
        self.initial_Tform_for_init = None
        self.reg_type = o3d.pipelines.registration.TransformationEstimationPointToPoint(True)
        self.n_frame_bad_homography = 0

        self._det_cache: tuple[int,tuple|None] = None

    def detect_plane(self, plane_name: str, frame_idx: int, frame: np.ndarray, camera_parameters: ocv.CameraParams) -> tuple[np.ndarray, np.ndarray]:
        _,_,_,iX,iY,correspondence,_ = self._get_detector_cache(frame_idx, frame, camera_parameters)
        if correspondence is None:
            return None, None
        # return image points and matching world points
        return self._get_plane_matching_image_points(iX, iY, correspondence, as_mm=True)

    def detect_target(self, fun_name: str, frame_idx: int, frame: np.ndarray, camera_parameters: ocv.CameraParams) -> dict[int,list]:
        target_locations = self._get_detector_cache(frame_idx, frame, camera_parameters)[-1]
        if target_locations is None:
            return None
        return {t_id:target_locations[t_id][:2]+self.targets[t_id] for t_id in target_locations}

    def detect_bright_frame(self, fun_name: str, frame_idx: int, frame: np.ndarray, camera_parameters: ocv.CameraParams) -> bool:
        return self._get_detector_cache(frame_idx, frame, camera_parameters)[0]

    def visualize_plane(self, plane_name: str, frame_idx: int, frame: np.ndarray, plane_points: np.ndarray):
        _,contours,_,iX,iY,correspondence,_ = self._get_detector_cache(frame_idx, None, None)

        if correspondence is None:
            if contours is None:
                return
            # just draw all contours
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
            # and detected circle centers
            for x,y in zip(iX, iY):
                drawing.openCVCircle(frame, (x,y), 1, (255, 0, 0), 1, 8)
            return

        # we've got correspondence. Draw only those identified as part of the fiducials
        cv2.drawContours(frame, [contours[c[0]] for c in correspondence], -1, (0, 255, 0), 1)
        for c in correspondence:
            drawing.openCVCircle(frame, (iX[c[0]], iY[c[0]]), 1, (255, 0, 0), 1, 8)
            cv2.putText(frame, str(c[1]), (int(iX[c[0]]), int(iY[c[0]])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, lineType=cv2.LINE_AA)

    def visualize_target(self, fun_name: str, frame: np.ndarray, frame_idx: int, _: dict[int,list]):
        target_locations = self._get_detector_cache(frame_idx, None, None)[-1]
        if target_locations is None:
            return
        for t_id in target_locations:
            cv2.drawContours(frame, target_locations[t_id][2], -1, (255, 0, 0), 1)
            pos = (target_locations[t_id][0], target_locations[t_id][1])
            drawing.openCVCircle(frame, pos, 2, (0, 0, 255), 1, 8)
            cv2.putText(frame, str(t_id), (int(pos[0]), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    def visualize_bright_frame(self, fun_name: str, frame: np.ndarray, frame_idx: int, is_bright_frame: bool):
        # identify frame as bright or dark
        cv2.putText(frame, 'bright' if is_bright_frame else 'dark', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, lineType=cv2.LINE_AA)

    def _get_detector_cache(self, frame_idx: int, frame: np.ndarray, camera_parameters: ocv.CameraParams):
        if self._det_cache is None or self._det_cache[0]!=frame_idx:
            if frame is None:
                return None
            self._det_cache = (frame_idx, self._run_detection_and_matching(frame, camera_parameters))
        return self._det_cache[1]

    def _get_plane_matching_image_points(self, iX, iY, correspondence, as_mm=True):
        # return image points and matching world points
        objP = np.zeros((correspondence.shape[0],1,3),'float')
        imgP = np.zeros((correspondence.shape[0],1,2),'float')
        fac = 10 if as_mm else 1
        for i,c in enumerate(correspondence):
            objP[i,0,:] = self.world_points[c[1],:]*fac     # cm -> mm, if wanted
            imgP[i,0,:] = [iX[c[0]], iY[c[0]]]
        return objP, imgP

    def _run_detection_and_matching(self, frame: np.ndarray, camera_parameters: ocv.CameraParams):
        # get points in image
        is_bright_frame, contours, areas, cX, cY = self._detect_points_for_frame(frame)
        if not contours:
            return is_bright_frame, None, None, None, None, None, None

        # do point registration
        correspondence = None
        target_locations = None
        if (len(cX)>2 and self.initial_Tform is not None) or len(cX)>5:
            impoints = np.hstack((np.array(cX).reshape((-1,1)),np.array(cY).reshape((-1,1)),np.zeros((len(cX),1))))
            image_pc = o3d.geometry.PointCloud()
            # remove probable target from point set
            has_target = areas[0]/3>np.mean(areas[1:])
            image_pc.points = o3d.utility.Vector3dVector(np.delete(impoints,0,0) if has_target else impoints)
            if self.n_frame_bad_homography>4:
                # too many bad homographies: may have converged on wrong registration, so force a clean restart
                self.initial_Tform = None
                self.initial_Tform_for_init = None
                self.n_frame_bad_homography = 0
            if self.initial_Tform is None:
                # initial estimate with larger point-point distance tolerance, to get the process going. iterate to smaller tolerances
                skip_i1 = False
                for i,tol in enumerate([5000, 5000, 500, 100]):
                    if i==1 and skip_i1:
                        # already worked, skip
                        continue
                    if i<=1:
                        # initialize only for the first iteration. Start from scratch if previous didn't converge
                        if self.initial_Tform_for_init is None or i==1:
                            self.initial_Tform = np.identity(4)
                            self.initial_Tform[2,2] = np.mean(image_pc.get_axis_aligned_bounding_box().get_extent()[:2]/self.world_points_pc.get_axis_aligned_bounding_box().get_extent()[:2])
                            self.initial_Tform[:2,3]= (self.world_points_pc.get_center()-image_pc.get_center())[:2]
                            skip_i1 = i==0  # don't have to try the same thing again
                        else:
                            self.initial_Tform = self.initial_Tform_for_init.copy()
                    # estimate
                    result = o3d.pipelines.registration.registration_icp(image_pc, self.world_points_pc, tol, self.initial_Tform, self.reg_type, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
                    if len(result.correspondence_set)>4:
                        self.initial_Tform = result.transformation.copy()
                        skip_i1 = True
                    elif i>0 or self.initial_Tform_for_init is None:
                        self.initial_Tform = None
                        break
            # run with small tolerance, for robustness to outlier detections and frame-to-frame variations in which points are detected
            if self.initial_Tform is not None:
                result = o3d.pipelines.registration.registration_icp(image_pc, self.world_points_pc, 1, self.initial_Tform, self.reg_type)
                failed = True
                if len(result.correspondence_set)>0:
                    # success
                    correspondence = np.asarray(result.correspondence_set)
                    if failed:=(len(set(correspondence[:,1]))<correspondence.shape[0]):
                        correspondence = None
                    elif has_target:
                        correspondence[:,0]+=1  # first image point, target, was not submitted to point registration routine, so all its indices are off-by-one
                if len(result.correspondence_set)<self.world_points.shape[0]/2 or failed:
                    # not good enough, try to init again
                    self.initial_Tform_for_init = None if failed else self.initial_Tform.copy()
                    self.initial_Tform = None
                else:
                    self.initial_Tform = result.transformation.copy()

            # get target locations
            if correspondence is not None and has_target:
                # get homography
                objP, imgP = self._get_plane_matching_image_points(cX, cY, correspondence, as_mm=False)
                n_points, H = pose.estimate_homography(objP, imgP, camera_parameters)
                if n_points:
                    # use homography to get target locations on image
                    out = transforms.apply_homography(np.array([self.targets[t] for t in self.targets]), np.linalg.inv(H))
                    if camera_parameters is not None and camera_parameters.has_intrinsics():
                        out = transforms.distort_points(out, camera_parameters)
                    # if we have a potential target (much larger first circle)
                    # check if any of these three locations falls within it
                    radius = np.sqrt(areas[0]/np.pi)
                    i_target = np.where(np.hypot(out[:,0]-cX[0],out[:,1]-cY[0])<radius)[0]
                    if i_target.size==1:
                        target_locations = {list(self.targets.keys())[int(i_target)]: [cX[0],cY[0],contours[0]]}
                else:
                    self.n_frame_bad_homography += 1
        return is_bright_frame, contours, areas, cX, cY, correspondence, target_locations

    def _detect_points_for_frame(self, frame: np.ndarray):
        cols = frame.shape[1]
        edge_cut_h = int(cols*self.edge_cut_fac)
        rows = frame.shape[0]
        edge_cut_v = int(rows*self.edge_cut_fac)
        offset = np.array([edge_cut_h, edge_cut_v]).reshape((1,1,2))

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.medianBlur(gray_frame, 5)
        gray_frame = gray_frame[edge_cut_v:(rows-edge_cut_v), edge_cut_h:(cols-edge_cut_h)]

        # determine if image is mostly bright or mostly dark, and choose appropriate thresholding function
        _, thresholded_frame = cv2.threshold(gray_frame, 80, 1, cv2.THRESH_BINARY)  # 80 as threshold for bright/dark seems to work on, since for some systems the camera auto exposure turns the bright image kinda dark
        is_bright_frame = bool(thresholded_frame.sum() > np.prod(gray_frame.shape)/4)
        thresh_func = cv2.THRESH_BINARY_INV if is_bright_frame else cv2.THRESH_BINARY

        # if requested, black out part of the image
        if self.blackout_rect is not None:
            x,y,w,h = self.blackout_rect
            gray_frame[y:y+h, x:x+w] = 127  # mid-gray to avoid messing up adaptive thresholding

        # get circularish contours based on thresholding
        contours,areas,cX,cY,_ = _get_contours(gray_frame, thresh_func, 33, np.pi*self.min_radius_threshold**2, np.pi*self.max_radius**2)

        # also get circles based on Hough transform. This and the thresholding-based method have different false alarms but same hits
        # Hough by itself with some spatial and filtering would work well, but, its center determination is much less precise than
        # thresholding. So use Hough transform output to filter false alarms from thresholding output
        circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1, 20,
                                param1=100, param2=15,
                                minRadius=self.min_radius_hough, maxRadius=self.max_radius)
        if circles is not None:
            contours1,areas1,cX1,cY1 = contours,areas,cX,cY
            contours,areas,cX,cY = [],[],[],[]
            for con,a,x,y in zip(contours1,areas1,cX1,cY1):
                # check if an object of similar size is found at about the same location in both sets. Then keep the thresholding one
                if np.any(np.logical_and(np.hypot(circles[0,:,0]-x,circles[0,:,1]-y)<12, np.abs((2*np.pow(circles[0,:,2],2)-a)/a)<=.8)):
                    contours.append(con)
                    areas.append(a)
                    cX.append(x)
                    cY.append(y)

        # prune the result set: if there are objects with almost the same midpoint,
        # keep the smallest one
        # this is needed because the adaptive thresholding causes a halo around points that have a different
        # contrast from a uniform background (AKA, our targets), which is also detected. Ignore these.
        to_remove = set()
        for a,x,y in zip(areas,cX,cY):
            same_pos = [np.hypot(x-x2,y-y2)<6 for x2,y2 in zip(cX,cY)]
            if sum(same_pos)>1:
                idxs = np.where(same_pos)[0]
                s_idxs = sorted(idxs, key=lambda i: areas[i])
                to_remove.update(s_idxs[1:])
        if to_remove:
            contours = [c for i,c in enumerate(contours) if i not in to_remove]
            areas = [c for i,c in enumerate(areas) if i not in to_remove]
            cX = [c for i,c in enumerate(cX) if i not in to_remove]
            cY = [c for i,c in enumerate(cY) if i not in to_remove]

        if not contours:
            return is_bright_frame, None, None, None, None

        # add back offset to get locations in the full image
        contours = [c+offset for c in contours]
        cX = [c+edge_cut_h for c in cX]
        cY = [c+edge_cut_v for c in cY]

        return is_bright_frame, contours, areas, cX, cY


def _get_contours(gray_frame, thresh_func, block_size, min_area, max_area):
    # threshold the image
    thresholded_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_func, block_size, 7)

    # Find contours
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # get moments and circularity for all the contours
    moments     = [cv2.moments(c) for c in contours]
    circularity = [0. if (p:=cv2.arcLength(c, True))==0 else 4*np.pi*m['m00']/p**2 for c,m in zip(contours,moments)]

    # filter out contours with wrong area or insufficiently circular
    out = list(zip(*[(con,m) for con,m,cir in zip(contours,moments,circularity) if m['m00']>=min_area and m['m00']<=max_area and cir>.75]))
    if not out: # nothing detected
        return [],[],[],[],None
    contours,moments = out
    # sort by area
    sorted_idx = sorted(range(len(contours)), key=lambda x: moments[x]['m00'], reverse=True)
    contours = [contours[i] for i in sorted_idx]
    moments  = [ moments[i] for i in sorted_idx]
    areas    = [m['m00'] for m in moments]
    # compute centroid
    cX, cY = zip(*[(m['m10']/a, m['m01']/a) for m,a in zip(moments,areas)])
    return contours, areas, cX, cY, thresholded_frame