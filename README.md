# cartographer
Drone Geopositioning by Satellite Image Matching

## Status
- SIFT/ORB/SURF/Superpoint-based sparse feature extraction  
- Mutual Nearest Neighbor (MNN) matching  
- RANSAC-based similarity / affine estimation  
- Hyper-param tuned for this setup

Works reliably for **moderate rotation, scale, and viewpoint differences**, and serves as a **validated classical baseline** for drone–satellite matching.

### Known Limitations
- Degrades for **large rotations (~>60–70°)**  
- Matching performance is still bad for full map size **inlier ratio ~ 5%**

## Next Steps 
- Robust learned matching - SuperGlue/Gluestick
- Automate result generation