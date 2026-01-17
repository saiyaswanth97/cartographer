# cartographer
Drone Geopositioning by Satellite Image Matching

## Status
- SIFT/ORB/SURF/Superpoint-based sparse feature extraction  
- Mutual Nearest Neighbor (MNN) matching  
- RANSAC-based similarity / affine estimation  

Works reliably for **moderate rotation, scale, and viewpoint differences**, and serves as a **validated classical baseline** for drone–satellite matching.

### Known Limitations
- Degrades for **large rotations (~>60–70°)**  
- Sensitive to strong illumination and viewpoint changes  
- Requires careful threshold tuning (ratio test, RANSAC)
- 
## Next Steps 
- Robust learned matching - SuperGlue/Gluestick
- Automate result generation