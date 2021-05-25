from math import pi
import numpy as np
import open3d as o3d

from   scipy.spatial import ConvexHull, convex_hull_plot_2d
from   numpy.linalg  import eig, inv
from   aggregation   import mcs # NOTE this requires https://github.com/jleinonen/aggregation

import matplotlib.pyplot as plt

def weights_on_mass(weights):
    """
    weights: Mass of the Voxels

    Returns: Dictionary with weight distributions

    """

    w=(weights-weights.min())/(weights.max()-weights.min())
    
    mass = np.sum(w)

    vec_test = (np.arange(0,1.0,0.05))[::-1]
    w_mass = np.empty(len(vec_test))*0.
    w_dens = np.empty(len(vec_test))*0.

    # Weight on average mass and on total mass
    for i in range(len(w_mass)):
        condi=np.where(w >= vec_test[i])
        w_mass[i] = np.mean(w[condi])
        w_dens[i] = np.sum(w[condi])/mass

    w_out = (w_mass*w_dens)/np.sum(w_mass*w_dens)
    th = vec_test[np.argmax(w_out)]* (weights.max()-weights.min()) + weights.min()

    out ={'w_mass':w_mass,
        'w_dens':w_dens,
        'w_out':w_out,
        'vec_test':vec_test,
        'th':th}
    
    return out


def weighted_hull_ellipsoid(points,weights,pix_size=1.0):
    """
    points:    [N,*] array with x,y,z position of the points
    weights:   array of length N. Weighting variable.
    pix_size:  size in [m] of the pixel.

    Returns three scalar: Volume (Hull) [m^3], Area (Hull) [m^2], Axis Ratio (spherioid) [-]

    """
    w=(weights-weights.min())/(weights.max()-weights.min())
    
    mass = np.sum(w)

    vec_test = (np.arange(0,1.0,0.05))[::-1]
    w_mass = np.zeros(len(vec_test))
    w_dens = np.zeros(len(vec_test))

    V_vec    = np.zeros(len(vec_test)) # Volume of Convex Hull
    A_vec    = np.zeros(len(vec_test)) # Area of Convex Hull
    R_a_vec  = np.zeros(len(vec_test)) # Axis ratio of fitted hellipsoid


    # Weight on average mass and on total mass
    for i in range(len(w_mass)):
        condi=np.where(w >= vec_test[i])
        data_in=points[condi]

        try:
            c_h = ConvexHull(data_in*pix_size) # Convex hull
            A_vec[i]=(c_h.area)**0.5           # Weight is given on a length factor, converted later to area or Volume !! 
            V_vec[i]=(c_h.volume)**(1./3.)

            w_mass[i] = np.mean(w[condi])
            w_dens[i] = np.sum(w[condi])/mass

            # fit ellipsoid on convex hull
            lH       = len(c_h.vertices)
            hull     = np.zeros((lH,3))
            for j in range(len(c_h.vertices)):
                hull[j] = pix_size*data_in[c_h.vertices[j]]
            hull     = np.transpose(hull)         
            
            eansa            = ls_ellipsoid(hull[0],hull[1],hull[2]) #get ellipsoid polynomial coefficients
            center,axes,inve = polyToParams3D(eansa,False)           #get ellipsoid 3D parameters

            R_a_vec[i] = axes.min()/axes.max()
        except:
            print("An exception occurred in the Convex Hull generation") 

    w_out = (w_mass*w_dens)/np.sum(w_mass*w_dens)
    #return (np.sum(w_out*A_vec))**2., (np.sum(w_out*V_vec))**3., np.sum(w_out*R_a_vec)
    return A_vec[np.argmax(w_out)]**2., V_vec[np.argmax(w_out)]**3., R_a_vec[np.argmax(w_out)]



def weighted_sphere_radius(points,weights,return_threshold=False):
    """
    points: N,3 coordinates of the points
    weights: N weights of the points
    return_threshold: if True, it returns the optimal threshold 
        (within the range of "weights") to censor the data for
        geometrical calculations

    Returns: DIAMETER (sorry for the misleading funtion name) weighted estimate
        of the enclosing sphere. Units are pixels in this case.

    """
    w=(weights-weights.min())/(weights.max()-weights.min())
    
    mass = np.sum(w)

    vec_test = (np.arange(0,1.0,0.05))[::-1]
    w_mass = np.empty(len(vec_test))*0.
    w_dens = np.empty(len(vec_test))*0.
    d_vec  = np.empty(len(vec_test))*0.

    # Weight on average mass and on total mass
    for i in range(len(w_mass)):
        condi=np.where(w >= vec_test[i])
        data_in=points[condi]
        w_mass[i] = np.mean(w[condi])
        w_dens[i] = np.sum(w[condi])/mass

        d_vec[i] = 2*mcs.minimum_covering_sphere(data_in)[1]

    w_out = (w_mass*w_dens)/np.sum(w_mass*w_dens)
    
    if not return_threshold:
        #d_out=np.sum(w_out*d_vec)
        d_out=d_vec[np.argmax(w_out)]
        return d_out
    else:
        th = vec_test[np.argmax(w_out)]* (weights.max()-weights.min()) + weights.min()
        return th


def ls_ellipsoid(xx,yy,zz):                                  
    #finds best fit ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    #least squares fit to a 3D-ellipsoid
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
    #
    # Note that sometimes it is expressed as a solution to
    #  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
    # where the last six terms have a factor of 2 in them
    # This is in anticipation of forming a matrix with the polynomial coefficients.
    # Those terms with factors of 2 are all off diagonal elements.  These contribute
    # two terms when multiplied out (symmetric) so would need to be divided by two
    
    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = xx[:,np.newaxis]
    y = yy[:,np.newaxis]
    z = zz[:,np.newaxis]
    
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
    K = np.ones_like(x) #column of ones
    
    #np.hstack performs a loop over all samples and creates
    #a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.
    
    JT=J.transpose()
    JTJ = np.dot(JT,J)
    InvJTJ=np.linalg.inv(JTJ);
    ABC= np.dot(InvJTJ, np.dot(JT,K))

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa=np.append(ABC,-1)

    return (eansa)

def polyToParams3D(vec,printMe):                             
    #gets 3D parameters of an ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    # convert the polynomial form of the 3D-ellipsoid to parameters
    # center, axes, and transformation matrix
    # vec is the vector whose elements are the polynomial
    # coefficients A..J
    # returns (center, axes, rotation matrix)
    
    #Algebraic form: X.T * Amat * X --> polynomial form
    
    if printMe: print('\npolynomial\n',vec)
    
    Amat=np.array(
    [
    [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
    [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
    [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
    [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
    ])
    
    if printMe: print('\nAlgebraic form of polynomial\n',Amat)
    
    #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
    # equation 20 for the following method for finding the center
    A3=Amat[0:3,0:3]
    A3inv=inv(A3)
    ofs=vec[6:9]/2.0
    center=-np.dot(A3inv,ofs)
    if printMe: print('\nCenter at:',center)
    
    # Center the ellipsoid at the origin
    Tofs=np.eye(4)
    Tofs[3,0:3]=center
    R = np.dot(Tofs,np.dot(Amat,Tofs.T))
    if printMe: print('\nAlgebraic form translated to center\n',R,'\n')
    
    R3=R[0:3,0:3]
    R3test=R3/R3[0,0]
    # print('normed \n',R3test)
    s1=-R[3, 3]
    R3S=R3/s1
    (el,ec)=eig(R3S)
    
    recip=1.0/np.abs(el)
    axes=np.sqrt(recip)
    if printMe: print('\nAxes are\n',axes  ,'\n')
    
    inve=inv(ec) #inverse is actually the transpose here
    if printMe: print('\nRotation matrix\n',inve)
    return (center,axes,inve)

def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,1, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result




def point_cloud_match(xyz, xyz2, filter_ref=None, filter_test=None, prune_cloud2=False,w2=None):
    """
    xyz:  N,3 point cloud vector (reference) --> use actual size units!
    xyz2: N2,3 point cloud vector to match   --> use actual size units!

    filter_ref: filter outliers in the reference point cloud according to a radius of
        filter_ref size
    filter_test: filter outliers in the point cloud to matchm according to a radius of
        filter_test size
    prune_cloud2: if True, prune the points of cloud2 according to an optimal mass threshold
    w2: if prune_cloud2 is True, the mass corresponding to xyz2 must be provided here.

    Returns:

    A translated and rotated xyz2

    """

    # Define and populate point clouds
    pcl=o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(xyz)

    pcl2=o3d.geometry.PointCloud()
    if not prune_cloud2:
        pcl2.points = o3d.utility.Vector3dVector(xyz2)
    else:
        th = weighted_sphere_radius(xyz2,w2,return_threshold=True)
        condi=np.where(w2 > th)
        pcl2.points = o3d.utility.Vector3dVector(xyz2[condi[0],:])

    # Preprocess point cloud (Global registration)
    source_down, source_fpfh = preprocess_point_cloud(pcl2, 0.0005)
    target_down, target_fpfh = preprocess_point_cloud(pcl, 0.0005)

    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            0.0005)

    #Outlier removal for two point clouds separately
    if filter_test is not None:
        pcl2_proc, outlier_index2 = o3d.geometry.PointCloud.remove_radius_outlier(pcl2,
                                                nb_points=6,
                                                radius=filter_test)
    else:
        pcl2_proc=pcl2

    if filter_ref is not None:
        pcl_proc, outlier_index = o3d.geometry.PointCloud.remove_radius_outlier(pcl,
                                                nb_points=2,
                                                radius=filter_ref)
    else:
        pcl_proc=pcl

    
    # Set up
    threshold = 0.01                    #Movement range threshold (1 cm seems fair)
    #trans_init = np.asarray([[1,0,0,0],  # 4x4 identity matrix, this is a transformation matrix,
    #                        [0,1,0,0],   # It means there is no displacement, no rotation, we enter
    #                        [0,0,1,0],   # This matrix is ​​the initial transformation
    #                        [0,0,0,1]])
    trans_init=result_ransac.transformation

    #Run icp
    reg_p2p = o3d.pipelines.registration.registration_icp(
            pcl2_proc, pcl_proc, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

    #Transform our matrix according to the output transformation matrix
    pcl2.transform(reg_p2p.transformation)
    if prune_cloud2:
        pcl2=o3d.geometry.PointCloud()
        pcl2.points = o3d.utility.Vector3dVector(xyz2)
        pcl2.transform(reg_p2p.transformation)

    return np.asarray(pcl2.points)[:,:3]

def match_at_scales(xyz,xyz2,mass,mass2,sc=0.001):
    """
    xyz: first set of points   (N,3)
    xyz2: second set of points  (N2,3)
    mass: N-array with mass for xyz
    mass2: N-array with mass for xyz2
    scale: the rematch scale [m]

    """    

    xmin=np.min([np.min(xyz[:,0]),np.min(xyz2[:,0])])
    ymin=np.min([np.min(xyz[:,1]),np.min(xyz2[:,1])])
    zmin=np.min([np.min(xyz[:,2]),np.min(xyz2[:,2])])

    xmax=np.max([np.max(xyz[:,0]),np.max(xyz2[:,0])])
    ymax=np.max([np.max(xyz[:,1]),np.max(xyz2[:,1])])
    zmax=np.max([np.max(xyz[:,2]),np.max(xyz2[:,2])])

    # Rescale the vectors in a different array
    NX=np.int(np.round((xmax-xmin)/sc)+1)
    NY=np.int(np.round((ymax-ymin)/sc)+1)
    NZ=np.int(np.round((zmax-zmin)/sc)+1)

    vec = np.zeros([NX,NY,NZ])
    vec2 = np.zeros([NX,NY,NZ])

    for x in range(NX):
        xx=round(x*sc+xmin,10)
        dist_x=np.abs(xx-xyz[:,0])
        dist2_x=np.abs(xx-xyz2[:,0])

        test = (dist_x <= sc*0.5)
        test2 = (dist2_x <= sc*0.5)
        if ((not test.any()) & (not test2.any())):
            continue

        for y in range(NY):
            yy=round(y*sc+ymin,10)
            dist_y=np.abs(yy-xyz[:,1])
            dist2_y=np.abs(yy-xyz2[:,1])

            test = (dist_y <= sc*0.5)
            test2 = (dist2_y <= sc*0.5)
            if ((not test.any()) & (not test2.any())):
               continue

            for z in range(NZ):
                zz=round(z*sc+zmin,10)
                dist_z=np.abs(zz-xyz[:,2])
                dist2_z=np.abs(zz-xyz2[:,2])

                # Fill the matrix
                condi = np.where((dist_x <= sc*0.5) & (dist_y <= sc*0.5) & (dist_z <= sc*0.5))
                vec[x,y,z]=np.sum(mass[condi])

                condi = np.where((dist2_x <= sc*0.5) & (dist2_y <= sc*0.5) & (dist2_z <= sc*0.5))
                vec2[x,y,z]=np.sum(mass2[condi])              

    test1=np.abs(np.sum(mass)-np.sum(vec))
    if test1 > 0.05*np.sum(mass):
        print("Warning: deviation in total mass of more than 5 percent after vector resampling: ")
        print(100*test1/np.sum(mass))

    test2=np.abs(np.sum(mass2)-np.sum(vec2))
    if test2 > 0.05*np.sum(mass2):
        print("Warning: deviation in total mass of more than 5 percent after vector resampling")
        print(100*test2/np.sum(mass2))

    return vec, vec2

def hss_occurrence(target,ref):

    """
    HSS score to test occurrence

    (based on: http://www.eumetrain.org/data/4/451/english/msg/ver_categ_forec/uos3/uos3_ko1.htm)

    """
    a = len(np.where((target > 0.) & (ref > 0.))[0])
    b= len(np.where((target > 0.) & (ref == 0.))[0])
    c= len(np.where((target == 0.) & (ref > 0.))[0])
    d= len(np.where((target == 0.) & (ref == 0.))[0])


    try:
        hss=2*(a*d-b*c)/((a+c)*(c+d)+(a+b)*(b+d))
    except:
        hss=1
    return hss

def rmse_occurrence(target=None,ref=None):

    """
    RMSE on occurrence

    """
    t=target.copy()
    r=ref.copy()

    t[np.where(target > 0.)]=1.
    r[np.where(ref > 0.)]=1.

    a = np.where((t > 0.) )              #| (ref > 0.))
    b=  np.where((r > 0.) )
    c = np.where((t > 0.) | (r > 0.))
    d = np.where((t > 0.) & (r > 0.))


    rmse = (np.mean((t[a]-r[a])**2.))**0.5    
    print("Hi")

    return rmse



        






