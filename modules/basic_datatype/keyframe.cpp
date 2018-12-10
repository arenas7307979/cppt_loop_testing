#include "tracer.h"
#include "util_datatype.h"
#include "front_end/utility.h"
#include "keyframe.h"

static float IC_Angle(const cv::Mat& image, cv::Point2f pt,  std::vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        //  v=1 umax[v]=15
        //  v=2 umax[v]=15
        //  v=3 umax[v]=15
        //  v=4 umax[v]=14
        //  v=5 umax[v]=14
        //  v=6 umax[v]=14
        //  v=7 umax[v]=13
        //  v=8 umax[v]=13
        //  v=9 umax[v]=12
        //  v=10 umax[v]=11
        //  v=11 umax[v]=10
        //  v=12 umax[v]=9
        //  v=13 umax[v]=8
        //  v=14 umax[v]=6
        //  v=15 umax[v]=3
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }
    return cv::fastAtan2((float)m_01, (float)m_10);
}

static void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,  std::vector<int>& umax)
{
    for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }
}

Keyframe::Keyframe(const FramePtr frame, std::vector<int>& umax){
    //TODO:: stereo matching to increase mappoint of this keyframe.
    mKeyFrameID = frame->mKeyFrameID;
    mTwc = frame->mTwc;
    mNumStereo = frame->mNumStereo;

#if DEBUG_POSEGRAPH
    mImgL = frame->mImgL.clone();
#endif

    int image_rows = frame->mImgL.rows;
    int image_cols = frame->mImgL.cols;
    for(int i=0; i<frame->mv_uv.size(); i++){
        if(!frame->mvMapPoint[i] || frame->mvMapPoint[i]->empty())
            continue;
        //remove border point
        if (frame->mv_uv[i].x < 20 || frame->mv_uv[i].y < 20 || frame->mv_uv[i].x >= image_cols - 20
                || frame->mv_uv[i].y >= image_rows - 20) {
            continue;
        }

        mvPtCount.push_back(frame->mvPtCount[i]);
        mv_uv.push_back(frame->mv_uv[i]);
        mvMapPoint.push_back(frame->mvMapPoint[i]);
        cv::KeyPoint tmp_mv_uv;
        tmp_mv_uv.pt = frame->mv_uv[i];
        tmp_mv_uv.octave = 0;
        tmp_mv_uv.size = 31;
        k_pts.push_back(tmp_mv_uv);
    }
    std::vector<cv::KeyPoint> tmp_keypoint;
    cv::FAST(frame->mImgL, tmp_keypoint, 20, true);
    for(int i=0; i<tmp_keypoint.size(); i++){
        //remove border point
        if (tmp_keypoint[i].pt.x < 20 || tmp_keypoint[i].pt.y < 20 || tmp_keypoint[i].pt.x >= image_cols - 20
                || tmp_keypoint[i].pt.y >= image_rows - 20) {
            continue;
        }
        tmp_keypoint[i].size = 31;
        k_pts.push_back(tmp_keypoint[i]);
    }

    //Compute angle of points from tmp_mv_uv
    computeOrientation(frame->mImgL, k_pts, umax);
    //Compute ORB desc
    computeORBDescriptors(frame);
}

Keyframe::~Keyframe(){};


void Keyframe::computeORBDescriptors(const FramePtr frame){
    cv::Mat tmp_descriptors = cv::Mat::zeros((int)k_pts.size(), 32, CV_8UC1);
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    extractor->compute(frame->mImgL, k_pts, tmp_descriptors);
    changeORBdescStructure(tmp_descriptors, descriptors);
}

//ORB vocabulary desc cv::Mat -> std::vector<cv::Mat> each_rows;
void Keyframe::changeORBdescStructure(const cv::Mat &plain, std::vector<cv::Mat> &out)
{
    out.resize(plain.rows);

    for(int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}
