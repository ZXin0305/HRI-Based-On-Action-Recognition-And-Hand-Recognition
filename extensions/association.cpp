#include <math.h>
#include <memory>
#include <vector>
#include <tuple>
#include <cuda_runtime_api.h>
#include <torch/torch.h>
#include "arraygpu.hpp"
#include "gpu/cuda_cal.h"
// #include <pybind11/numpy.h>
// #include <pybind11/eigen.h>
// #include "Timer.hpp"

namespace py= pybind11;
using namespace std;
using Tensor = torch::Tensor;

const int nJoints = 15;
const int nLimbs = 14;
const int maxPeaks = 127;
vector<int> heatmapDim = {43, 128, 208};  // 1/4 height, 1/4 width,  maybe just adjust here
const float dsScale = 4;
vector<unsigned int> jointPairs = {0, 1,   0, 2,   0, 9,   9, 10,   10, 11,
		                           0, 3,   3, 4,   4, 5,   2, 12,   12, 13, 
		                           13,14,  2, 6,   6, 7,   7, 8};
// statistic bone length 
vector<float> bone_length = {26.42178982, 48.36980909,
                             14.88291009, 31.28002332, 23.915707,
                             14.97674918, 31.28002549, 23.91570732,
                             12.4644364,  48.26604433, 39.03553194,
                             12.4644364, 48.19076948, 39.03553252};

// extract heatmap peaks --> joints location
tuple< vector<Tensor>, vector<Tensor> > extract(Tensor &hmsIn)
{
    
    const float nmsThreshold = 0.2f;
    const float nmsOffset = 0.5f;
    
	vector<unsigned int> mapIdx; // idx of paf
	auto mapOffset = nJoints;    // no bkg
	// get the paf maps idx
    for (auto i = 0; i < nLimbs; i++) {
        mapIdx.push_back(mapOffset + 2*i);
        mapIdx.push_back(mapOffset + 2*i+1);
    }

    shared_ptr<ArrayGpu<float>> peaks(new ArrayGpu<float>({ nJoints, maxPeaks + 1, 3 }));
    shared_ptr<ArrayGpu<int>> peakKernal(new ArrayGpu<int>({ heatmapDim[0], heatmapDim[1], heatmapDim[2] }));  // channel, h, w
    shared_ptr<ArrayGpu<float>> heatMap(new ArrayGpu<float>({ heatmapDim[0], heatmapDim[1], heatmapDim[2] })); // channel, h, w

    cudaMemcpy(heatMap->getPtr(), hmsIn.data_ptr<float>(), heatmapDim[0]*heatmapDim[1]*heatmapDim[2]*sizeof(float), cudaMemcpyDeviceToDevice);

    array<int, 4> peakSourceSize{ 1, nJoints, heatmapDim[1], heatmapDim[2] };  // 1, num_joints, h, w
    array<int, 4> peakTargetSize{ 1, nJoints, maxPeaks + 1, 3 };               // 1, num_joints, h, w

    nmsGpu(peaks->getPtr(), peakKernal->getPtr(), heatMap->getPtr(), nmsThreshold, peakTargetSize, peakSourceSize, nmsOffset);

    // get PAF scores
    shared_ptr<ArrayGpu<float>> pairScoresGpuPtr(new ArrayGpu<float>({ nLimbs, maxPeaks, maxPeaks }));
    shared_ptr<ArrayGpu<unsigned int>> pBodyPartPairsGpuPtr(new ArrayGpu<unsigned int>({ (int)jointPairs.size() })); // 1D
	cudaMemcpy(pBodyPartPairsGpuPtr->getPtr(), jointPairs.data(), jointPairs.size() * sizeof(unsigned int),
		cudaMemcpyHostToDevice);
    shared_ptr<ArrayGpu<unsigned int>> pMapIdxGpuPtr(new ArrayGpu<unsigned int>({ (int)mapIdx.size() })); // 1D
	cudaMemcpy(pMapIdxGpuPtr->getPtr(), mapIdx.data(), mapIdx.size() * sizeof(unsigned int),
		cudaMemcpyHostToDevice);

    connectBodyPartsGpu(pairScoresGpuPtr->getPtr(),
        heatMap->getConstPtr(), heatMap->getSize()[2], heatMap->getSize()[1], peaks->getConstPtr(),
        pBodyPartPairsGpuPtr->getConstPtr(), pMapIdxGpuPtr->getConstPtr());

    // gpu --> cpu
    unique_ptr<float> peaksUpData(new float[(maxPeaks+1)*3*nJoints]);
    float* peaksCpuPtr = peaksUpData.get();
    cudaMemcpy(peaksCpuPtr, peaks->getConstPtr(), 
               sizeof(float)*nJoints*(maxPeaks+1)*3, cudaMemcpyDeviceToHost);
    
    // save nms
    vector<Tensor> poseCandidates;
    // #pragma omp parallel for  
    for (int i = 0; i < nJoints; i++) {
        float* curPeaksCpuPtr = peaksCpuPtr + i * (maxPeaks + 1) * 3;
        int mPeakSize = curPeaksCpuPtr[0];
        Tensor mPeakData = torch::empty({mPeakSize, 3});  //每一种关节点的size都为 (Num_this_type_joint, (y,x,score))
        for (int i = 1; i <= mPeakSize; i++) {
            mPeakData[i-1][0] = curPeaksCpuPtr[3*i];
            mPeakData[i-1][1] = curPeaksCpuPtr[3*i+1];
            mPeakData[i-1][2] = curPeaksCpuPtr[3*i+2];
        }
        poseCandidates.push_back(mPeakData);
    }

    // gpu --> cpu
    unique_ptr<float> pariUpData(new float[maxPeaks*maxPeaks*nLimbs]);
	float* pairScoresCpuPtr = pariUpData.get();
	cudaMemcpy(pairScoresCpuPtr, pairScoresGpuPtr->getConstPtr(), 
               sizeof(float)*nLimbs*maxPeaks*maxPeaks, cudaMemcpyDeviceToHost);
    
    // save paf score
    vector<Tensor> pafCandidates(nLimbs);
    // #pragma omp parallel for
    for (int i = 0; i < nLimbs; i++) {
        float* curPairScoresCpuPtr = pairScoresCpuPtr + i*maxPeaks*maxPeaks;  //指针赋值
        int joint1Idx = jointPairs[2*i];
        int joint2Idx = jointPairs[2*i+1];
        int nPeaks1 = poseCandidates[joint1Idx].sizes()[0];
        int nPeaks2 = poseCandidates[joint2Idx].sizes()[0];
        Tensor mPafData = torch::empty({nPeaks1, nPeaks2});
        for (int i = 0; i < nPeaks1; i++) {
            for (int j = 0; j < nPeaks2; j++) {
                float score = curPairScoresCpuPtr[maxPeaks*i+j];
                mPafData[i][j] = score;
            }
        }
        pafCandidates[i] = mPafData;
    }

    tuple< vector<Tensor>, vector<Tensor> > resInfo = {poseCandidates, pafCandidates};

    return resInfo;
}


Tensor findConnectedJoints(Tensor hmsIn, Tensor rDepth, int rootIdx = 2, bool distFlag = true)
{
    // rootIdx: 2 for pelvis, 0 for neck
    // rDepth: root depth map, channel --> 1
    tuple< vector<Tensor>, vector<Tensor> > resInfo = extract(hmsIn);

    vector<Tensor> peaks = get<0>(resInfo);
    // cout<<peaks<<endl;
    vector<Tensor> pafScores = get<1>(resInfo);

    int personNum = (int)peaks[rootIdx].sizes()[0];  //在这里根据深度图中的个数判定有多少人
    //cout<<"human num: "<< personNum << endl;
    if (personNum == 0) {
        Tensor empty = torch::empty({0});
        return empty;
    }

    // extract root depth ..
    Tensor predRootDepth = torch::empty({personNum});
    for (int i = 0; i < personNum; i++) {
        Tensor dep = rDepth[ peaks[rootIdx][i][1].item().to<int>() ][ peaks[rootIdx][i][0].item().to<int>() ];  // rDepth[(y, x)] , .item().to<int>()
        predRootDepth[i] = dep;
    }
    // ordinal prior
    auto sortDep = predRootDepth.sort(0, false);  // 对根深度值进行排序, 由小到大 
    auto sortDepth = get<0>(sortDep);             // 深度值的数值
    auto sortIndex = get<1>(sortDep);             // 对应的序号
    // cout<<"original order --> "<< predRootDepth <<endl;
    // cout<<"sort order --> " << sortDepth <<endl;
    // cout<<"sort index --> " << sortIndex <<endl;

    // 加上的 --> 置信度优先关联  2021.11.13
    vector<Tensor> sortIdxList(nJoints);    // 为了改变dstList中的配对顺序, (15, ...)
    for(int i = 0; i < nJoints; i++)
    {
        
        int candidates_num = (int)peaks[i].sizes()[0];
        if(candidates_num == 0)
        {          
            continue;
        }

        Tensor sortList = torch::empty({candidates_num});
        for(int j = 0; j<candidates_num; j++)
        {
            auto score = peaks[i][j][2];   // score
            sortList[j] = score;
        }

        auto sortScorePair = sortList.sort(0, true);   // 大 --> 小
        auto tmpSortIdx = get<1>(sortScorePair);
        sortIdxList[i] = tmpSortIdx;  //根据置信度,将置信度较大的候选节点的id往前排

    }

    // 重新对数值进行映射
    // 为了配合深度优先关联的方法, 但是只是对于[2,0]进行了深度关联
    // 这里面的序号个数和人数相关,因为这里面的序号都是作为开始点的序号使用的
    vector<vector<int>> remap(nJoints);               
    for (int i = 0; i < nJoints; i++) {
        for (int j = 0; j < personNum; j++) {
            if (i == rootIdx)  remap[i].push_back(sortIndex[j].item().to<int>());
            else  remap[i].push_back(j);
        }
    }

    // 将更深的根深度值首先进行分组
    Tensor predBodys = torch::zeros({personNum, nJoints, 4});  //shape --> (人数,关节点个数,(y,x,Z,score))
    for (int i = 0; i < personNum; i++) {
        int sidx = sortIndex[i].item().to<int>();
        predBodys[i][rootIdx][0] = peaks[rootIdx][sidx][0];
        predBodys[i][rootIdx][1] = peaks[rootIdx][sidx][1];
        predBodys[i][rootIdx][3] = peaks[rootIdx][sidx][2];
    }

    // 开始链接 从 2-->0进行配对
    for (int j = 0; j < nLimbs; j++) {
        int i, srcJointId, dstJointId;
        bool flip = false;
        // ATTN: messy !!!
        if (j == 0)  i = 1;  
        else if (j == 1)  i = 0;
        else  i = j;

        if ( rootIdx == 2 && i == 1 ) {
            srcJointId = jointPairs[2*i+1];   // 只有一个 2 --> 0
            dstJointId = jointPairs[2*i];
            flip = true;
        } else {                                   
            srcJointId = jointPairs[2*i];
            dstJointId = jointPairs[2*i+1];
        }

        vector<int> remapSrc = remap[srcJointId];

        // ----------------------------------------------------
        // 用处:
        // 从[2-->0]开始,进行关联,
        // 先从predBodys中找到关联对中的开始点(具体的点,个数等于找到的人数) --> 对应的是关联对的结束点的所有的点的集合
        // 如: srcList[(y1,x1,0,score1), (y2,x2,0,score2)]
        //     dstList[(y11,x11,0,score11), (y22,x22,0,score22)]  --> peaks.. 这里是两个人的情况
        Tensor srcList = predBodys.select(1, srcJointId);   // tensor.select() 选定维度对tensor张量进行切片. 第一个参数是进行维度选择,后面的参数索引的序号
        Tensor dstList = peaks[dstJointId];             
        int dstSize = dstList.sizes()[0];
        if (dstSize == 0)  continue;
        Tensor dstIdxList = sortIdxList[dstJointId]; 

        Tensor curPafScore = pafScores[i];   //当前关联对的paf分数            
        vector<int> used(dstSize, 0);        //一个vector, 大小是关联对中结束点的大小 初始化都为0, 成功关联上就变成了1, 表示已经被关联上

        // 当前关联对的开始点的循环
        for (int k1 = 0; k1 < srcList.sizes()[0]; k1++) {
            // 如果置信度为0
            if (srcList[k1][3].item().to<float>() < 1e-5)
                continue;
            // 取出这个开始关节点
            vector<float> srcJoint;
            if (distFlag) {
                srcJoint = {srcList[k1][0].item().to<float>(), srcList[k1][1].item().to<float>()};
            }
            
            // 根据不同的深度进行骨长的变化,具体的思想就是,远小近大:深度小的骨长大,深度大的骨长小,因此除以的是深度值
            float bone_dist = 1.2 * bone_length[i] / sortDepth[k1].item().to<float>();
            float maxScore = 0.0;
            int maxIdx = -1;

            // 当前关联对的结束点的循环
            for (int k2 = 0; k2 < dstList.sizes()[0]; k2++) {
                int newK2 = dstIdxList[k2].item().to<int>();   // 置信度优先关联的开关
                if (used[newK2])  continue;
                float score;

                //根据根深度进行优先排序, 但是好像只有在 2 --> 0 这一步有
                if (flip)  score = curPafScore[newK2][remapSrc[k1]].item().to<float>();   // 2-->0
                else  score = curPafScore[remapSrc[k1]][newK2].item().to<float>();

                // score == paf_score + adaptive_bone_score
                // 实质上就是重新加了一个分数
                if (distFlag) {
                    if (score > 0) {
                        vector<float> dstJoint = {dstList[newK2][0].item().to<float>(), dstList[newK2][1].item().to<float>()};  // y,x
                        float limb_dist = sqrt(pow(srcJoint[0] - dstJoint[0], 2) + pow(srcJoint[1] - dstJoint[1], 2));    //calculate bone length  (在像素平面计算)
                        // adaptive distance constraint
                        // 动态骨长约束
                        score += min(bone_dist / limb_dist / dsScale - 1, 0.0f);  //paf的分数加上动态骨长约束的分数
                    }
                }
                if (score > maxScore) {
                    maxScore = score;
                    maxIdx = newK2;
                }
            }
            if (maxScore > 0) {   // can be tuned, 0 as threshold
                predBodys[k1][dstJointId][0] = peaks[dstJointId][maxIdx][0];
                predBodys[k1][dstJointId][1] = peaks[dstJointId][maxIdx][1];
                predBodys[k1][dstJointId][3] = peaks[dstJointId][maxIdx][2]; 
                // remap
                remap[dstJointId][k1] = maxIdx;     //保证这个点在后来作为关联对的开始点的时候，是作为根深度值最小的那一个开始的。。

                used[maxIdx] = 1;                   //已经配对完成
            }
        }
    }

    return predBodys;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("extract", &extract);
    m.def("connect", &findConnectedJoints, py::arg("hmsIn"), py::arg("rDepth"), 
                                           py::arg("rootIdx")=2, py::arg("distFlag")=true);
}
