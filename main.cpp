#include<string>
#include<time.h>
#include<windows.h>
#include<iostream>
#include<fstream>
#include<Kinect.h>
#include <NuiKinectFusionApi.h>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#ifndef SAFE_FUSION_RELEASE_IMAGE_FRAME
#define SAFE_FUSION_RELEASE_IMAGE_FRAME(p) { if (p) { static_cast<void>(NuiFusionReleaseImageFrame(p)); (p)=NULL; } }
#endif
#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p); (p)=NULL; } }
#endif

// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}


void UpdateIntrinsics(NUI_FUSION_IMAGE_FRAME * pImageFrame, NUI_FUSION_CAMERA_PARAMETERS * params)
{
	if (pImageFrame != nullptr && pImageFrame->pCameraParameters != nullptr && params != nullptr)
	{
		pImageFrame->pCameraParameters->focalLengthX = params->focalLengthX;
		pImageFrame->pCameraParameters->focalLengthY = params->focalLengthY;
		pImageFrame->pCameraParameters->principalPointX = params->principalPointX;
		pImageFrame->pCameraParameters->principalPointY = params->principalPointY;
	}

	// Confirm we are called correctly
	_ASSERT(pImageFrame != nullptr && pImageFrame->pCameraParameters != nullptr && params != nullptr);
}

class KinectFusion
{
public:
	KinectFusion();
	HRESULT CreateFirstConnected();
	HRESULT InitializeKinectFusion();
	HRESULT SetupUndistortion();
	HRESULT OnCoordinateMappingChanged();
	HRESULT MapColorToDepth();
	bool checkCoordinateChange();
	void update();
	void processDepth();
	void fillShowDepth();
	void processIRImage();
	void processDepthToColor();
	void DepthToCloudWithMask(Mat depth, Mat mask, string fileName);
	void DepthToCloud(Mat depth, string fileName);


	~KinectFusion();
	Mat getColorImg() { return i_rgb; };
	Mat getDepthImg() { return i_depth; };
	Mat getShowDepth() { return i_depth_show; };
	Mat getIRImg() { return i_ir; };
	Mat getDepthToRgb() { return i_depthToRgb; };
	
	int                         m_cFrameCounter;		//camera tracking中已经有的frame的个数
	bool						m_bTrackingFailed;		//camera tracking是否失败标志
	int							m_cLostFrameCounter;	//连续帧之间丢失的帧数
private:

	IKinectSensor*              m_pNuiSensor;			// Current Kinect	
	IMultiSourceFrameReader*	m_pMultiFrameReader;
	IColorFrameReader*			m_pColorFrameReader;	//Color reader	
	IDepthFrameReader*          m_pDepthFrameReader;	// Depth reader
	IInfraredFrameReader*       m_pInfraredFrameReader;	//Infrared reader
	ICoordinateMapper*          m_pMapper;				//坐标系转换
	WAITABLE_HANDLE             m_coordinateMappingChangedEvent;//坐标系映射变换标志

	static const UINT                         m_cDepthWidth = 512;
	static const UINT                         m_cDepthHeight = 424;
	static const UINT                         m_cDepthImagePixels = 512 * 424;
	static const UINT						  m_cColorWidth = 1920;
	static const UINT						  m_cColorHeight = 1080;
	static const int						  cBytesPerPixel = 4; // for depth float and int-per-pixel raycast images
	static const int						  cVisibilityTestQuantShift = 2; // shift by 2 == divide by 4
	static const UINT16						  cDepthVisibilityTestThreshold = 50; //50 mm

	Mat i_rgb;				//注意：这里必须为4通道的图，Kinect的数据只能以Bgra格式传出
	UINT16*	depthData;		//原始的深度图的值
	Mat i_depth;			//uint16
	Mat i_depth_show;		//8位
	Mat i_depthToRgb;		//对应深度图的彩色图（大小和深度图一样，位置和深度图一一对应）
	Mat i_ir;				//红外图
	unsigned short* irData = nullptr;
	Mat i_irShow;			//用于显示的红外图
	

	NUI_FUSION_CAMERA_PARAMETERS			 m_cameraParameters;				//camera参数 focalx,focaly, principalPointX, principalPointsY;	
	int										 m_deviceIndex;				//当使用GPU时，选择的设备的索引
	NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE m_processorType;			//使用GPU或CPU

	ColorSpacePoint*			m_pColorCoordinates;		//颜色坐标
	DepthSpacePoint*			m_pDepthDistortionMap;	//深度图坐标（每个值代表深度图上的坐标）
	UINT*                       m_pDepthDistortionLT;		////标记camera frame下的坐标是否可以投影到深度图上，若不行，这将对应位置（此处用一维表示）的值设置为很大；若可以，将对应位置值赋值为位置坐标值（变成一维），用于过滤原始深度图
	UINT16*                     m_pDepthImagePixelBuffer;	//经过原始的深度图经过m_pDepthDistortionLT过滤后的深度值，不可取的地方赋值为0
	UINT16*                     m_pDepthVisibilityTestMap;
	NUI_FUSION_IMAGE_FRAME*     m_pColorImage;				//color数据

	bool						m_bHaveValidCameraParameters;	//是否有合法的相机参数
	bool                        m_bInitializeError;				//初始化是否成功标志
};

KinectFusion::KinectFusion() {
	m_pMultiFrameReader = NULL;
	i_rgb.create(m_cColorHeight, m_cColorWidth, CV_8UC4);
	depthData = new UINT16[m_cDepthImagePixels];
	i_depth.create(m_cDepthHeight, m_cDepthWidth, CV_16UC1);
	i_depth_show.create(m_cDepthHeight, m_cDepthWidth, CV_8UC1);
	i_depthToRgb = cv::Mat::zeros(m_cDepthHeight, m_cDepthWidth, CV_8UC4);
	i_ir.create(m_cDepthHeight, m_cDepthWidth, CV_16UC1);
	i_irShow.create(m_cDepthHeight, m_cDepthWidth, CV_8UC1);
	m_pNuiSensor = NULL;
	m_pColorFrameReader = NULL;
	m_pDepthFrameReader = NULL;
	m_pInfraredFrameReader = NULL;
	m_pMapper = NULL;
	m_coordinateMappingChangedEvent = NULL;
	m_pColorCoordinates = new ColorSpacePoint[m_cDepthImagePixels];

	m_deviceIndex = -1;	//自动选择GPU设备
	m_processorType = NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_AMP; //使用GPU或CPU

	// We don't know these at object creation time, so we use nominal values.
	// These will later be updated in response to the CoordinateMappingChanged event.
	m_cameraParameters.focalLengthX = NUI_KINECT_DEPTH_NORM_FOCAL_LENGTH_X;		//这个值在开始是不知道的，但在后面会的得到并需要更新
	m_cameraParameters.focalLengthY = NUI_KINECT_DEPTH_NORM_FOCAL_LENGTH_Y;
	m_cameraParameters.principalPointX = NUI_KINECT_DEPTH_NORM_PRINCIPAL_POINT_X;
	m_cameraParameters.principalPointY = NUI_KINECT_DEPTH_NORM_PRINCIPAL_POINT_Y;
	
	m_pDepthImagePixelBuffer = nullptr;
	m_pDepthDistortionMap = nullptr;
	m_pDepthDistortionLT = nullptr;
	m_bHaveValidCameraParameters = false;
	m_bInitializeError = false;
	m_bTrackingFailed = false;
	m_cFrameCounter = 0;
	m_cLostFrameCounter = 0;

	m_pColorImage = nullptr;

	m_pDepthVisibilityTestMap = nullptr;

}
KinectFusion::~KinectFusion() {
	SAFE_DELETE_ARRAY(depthData);
	SafeRelease(m_pColorFrameReader);
	SafeRelease(m_pMultiFrameReader);
	SafeRelease(m_pDepthFrameReader);
	SafeRelease(m_pInfraredFrameReader);
	SafeRelease(m_pMapper);
	if (nullptr != m_pMapper)
		m_pMapper->UnsubscribeCoordinateMappingChanged(m_coordinateMappingChangedEvent);
	if (m_pNuiSensor) {
		m_pNuiSensor->Close();
	}
	SafeRelease(m_pNuiSensor);
	
	SAFE_DELETE_ARRAY(m_pColorCoordinates);
	SAFE_DELETE_ARRAY(m_pDepthDistortionMap);
	SAFE_DELETE_ARRAY(m_pDepthDistortionLT);
	SAFE_DELETE_ARRAY(m_pDepthImagePixelBuffer);
	SAFE_DELETE_ARRAY(m_pDepthVisibilityTestMap);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pColorImage);
}
HRESULT KinectFusion::CreateFirstConnected()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pNuiSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	//if (m_pNuiSensor)
	//{
	//	//Initialize the Kinect
	//	hr = m_pNuiSensor->Open();
	//	//get color reader
	//	IColorFrameSource* pColorFrameSource = NULL;		
	//	if (SUCCEEDED(hr))
	//		hr = m_pNuiSensor->get_ColorFrameSource(&pColorFrameSource);
	//	if (SUCCEEDED(hr))
	//		hr = pColorFrameSource->OpenReader(&m_pColorFrameReader);
	//	SafeRelease(pColorFrameSource);
	//	// get the depth reader
	//	IDepthFrameSource* pDepthFrameSource = NULL;
	//	if (SUCCEEDED(hr))
	//	{
	//		hr = m_pNuiSensor->get_DepthFrameSource(&pDepthFrameSource);
	//	}
	//	if (SUCCEEDED(hr))
	//	{
	//		hr = m_pNuiSensor->get_CoordinateMapper(&m_pMapper);
	//	}
	//	if (SUCCEEDED(hr))
	//	{
	//		hr = m_pMapper->SubscribeCoordinateMappingChanged(&m_coordinateMappingChangedEvent);
	//	}
	//	if (SUCCEEDED(hr))
	//	{
	//		hr = pDepthFrameSource->OpenReader(&m_pDepthFrameReader);
	//	}
	//	if (SUCCEEDED(hr)) {
	//		hr = m_pNuiSensor->OpenMultiSourceFrameReader(FrameSourceTypes::FrameSourceTypes_Color |
	//			FrameSourceTypes::FrameSourceTypes_Depth, &m_pMultiFrameReader);
	//	}
	//	SafeRelease(pDepthFrameSource);
	//}

	if (m_pNuiSensor) {
		hr = m_pNuiSensor->Open();
		if (SUCCEEDED(hr)) {
			hr = m_pNuiSensor->OpenMultiSourceFrameReader(FrameSourceTypes::FrameSourceTypes_Color |
				FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Infrared, &m_pMultiFrameReader);
		}
		if (SUCCEEDED(hr))
			hr = m_pNuiSensor->get_CoordinateMapper(&m_pMapper);
		if (SUCCEEDED(hr))
			hr = m_pMapper->SubscribeCoordinateMappingChanged(&m_coordinateMappingChangedEvent);
	}
	if (nullptr == m_pNuiSensor || FAILED(hr))
	{
		cout << "No ready Kinect found!" << endl;
		return E_FAIL;
	}
	return hr;
}
HRESULT KinectFusion::InitializeKinectFusion() {
	HRESULT hr = S_OK;
	//设备检查
	// Check to ensure suitable DirectX11 compatible hardware exists before initializing Kinect Fusion
	WCHAR description[MAX_PATH];    //The description of the device.
	WCHAR instancePath[MAX_PATH];	//The DirectX instance path of the GPU being used for reconstruction.
	UINT memorySize = 0;
	if (FAILED(hr = NuiFusionGetDeviceInfo(m_processorType, m_deviceIndex, &description[0], ARRAYSIZE(description), &instancePath[0], ARRAYSIZE(instancePath), &memorySize)))
	{
		if (hr == E_NUI_BADINDEX)
		{
			// This error code is returned either when the device index is out of range for the processor 
			// type or there is no DirectX11 capable device installed. As we set -1 (auto-select default) 
			// for the device index in the parameters, this indicates that there is no DirectX11 capable 
			// device. The options for users in this case are to either install a DirectX11 capable device
			// (see documentation for recommended GPUs) or to switch to non-real-time CPU based 
			// reconstruction by changing the processor type to NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_CPU.
			cout << "No DirectX11 device detected, or invalid device index - Kinect Fusion requires a DirectX11 device for GPU-based reconstruction." << endl;
		}
		else
			cout << "Failed in call to NuiFusionGetDeviceInfo." << endl;
		return hr;
	}
	// 创建颜色帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_COLOR, m_cColorWidth, m_cColorHeight, &m_cameraParameters, &m_pColorImage);
	if (FAILED(hr))
	{
		cout << "Failed to initialize Kinect Fusion m_pShadedSurface." << endl;
		return hr;
	}

	//分配深度图相关的内存
	_ASSERT(m_pDepthImagePixelBuffer == nullptr);
	m_pDepthImagePixelBuffer = new(std::nothrow) UINT16[m_cDepthImagePixels];
	if (nullptr == m_pDepthImagePixelBuffer)
	{
		cout << "Failed to initialize Kinect Fusion depth image pixel buffer." << endl;
		return hr;
	}
	_ASSERT(m_pDepthDistortionMap == nullptr);
	m_pDepthDistortionMap = new(std::nothrow) DepthSpacePoint[m_cDepthImagePixels];
	if (nullptr == m_pDepthDistortionMap)
	{
		cout << "Failed to initialize Kinect Fusion depth image distortion buffer." << endl;
		return E_OUTOFMEMORY;
	}
	SAFE_DELETE_ARRAY(m_pDepthDistortionLT);
	m_pDepthDistortionLT = new(std::nothrow) UINT[m_cDepthImagePixels];
	if (nullptr == m_pDepthDistortionLT)
	{
		cout << "Failed to initialize Kinect Fusion depth image distortion Lookup Table." << endl;
		return E_OUTOFMEMORY;
	}


	SAFE_DELETE_ARRAY(m_pDepthVisibilityTestMap);
	m_pDepthVisibilityTestMap = new(std::nothrow) UINT16[(m_cColorWidth >> cVisibilityTestQuantShift) * (m_cColorHeight >> cVisibilityTestQuantShift)];

	if (nullptr == m_pDepthVisibilityTestMap)
	{
		cout << "Failed to initialize Kinect Fusion depth points visibility test buffer." << endl;
		return E_OUTOFMEMORY;
	}


	// If we have valid parameters, let's go ahead and use them.
	if (m_cameraParameters.focalLengthX != 0)
		SetupUndistortion();
	return hr;
}
HRESULT KinectFusion::SetupUndistortion()
{
	HRESULT hr = E_UNEXPECTED;

	//深度图坐标系原点不能在图像中心，否则此摄像机参数就不合法
	if (m_cameraParameters.principalPointX != 0)
	{
		//深度图的四个边坐标：左上（0，0），右上（1，0）（因为k矩阵里参数分别都除以深度图的宽和高），左下（0，1），右下（1，1）反投影成camera frame 下且z为1（1m）的空间点
		CameraSpacePoint cameraFrameCorners[4] = //at 1 meter distance. Take into account that depth frame is mirrored
		{
			/*LT*/{ -m_cameraParameters.principalPointX / m_cameraParameters.focalLengthX, m_cameraParameters.principalPointY / m_cameraParameters.focalLengthY, 1.f },
			/*RT*/{ (1.f - m_cameraParameters.principalPointX) / m_cameraParameters.focalLengthX, m_cameraParameters.principalPointY / m_cameraParameters.focalLengthY, 1.f },
			/*LB*/{ -m_cameraParameters.principalPointX / m_cameraParameters.focalLengthX, (m_cameraParameters.principalPointY - 1.f) / m_cameraParameters.focalLengthY, 1.f },
			/*RB*/{ (1.f - m_cameraParameters.principalPointX) / m_cameraParameters.focalLengthX, (m_cameraParameters.principalPointY - 1.f) / m_cameraParameters.focalLengthY, 1.f }
		};

		//将4个1m处的空间点边界内的空间划分为个数和深度图大小相同的空间点，然后将这些点投影回深度图上。
		for (UINT rowID = 0; rowID < m_cDepthHeight; rowID++)
		{
			const float rowFactor = float(rowID) / float(m_cDepthHeight - 1);
			const CameraSpacePoint rowStart =
			{
				cameraFrameCorners[0].X + (cameraFrameCorners[2].X - cameraFrameCorners[0].X) * rowFactor,
				cameraFrameCorners[0].Y + (cameraFrameCorners[2].Y - cameraFrameCorners[0].Y) * rowFactor,
				1.f
			};

			const CameraSpacePoint rowEnd =
			{
				cameraFrameCorners[1].X + (cameraFrameCorners[3].X - cameraFrameCorners[1].X) * rowFactor,
				cameraFrameCorners[1].Y + (cameraFrameCorners[3].Y - cameraFrameCorners[1].Y) * rowFactor,
				1.f
			};

			const float stepFactor = 1.f / float(m_cDepthWidth - 1);
			const CameraSpacePoint rowDelta =
			{
				(rowEnd.X - rowStart.X) * stepFactor,
				(rowEnd.Y - rowStart.Y) * stepFactor,
				0
			};

			_ASSERT(m_cDepthWidth == NUI_DEPTH_RAW_WIDTH);
			CameraSpacePoint cameraCoordsRow[NUI_DEPTH_RAW_WIDTH];

			CameraSpacePoint currentPoint = rowStart;
			for (UINT i = 0; i < m_cDepthWidth; i++)
			{
				cameraCoordsRow[i] = currentPoint;
				currentPoint.X += rowDelta.X;
				currentPoint.Y += rowDelta.Y;
			}

			hr = m_pMapper->MapCameraPointsToDepthSpace(m_cDepthWidth, cameraCoordsRow, m_cDepthWidth, &m_pDepthDistortionMap[rowID * m_cDepthWidth]);
			if (FAILED(hr))
			{
				cout << "Failed to initialize Kinect Coordinate Mapper." << endl;
				return hr;
			}
		}

		if (nullptr == m_pDepthDistortionLT)
		{
			cout << "Failed to initialize Kinect Fusion depth image distortion Lookup Table." << endl;
			return E_OUTOFMEMORY;
		}

		//若反投影回的深度图位置不合法，这将此处位置的深度图标记为不可从空间坐标投影回来，用于后面过滤采集到的深度图
		UINT* pLT = m_pDepthDistortionLT;
		for (UINT i = 0; i < m_cDepthImagePixels; i++, pLT++)
		{
			//nearest neighbor depth lookup table 
			UINT x = UINT(m_pDepthDistortionMap[i].X + 0.5f);
			UINT y = UINT(m_pDepthDistortionMap[i].Y + 0.5f);

			*pLT = (x < m_cDepthWidth && y < m_cDepthHeight) ? x + y * m_cDepthWidth : UINT_MAX;
		}
		m_bHaveValidCameraParameters = true;
	}
	else
	{
		m_bHaveValidCameraParameters = false;
	}
	return S_OK;
}
HRESULT KinectFusion::OnCoordinateMappingChanged()
{
	HRESULT hr = E_UNEXPECTED;

	// Calculate the down sampled image sizes, which are used for the AlignPointClouds calculation frames
	CameraIntrinsics intrinsics = {};

	m_pMapper->GetDepthCameraIntrinsics(&intrinsics);

	float focalLengthX = intrinsics.FocalLengthX / NUI_DEPTH_RAW_WIDTH;
	float focalLengthY = intrinsics.FocalLengthY / NUI_DEPTH_RAW_HEIGHT;
	float principalPointX = intrinsics.PrincipalPointX / NUI_DEPTH_RAW_WIDTH;
	float principalPointY = intrinsics.PrincipalPointY / NUI_DEPTH_RAW_HEIGHT;

	if (m_cameraParameters.focalLengthX == focalLengthX && m_cameraParameters.focalLengthY == focalLengthY &&
		m_cameraParameters.principalPointX == principalPointX && m_cameraParameters.principalPointY == principalPointY)
		return S_OK;

	m_cameraParameters.focalLengthX = focalLengthX;
	m_cameraParameters.focalLengthY = focalLengthY;
	m_cameraParameters.principalPointX = principalPointX;
	m_cameraParameters.principalPointY = principalPointY;

	_ASSERT(m_cameraParameters.focalLengthX != 0);

	UpdateIntrinsics(m_pColorImage, &m_cameraParameters);

	if (nullptr == m_pDepthDistortionMap)
	{
		cout << "Failed to initialize Kinect Fusion depth image distortion buffer." << endl;
		return E_OUTOFMEMORY;
	}

	hr = SetupUndistortion();
	return hr;
}
bool KinectFusion::checkCoordinateChange() {
	if (nullptr == m_pNuiSensor)
	{
		cout << "cannot get kinect sensor!" << endl;
		exit(0);
	}
	//检查相机参数变化
	if (m_coordinateMappingChangedEvent != NULL && WAIT_OBJECT_0 == WaitForSingleObject((HANDLE)m_coordinateMappingChangedEvent, 0))
	{
		cout << "camere corrdinate map chainge!" << endl;
		OnCoordinateMappingChanged();
		ResetEvent((HANDLE)m_coordinateMappingChangedEvent);
		return true;
	}
	return false;
}

void KinectFusion::update() {

	if (nullptr == m_pNuiSensor)
	{
		cout << "cannot get kinect sensor!" << endl;
		return;
	}
	if (nullptr == m_pMultiFrameReader)
	{
		cout << "cannot get multiFrameReader!" << endl;
		return;
	}
	IDepthFrameReference* m_pDepthFrameReference = NULL;
	IColorFrameReference* m_pColorFrameReference = NULL;
	IInfraredFrameReference* m_pInfraredFrameReference = NULL;
	IDepthFrame* pDepthFrame = NULL;
	IColorFrame* pColorFrame = NULL;
	IInfraredFrame* pInfraredFrame = NULL;
	IMultiSourceFrame* pMultiFrame = nullptr;
	HRESULT hr = m_pMultiFrameReader->AcquireLatestFrame(&pMultiFrame);
	if (SUCCEEDED(hr)) {
		if (SUCCEEDED(hr))
			hr = pMultiFrame->get_ColorFrameReference(&m_pColorFrameReference);
		if (SUCCEEDED(hr))
			hr = m_pColorFrameReference->AcquireFrame(&pColorFrame);
		if (SUCCEEDED(hr))
			hr = pMultiFrame->get_DepthFrameReference(&m_pDepthFrameReference);
		if (SUCCEEDED(hr))
			hr = m_pDepthFrameReference->AcquireFrame(&pDepthFrame);
		if (SUCCEEDED(hr))
			hr = pMultiFrame->get_InfraredFrameReference(&m_pInfraredFrameReference);
		if (SUCCEEDED(hr))
			hr = m_pInfraredFrameReference->AcquireFrame(&pInfraredFrame);

		////读取彩色数据
		//NUI_FUSION_BUFFER *destColorBuffer = m_pColorImage->pFrameBuffer;
		//if (nullptr == pColorFrame || nullptr == destColorBuffer)
		//{
		//	cout << "create destColorBuffer error!" << endl;
		//	return;
		//}
		if (SUCCEEDED(hr)) {
			//hr = pColorFrame->CopyConvertedFrameDataToArray(m_cColorWidth * m_cColorHeight * sizeof(RGBQUAD), destColorBuffer->pBits, ColorImageFormat_Bgra);
			//for (int i = 0; i < m_cColorWidth * m_cColorHeight; i++) {
			//	i_rgb.data[i * 4] = destColorBuffer->pBits[i * 4];
			//	i_rgb.data[i * 4 + 1] = destColorBuffer->pBits[i * 4 + 1];
			//	i_rgb.data[i * 4 + 2] = destColorBuffer->pBits[i * 4 + 2];
			//	i_rgb.data[i * 4 + 3] = destColorBuffer->pBits[i * 4 + 3];
			//}
			hr = pColorFrame->CopyConvertedFrameDataToArray(m_cColorWidth * m_cColorHeight * 4, reinterpret_cast<BYTE*>(i_rgb.data), ColorImageFormat::ColorImageFormat_Bgra);
		}

		//读取深度数据
		UINT nBufferSize = 0;	//缓存帧大小
		if (SUCCEEDED(hr))
			//hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &depthData);
			//hr = pDepthFrame->CopyFrameDataToArray(m_cDepthHeight * m_cDepthWidth, reinterpret_cast<UINT16*>(i_depth.data));
			hr = pDepthFrame->CopyFrameDataToArray(m_cDepthHeight * m_cDepthWidth, depthData);

		//赋值i_depth
		if (SUCCEEDED(hr)) {
			processDepth();
			fillShowDepth();
		}
			


		//读取红外图		
		if (SUCCEEDED(hr)) {
			//hr = pInfraredFrame->AccessUnderlyingBuffer(&nBufferSize, &irData);
			hr = pInfraredFrame->CopyFrameDataToArray(m_cDepthHeight * m_cDepthWidth, reinterpret_cast<UINT16*>(i_ir.data));
		}
		
		//copy and remap depth
		const UINT bufferLength = m_cDepthImagePixels;
		UINT16 * pDepth = m_pDepthImagePixelBuffer;
		for (UINT i = 0; i < bufferLength; i++, pDepth++)
		{
			const UINT id = m_pDepthDistortionLT[i];
			*pDepth = id < bufferLength ? depthData[i] : 0;
		}	
		//将深度图映射到彩色图空间		
		if (SUCCEEDED(hr)) {
			hr = m_pMapper->MapDepthFrameToColorSpace(m_cDepthHeight * m_cDepthWidth, m_pDepthImagePixelBuffer, m_cDepthHeight * m_cDepthWidth, m_pColorCoordinates);
		}
		if (SUCCEEDED(hr)) {
			processDepthToColor();
			//imshow("rgb2depth", i_depthToRgb);
			//waitKey(1);
		}
	}
	SafeRelease(pDepthFrame);
	SafeRelease(pColorFrame);
	SafeRelease(pInfraredFrame);
	SafeRelease(m_pColorFrameReference);
	SafeRelease(m_pDepthFrameReference);
	SafeRelease(m_pInfraredFrameReference);
	SafeRelease(pMultiFrame);
}

void KinectFusion::processDepth() {
	for (int i = 0; i < m_cDepthImagePixels; i++)
	{		
		reinterpret_cast<UINT16*>(i_depth.data)[i] = static_cast<UINT16>(depthData[i]);
	}
}
void KinectFusion::fillShowDepth() {
	for (int i = 0; i < m_cDepthImagePixels; i++)
	{
		reinterpret_cast<BYTE*>(i_depth_show.data)[i] = static_cast<BYTE>(depthData[i]%256);
	}
}
void KinectFusion::processIRImage() {
	for (int i = 0; i < m_cDepthHeight; i++) {
		for (int j = 0; j < m_cDepthWidth; j++) {
			i_irShow.at<unsigned char>(i, j) = i_ir.at<unsigned short>(i, j)%256;
		}
	}
	imshow("ir8uc1", i_irShow);
	waitKey(1);
}
void KinectFusion::processDepthToColor() {
	i_depthToRgb = cv::Mat::zeros(m_cDepthHeight, m_cDepthWidth, CV_8UC4);
	for (int i = 0; i < m_cDepthHeight; i++) {
		for (int j = 0; j < m_cDepthWidth; j++) {
			unsigned int index = i * m_cDepthWidth + j;
			ColorSpacePoint p = m_pColorCoordinates[index];
			if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity()) {
				int colorX = static_cast<int>(floor(p.X + 0.5f));
				int colorY = static_cast<int>(floor(p.Y + 0.5f));
				if ((colorX >= 0 && colorX < 1920) && (colorY >= 0 && colorY < 1080))
				{
					i_depthToRgb.at<Vec4b>(i, j) = i_rgb.at<Vec4b>(colorY, colorX);
				}
			}
		}
	}
}
void KinectFusion::DepthToCloudWithMask(Mat depth, Mat mask, string fileName){

	UINT depthCount = 0;
	for (int row = 0; row < mask.rows; row++) {
		for (int col = 0; col < mask.cols; col++) {
			if (mask.at<uchar>(row, col) == 255) {
				depthCount++;
			}
		}
	}
	//cout << "count:" << depthCount << endl;
	DepthSpacePoint* dsp = new DepthSpacePoint[depthCount];
	UINT16* depthV = new UINT16[depthCount];

	int index = 0;
	for (int row = 0; row < mask.rows; row++) {
		for (int col = 0; col < mask.cols; col++) {
			if (mask.at<uchar>(row, col) == 255) {
				dsp[index].X = col;
				dsp[index].Y = m_cDepthHeight - 1 -row;
				depthV[index] = depth.at<unsigned short>(row, col);
				index++;
			}
		}
	}
	CameraSpacePoint* csp = new CameraSpacePoint[depthCount];
	HRESULT hr = m_pMapper->MapDepthPointsToCameraSpace(depthCount, dsp, depthCount, depthV, depthCount, csp);
	if (SUCCEEDED(hr))			
	{
		int count = 0;
		for (int i = 0; i < depthCount; i++)
		{
			CameraSpacePoint p = csp[i];
			if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
			{
				count++;
			}
		}
		ofstream ofs(fileName);
		string num;
		stringstream ss;
		ss << count;
		ss >> num;
		string str = "ply\nformat ascii 1.0\nelement face 0\n property list uchar int vertex_indices\nelement vertex " + string(num) + "\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\nproperty uchar alpha\nend_header\n";
		ofs << str;
		for (int i = 0; i < depthCount; i++)
		{
			CameraSpacePoint p = csp[i];
			if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
			{
				float cameraX = static_cast<float>(p.X);
				float cameraY = static_cast<float>(p.Y);
				float cameraZ = static_cast<float>(p.Z);
				ofs << cameraX << ' ' << cameraY << ' ' << cameraZ << ' ' << static_cast<int>(0) << ' ' << static_cast<int>(0) << ' ' << static_cast<int>(0) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << endl;
			}
		}
		ofs.close();
		//cout << "ok" << endl;
	}
}
void KinectFusion::DepthToCloud(Mat depth, string fileName) {

	int depthCount = depth.cols * depth.rows;
	DepthSpacePoint* dsp = new DepthSpacePoint[depthCount];
	UINT16* depthV = new UINT16[depthCount];

//	cv::flip(depth, depth, 1);
	cv::flip(depth, depth, 0);
	for (int row = 0; row < depth.rows; row++) {
		for (int col = 0; col < depth.cols; col++) {
			unsigned index = row * depth.cols + col;
			dsp[index].X = col;
			dsp[index].Y = row;
			depthV[index] = depth.at<unsigned short>(row, col);
		}
	}

	CameraSpacePoint* csp = new CameraSpacePoint[depthCount];
	HRESULT hr = m_pMapper->MapDepthPointsToCameraSpace(depthCount, dsp, depthCount, depthV, depthCount, csp);

	int count = 0;
	for (int i = 0; i < m_cDepthImagePixels; i++)
	{
		CameraSpacePoint p = csp[i];
		if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
		{
			count++;
		}
	}
	ofstream ofs;
	ofs.open(fileName);
	if (ofs.is_open()) {
		ofs << "ply\n";
		ofs << "format ascii 1.0\n";
		ofs << "element vertex " << count << "\n";
		ofs << "property float x\n";
		ofs << "property float y\n";
		ofs << "property float z\n";
		ofs << "property float nx\n";
		ofs << "property float ny\n";
		ofs << "property float nz\n";
		ofs << "property uchar diffuse_red\n";
		ofs << "property uchar diffuse_green\n";
		ofs << "property uchar diffuse_blue\n";
		ofs << "property uchar alpha\n";
		ofs << "end_header\n";
		for (int i = 0; i < m_cDepthImagePixels; i++)
		{
			CameraSpacePoint p = csp[i];
			if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
			{
				float cameraX = static_cast<float>(p.X);
				float cameraY = static_cast<float>(p.Y);
				float cameraZ = static_cast<float>(p.Z);
				ofs << cameraX << " " << cameraY << " " << cameraZ << " ";
				ofs << "0 0 0 ";
				ofs << "255 255 255 ";
				ofs << "255" << endl;
			}
		}
		ofs.close();
		cout << "depthtocloud ok!" << endl;
	}
	else {
		cout << "DepthToCloud: open file error!" << endl;
		return;
	}
}

void GetMaskedCloud(KinectFusion kf, string fileName) {
	//KinectFusion kf;
	//HRESULT hr = kf.CreateFirstConnected();
	//if (FAILED(hr)) {
	//	cout << "CreateFirstConnected error !" << endl;
	//	return 0;
	//}
	//hr = kf.InitializeKinectFusion();
	//if (FAILED(hr)) {
	//	cout << "InitializeKinectFusion error!" << endl;
	//	return 0;
	//}
	//while (!kf.checkCoordinateChange());
	Mat mask = imread(".\\mask.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat depth = imread(".\\depth.png", CV_LOAD_IMAGE_ANYDEPTH);
	CV_Assert(!mask.empty());
	CV_Assert(!depth.empty());

	kf.DepthToCloudWithMask(depth, mask, fileName);
	cout << "GetMaskedCloud over!" << endl;
}

void WriteToFile(const Mat r, const Mat t, const string fileName){
	if (r.rows != 3 || r.cols != 3 || r.type() != CV_64FC1
		|| t.rows != 3 || t.cols != 1 || t.type() != CV_64FC1)
	{
		std::cout << "WriteToFile: check inputs!(r, t format error!)" << std::endl;
		exit(0);
	}
	ofstream ofs;
	ofs.open(fileName);
	if (ofs.is_open()) {
		ofs << r.at<double>(0, 0) << " " << r.at<double>(0, 1) << " " << r.at<double>(0, 2) << endl;
		ofs << r.at<double>(1, 0) << " " << r.at<double>(1, 1) << " " << r.at<double>(1, 2) << endl;
		ofs << r.at<double>(2, 0) << " " << r.at<double>(2, 1) << " " << r.at<double>(2, 2) << endl;
		ofs << t.at<double>(0, 0) << " " << t.at<double>(1, 0) << " " << t.at<double>(2, 0) << endl;
	}
	else{
		cout << "WriteToFile: open file error!" << endl;
		exit(0);
	}
}


void savePointCloud(const vector<cv::Point3f>& Pts, string fileName) {

	int count = Pts.size();

	ofstream ofs(fileName);
	string num;
	stringstream ss;
	ss << count;
	ss >> num;
	string str = "ply\nformat ascii 1.0\nelement face 0\n property list uchar int vertex_indices\nelement vertex " + string(num) + "\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar diffuse_red\nproperty uchar diffuse_green\nproperty uchar diffuse_blue\nproperty uchar alpha\nend_header\n";
	ofs << str;
	for (int i = 0; i < count; i++)
	{
		ofs << Pts[i].x << ' ' << Pts[i].y << ' ' << Pts[i].z << ' ' << static_cast<int>(0) << ' ' << static_cast<int>(0) << ' ' << static_cast<int>(0) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << ' ' << static_cast<int>(255) << endl;
	}
	ofs.close();
	cout << "ok" << endl;
}
void DepthToWorld(Mat R, Mat T, string fileName) {

	//read cloud file
	vector<cv::Point3f> Pts;
	ifstream ifs(fileName);
	while (!ifs.eof()) {
		string str;
		getline(ifs, str);
		if (str == "end_header") {
			break;
		}
	}
	while (!ifs.eof()) {
		float x, y, z;
		ifs >> x >> y >> z;
		Point3f pf(x, y, z);
		Pts.push_back(pf);
		string str;
		getline(ifs, str);
	}
	ifs.close();

	Mat org(3, 1, CV_64FC1);
	vector<cv::Point3f> Pts2;
	for (int i = 0; i < Pts.size(); i++) {
		org.at<double>(0, 0) = Pts[i].x;
		org.at<double>(1, 0) = Pts[i].y;
		org.at<double>(2, 0) = Pts[i].z;
		Mat res = R.inv() * (org - T);
		//Mat res = R * org + T;
		Pts2.push_back(Point3f(res.at<double>(0, 0), res.at<double>(1, 0), res.at<double>(2, 0)));
	}
	savePointCloud(Pts2, "check2.ply");
}

void ReadRAndT(Mat&R, Mat&T, const string fileName){
	R.create(3, 3, CV_64FC1);
	T.create(3, 1, CV_64FC1);
	ifstream ifs;
	ifs.open(fileName);
	if (ifs.is_open()) {
		for (int i = 0; i < 3; i++) {
			double a, b, c;
			ifs >> a >> b >> c;
			//cout << a << ",  " << b << ",  " << c << endl;
			R.at<double>(i, 0) = a; R.at<double>(i, 1) = b; R.at<double>(i, 2) = c;
		}
		float a, b, c;
		ifs >> a >> b >> c;
		//cout << a << ",  " << b << ",  " << c << endl;
		T.at<double>(0, 0) = a; T.at<double>(1, 0) = b; T.at<double>(2, 0) = c;
		ifs.close();
		R = R.t();
	}
	else {
		cout << "ReadRAndT: open file error!" << endl;
		exit(0);
	}
}


void CalibrateRT(string fileName) {
	//Mat img = imread(fileName, CV_LOAD_IMAGE_COLOR);
	//CV_Assert(!img.empty());
	//cvtColor(img, imageGray, CV_RGB2GRAY);
	//imshow("img", img);
	//waitKey(100);

	vector<Point2f> corners;
	/* 提取角点 */
	Mat imageGray = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
	CV_Assert(!imageGray.empty());
	//equalizeHist(imageGray, imageGray);
	

	bool patternfound = findChessboardCorners(imageGray, Size(8, 14), corners);// +CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	if (patternfound)
		cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1)); // cvFindChessboardCorners找到的角点仅仅是近似值，必须调用此函数达到亚像素精度，如果第一次定位...
	else {
		cout << "calibrateRT:can not find correct corner points!" << endl;
		//return;
	}
	cout << "corners size:" << corners.size() << endl;
	//corners.resize(9);
	//Draw corners  
	drawChessboardCorners(imageGray, Size(8, 14), corners, patternfound);//found为cvFindChessboardCorners的返回值  
	imshow("points", imageGray);
	waitKey(0);
	const float square_size = 0.06;			//60mm
	vector<Point2f> imagePoints;
	vector<Point3f> worldPoint;
	for (int row = 0; row < 2; row++) {
		for (int col = 0; col < 2; col++){
			int index = row * 8 + col;
			imagePoints.push_back(corners[index]);
			Point3f tempPoint;
			tempPoint.x = col * square_size;
			tempPoint.y = row * square_size;
			tempPoint.z = 0;
			worldPoint.push_back(tempPoint);
		}
	}
	Mat k = Mat::zeros(3, 3, CV_32FC1);;
	k.at<float>(0, 0) = 364.629913f;
	k.at<float>(1, 1) = 364.629913f;
	k.at<float>(0, 2) = 256;// 258.836090f;
	k.at<float>(1, 2) = 212;// 201.942505f;
	k.at<float>(2, 2) = 1;
	cv::Mat disCoeffs = Mat::zeros(4, 1, CV_64FC1);
	cv::Mat r = Mat::zeros(3, 1, CV_64FC1);
	cv::Mat t = Mat::zeros(3, 1, CV_64FC1);
	Mat inliner;
	solvePnPRansac(worldPoint, imagePoints, k, disCoeffs, r, t, false, 100, 8.0f, 100, inliner, ITERATIVE);
	cout << inliner << endl;
	Mat R;
	Rodrigues(r, R);
	WriteToFile(R, t, "RT.txt");	
	DepthToWorld(R, t, ".\\depthCloudTest.ply");
	
}

Mat image;
string outmaskName = ".\\mask.png";
int threshval = 160;
void on_trackbar(int, void*)
{
	const int binThreshold = 163;
	//Mat image = imread(irImgPath, CV_LOAD_IMAGE_GRAYSCALE);
	//CV_Assert(!image.empty());
	equalizeHist(image, image);
	vector<vector<Point> > contours;
	//这句相当于二值化;  
	Mat bimage = image > threshval;
	//threshold(image, bimage, sliderPos, 255,CV_THRESH_BINARY);  

	//提取轮廓,相当于matlab中连通区域分析  
	findContours(bimage, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	for (size_t i = 0; i < contours.size(); i++)
	{
		//我们将在cimage上面绘图  
		Mat cimage = Mat::zeros(bimage.size(), CV_8UC1);
		//轮廓的边缘点个数  
		size_t count = contours[i].size();
		//Fitzgibbon的椭圆拟合方法，要求至少6个点，文献：Direct Least Squares Fitting of Ellipses[1999]  
		if (count < 60)//500 || count > 1000)
			continue;

		//将轮廓中的点转换为以Mat形式存储的2维点集(x,y)  
		Mat pointsf;		
		Mat(contours[i]).convertTo(pointsf, CV_32F);

		//最小二次拟合（Fitzgibbon的方法）; box包含了椭圆的5个参数：(x,y,w,h,theta)  
		RotatedRect box = fitEllipse(pointsf);

		//把那些长轴与短轴之比很多的那些椭圆剔除。  
		if (MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height) * 3)
			continue;	
		//剔除那些长轴还不到图像宽的1/3的椭圆
		//if (MAX(box.size.width, box.size.height) < (image.cols/3))
		//	continue;
		//剔除那些拟合的椭圆边界不在图像里的椭圆结果
		if (MAX(box.size.width, box.size.height) >= image.cols || MAX(box.size.width, box.size.height) >= image.rows)
			continue;
		//成员函数points 返回 4个矩形的顶点(x,y)  
		Point2f vtx[4];		
		box.points(vtx);
		bool flag = false;
		for (int i = 0; i < 4; i++) {
			if (vtx[i].x > image.cols || vtx[i].x < 0 || vtx[i].y > image.rows || vtx[i].y < 0) {
				flag = true;
				break;
			}
		}
		if (flag)
			continue;
		//填充轮廓  
		//drawContours(cimage, contours, (int)i, Scalar::all(255), 1, 8);
		drawContours(cimage, contours, (int)i, Scalar::all(255), CV_FILLED, 8);
		//pointPolygonTest,判断点是否在轮廓内
		//imshow("lunk", cimage);
		//waitKey(0);
		//绘制椭圆  
		//ellipse(cimage, box, Scalar::all(255), 1, CV_AA);
		//绘制椭圆  
		//ellipse(cimage, box.center, box.size*0.5f, box.angle, 0, 360, Scalar(0,255,255), 1, CV_AA); 
		int inliers = cv::countNonZero(cimage);
		if (inliers < 1000)
			continue;
		imwrite(outmaskName, cimage);
		//显示窗口  
		imshow("Connected Components", cimage);
		return;
		////绘制矩形框  
		//Point2f vtx[4];
		////成员函数points 返回 4个矩形的顶点(x,y)  
		//box.points(vtx);
		//for (int j = 0; j < 4; j++)
		//	line(cimage, vtx[j], vtx[(j + 1) % 4], Scalar(0, 255, 0), 1, CV_AA);	
	}	
	//return;
}
bool GetMaskImage(string irImgPath, string maskName)
{
	//载入图片  
	image = imread(irImgPath, 0);
	if (!image.data) { cout << "read image failed !" << endl; return false; }
	outmaskName = maskName;

	//创建处理窗口  
	namedWindow("Connected Components", WINDOW_AUTOSIZE);
	//创建轨迹条  
	createTrackbar("Threshold", "Connected Components", &threshval, 255, on_trackbar);
	on_trackbar(threshval, 0);//轨迹条回调函数  
	waitKey(0);
	return true;
}
int  main() {
	KinectFusion kf;
	HRESULT hr = kf.CreateFirstConnected();
	if (FAILED(hr)) {
		cout << "111" << endl;
		return 0;
	}
	hr = kf.InitializeKinectFusion();
	if (FAILED(hr)) {
		cout << "1112" << endl;
		return 0;
	}
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(1);    // 无压缩png.
	while(!kf.checkCoordinateChange());
	while (1) {
		cout << "press select function:\n	press'1': data collection\n	press'2': get mask image(make sure depth image exist)\n	press'3': depth cloud(make sure depth image exist)\n	press'4': cloud with mask(make sure mask、depth images exist)\n	press'9': show depth、ir、color image\n	press '0': eixt" << endl;
		string str;
		getline(cin, str);
		if (str == "1"){
			cout << "press Enter to start collection depth and ir image (without object on it)" << endl;
			getchar();
			//get orig depth and ir image
			while (1) {
				kf.update();
				imwrite(".\\color.png", kf.getColorImg(), compression_params);
				imwrite(".\\depth.png", kf.getDepthImg(), compression_params);
				imwrite(".\\mask.png", kf.getIRImg(), compression_params);
				break;
			}
			cout << "depth and ir image acquired!\n" << endl;
			cout << "press 's' to start collection depth、 color and ir image((with object on it));\n press 'r' return top layer\n" << endl;
			string ss;
			getline(cin, ss);
			if (ss == "s") {
				int count = 0;
				clock_t start, end;
				start = clock();
				while (1) {
					kf.update();
					imshow("color", kf.getColorImg());
					imshow("depth", kf.getDepthImg());
					imshow("ir", kf.getIRImg());
					if (waitKey(1) == 32) 
						break;

					stringstream ss;
					ss << count << endl;
					string out;
					ss >> out;
					while (out.length() < 4) {
						out = "0" + out;
					}
					imwrite(".\\depth_" + out + ".png", kf.getDepthImg(), compression_params);
					imwrite(".\\color_" + out + ".png", kf.getColorImg(), compression_params);
					imwrite(".\\ir_" + out + ".png", kf.getIRImg(), compression_params);
					count++;
					//end = clock();
					//double dur = (double)(end - start);
					//if (static_cast<int>(dur / CLOCKS_PER_SEC) >= 3) {
					//	cout << "seconds:" << static_cast<int>(dur / CLOCKS_PER_SEC) << endl;
					//	break;
					//}
				}
				cv::destroyAllWindows();
				cout << "data collection finished!\n" << endl;
			}
			else if (ss == "r") {
				continue;
			}	
		}
		else if (str == "2") {
			if (GetMaskImage(".\\ir.png", ".\\mask.png"))
				cout << "get mask image ok!" << endl;
			else
				cout << "cannot get mask image!" << endl;
		}
		else if (str == "3") {
			Mat depth = imread(".\\DepthImage3.png", CV_LOAD_IMAGE_UNCHANGED);
			CV_Assert(!depth.empty());
			kf.DepthToCloud(depth, ".\\depthCloudTest.ply");
		}
		else if (str == "4") {
			GetMaskedCloud(kf, ".\\cloudWithMask.ply");
		}
		else if (str == "5") {
			CalibrateRT("IrImage3.bmp");
		}
		else if (str == "9") {
			while (1) {
				kf.update();
				//imshow("color", kf.getColorImg());
				//imshow("depth", kf.getDepthImg());
				//imshow("ir", kf.getIRImg());
				imshow("showdepth", kf.getShowDepth());
				imshow("rgb2depth", kf.getDepthToRgb());
				if (waitKey(1) == 32) {
					break;
				}			
			}
			cv::destroyAllWindows();
		}
		else if (str == "0") {
			break;
		}
	}
	return 0;
	
	//
}

