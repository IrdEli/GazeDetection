using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.Azure.Kinect.BodyTracking;
using Microsoft.Azure.Kinect.Sensor;
using System;
//using UnityEngine.UI;
//using UnityEngine.UIElements;
using System.Net.Sockets;
using System.IO;
using System.Text;
using System.Globalization;
using System.Threading;

public class StreamPythonTasks : MonoBehaviour
{
    Device kinect;
    Tracker bodyTracker;
    [SerializeField] GameObject x;
    [SerializeField] UnityEngine.UI.Text robAerror;
    [SerializeField] UnityEngine.UI.Text robBerror;
    Texture2D kinectColorTexture;
    [SerializeField] UnityEngine.UI.RawImage rawColorImg;


    [SerializeField] GameObject kinectDevice;
    [SerializeField] GameObject RobA;
    [SerializeField] GameObject RobB;
    [SerializeField] GameObject WorldZero;

    [SerializeField] Vector3 rotFixed = new Vector3(0, 0, 0);
    static Byte[] Buffer { get; set; }
    static Socket socket;
    NetworkStream stream;
    Vector3 gazeVector;
    private Vector3 gazeVectorWorld;
    private Quaternion quatHead;
    private Vector3 headJoinPos;

    private Vector3 headPosUC;
    private Vector3 headPosWC;
    
    private Quaternion quatHeadWC;
    
    
    private float errorA;
    private float errorB;
    private Calibration calibration;
    private Transformation transformation;

    private int frame = 0;
    private string gazeVectorString = "Empty";
    
    private Thread receiveThread;
    private Thread sendThread;

    Queue<byte[]> captures = new Queue<byte[]>();
    private int colorWidth;
    private int colorHeight;
    private System.Numerics.Vector2 prevHeadPos2D;

    void Awake()
    {
        InitKinect();
        Debug.Log("Waiting for Connection");
        TcpClient client = new TcpClient("192.168.3.68", 8000);//"192.168.3.68"
        Debug.Log("Connected");
        stream = client.GetStream();
        Debug.Log("Stream Acquired");


        receiveThread = new Thread(new ThreadStart(recvData));
        receiveThread.Start();
        sendThread = new Thread(new ThreadStart(sendData));
        sendThread.Start();
        //RobA.transform.position = WorldZero.transform.InverseTransformPoint(RobA.transform.position);
        //RobB.transform.position = WorldZero.transform.InverseTransformPoint(RobB.transform.position);
    }

    private void InitKinect()
    {
        Instantiate(x, new Vector3(0, 0, 0), new Quaternion(0, 0, 0, 0));
        kinect = Device.Open(0);
        var configuration = new DeviceConfiguration
        {
            ColorFormat = ImageFormat.ColorBGRA32,
            CameraFPS = FPS.FPS30,
            DepthMode = DepthMode.WFOV_2x2Binned,
            SynchronizedImagesOnly = true,
            ColorResolution = ColorResolution.R720p

        };
        kinect.StartCameras(configuration);
        colorHeight = kinect.GetCalibration().ColorCameraCalibration.ResolutionHeight;
        colorWidth = kinect.GetCalibration().ColorCameraCalibration.ResolutionWidth;

        prevHeadPos2D.X = colorWidth / 2;
        prevHeadPos2D.Y = colorHeight / 2;

        calibration = kinect.GetCalibration(configuration.DepthMode, configuration.ColorResolution);
        transformation = calibration.CreateTransformation();
        kinectColorTexture = new Texture2D(colorWidth, colorHeight);
        bodyTracker = Tracker.Create(kinect.GetCalibration(), new TrackerConfiguration
        {
            ProcessingMode = TrackerProcessingMode.Cuda,
            SensorOrientation = SensorOrientation.Default
        });

    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;
        //Gizmos.DrawRay(eye)
        Gizmos.DrawLine(new Vector2(x.transform.position.y, x.transform.position.z), new Vector2(RobA.transform.position.y, RobA.transform.position.z));
        //Gizmos.DrawLine(x.transform.position, RobA.transform.position);
        Gizmos.color = Color.black;
        Gizmos.DrawLine(new Vector2(x.transform.position.y, x.transform.position.z), new Vector2(RobB.transform.position.y, RobB.transform.position.z));
        //Gizmos.DrawLine(x.transform.position, RobB.transform.position); ;
        //Gizmos.DrawLine(startPoint, endPoint);
        //Gizmos.DrawSphere(startPoint, lineWidth);
        //Gizmos.DrawSphere(endPoint, lineWidth);
    }
    void OnDestroy()
    {
        kinect.StopCameras();
        //receiveThread.Suspend();
    }

    Texture2D ExtractRectangle(Texture2D image, int x, int width, int y, int height)
    {
        var xMin = Math.Max(x - width / 2, 0);
        var yMin = Math.Max(y - height / 2, 0);
        var xMax = Math.Min(x + width / 2, image.width);
        var yMax = Math.Min(y + height / 2, image.height);
        Rect sourceRect = new Rect(xMin, yMin, Math.Abs(xMax-xMin), Math.Abs(yMax-yMin));

        //Debug.Log("Rect" + sourceRect.ToString());
        // Create a new Texture2D object to hold the copied pixels
        Texture2D newTexture = new Texture2D((int)sourceRect.width, (int)sourceRect.height);

        // Get the pixels from the source texture within the specified area
        Color[] sourcePixels = image.GetPixels((int)sourceRect.x, (int)sourceRect.y, (int)sourceRect.width, (int)sourceRect.height);

        // Set the pixels in the new texture
        newTexture.SetPixels(sourcePixels);

        // Apply the changes to the new texture
        newTexture.Apply();

        return newTexture;
    }

    Texture2D drawRectangle(Texture2D image, int x, int width, int y, int height)
    {
        //Debug.Log("Image Width: " + image.width.ToString());
        //Debug.Log("Image Height" + image.height.ToString());
        var xMin = Math.Max(x - width / 2, 0);
        var yMin = Math.Max(y - height / 4, 0);
        var xMax = Math.Min(x + width / 2, image.width-1);
        var yMax = Math.Min(y + 3*height / 4, image.height-1);

        Debug.Log("ArrayRange: " + xMin.ToString()+ " "+yMin.ToString() + " " + xMax.ToString()+ " " + yMax.ToString());
        //Color32[] colors = new Color32[xMax-xMin * yMax-yMin];

        // Fill the array with red color
        //Color32 red = new Color32(255, 0, 0, 255); // r=255, g=0, b=0, a=255 (fully opaque)
        for (int i = xMin; i < xMax; i++)
        {
            for(int j = yMin; j< yMax; j++)
            {
                kinectColorTexture.SetPixel(i, j, Color.black);
            }
        }
        image.Apply();
        return image;
    }
    void Update()
    {
        ///////////////////
        ///Check out queueing thing again
        /// 
        /// 
        /// 
        /// 
        //////////////////
        //Debug.Log("Queue COunt:"+ captures.Count.ToString());
        //var prevWorldPos = WorldZero.transform.position;
        //gameObjectPosInfo.text = x.transform.position.ToString();
        using (Image depthImage = new Image(Microsoft.Azure.Kinect.Sensor.ImageFormat.Depth16, colorWidth, colorHeight, colorWidth * sizeof(UInt16)))
        using (Capture capture = kinect.GetCapture())
        {
            //captures.Enqueue(capture);
            bodyTracker.EnqueueCapture(capture);
            //Debug.Log(capture.Color.DeviceTimestamp);
            var frame = bodyTracker.PopResult();
            Image colorImg = capture.Color;
            transformation.DepthImageToColorCamera(capture, depthImage);
            //var pixel = colorImg.GetPixel((int)(0), (int)5);
            Color32[] pixels = colorImg.GetPixels<Color32>().ToArray();
            kinectColorTexture.SetPixels32(pixels);
            kinectColorTexture.Apply();
            //rawColorImg.texture = kinectColorTexture;
            
            if (frame.NumberOfBodies > 0)
            {
                var body = frame.GetBody(0);
                var headoint = body.Skeleton.GetJoint(JointId.Nose);

                headJoinPos = new Vector3(headoint.Position.X / 50, headoint.Position.Y / 50, headoint.Position.Z / 50);
                var headPos2D = calibration.TransformTo2D(headoint.Position, CalibrationDeviceType.Depth, CalibrationDeviceType.Color);
                //Debug.Log("Head Pos 2D"+headPos2D.ToString());
                //Debug.Log(depthImage.WidthPixels.ToString()+ depthImage.HeightPixels.ToString());
                float refBox = 150;
                if (headPos2D.HasValue)
                {
                    prevHeadPos2D = headPos2D.Value;
                    //Debug.Log("2D point Value:" + headPos2D.Value.ToString());
                    float depthPos2D = depthImage.GetPixel<ushort>((int)Math.Max(headPos2D.Value.Y,0), (int)Math.Max(headPos2D.Value.X,0));
                    Debug.Log("Depth ppoint found before division:" +depthPos2D.ToString());
                    depthPos2D = (depthPos2D/1000);
                    //Debug.Log("Depth ppoint found after division:" + depthPos2D.ToString());
                    //depthPos2D = 1;
                    
                    float width = refBox /depthPos2D;
                    float height = (refBox /depthPos2D);
                    Debug.Log("width with value:" + width.ToString());
                    Debug.Log("Height with value:" + height.ToString());
                    //Debug.Log("Depth" + depthPos2D.ToString());
                    if(depthPos2D != 0)
                    {
                        Texture2D updatedKinectColorTexture = ExtractRectangle(kinectColorTexture, (int)headPos2D.Value.X, (int)(width), (int)headPos2D.Value.Y, (int)(height));
                        rawColorImg.texture = updatedKinectColorTexture;
                        byte[] imageData = updatedKinectColorTexture.EncodeToJPG();
                        captures.Enqueue(imageData);


                    }
                    else
                    {
                        Texture2D updatedKinectColorTexture = ExtractRectangle(kinectColorTexture, (int)headPos2D.Value.X, (int)(refBox), (int)headPos2D.Value.Y, (int)(refBox));
                        rawColorImg.texture = updatedKinectColorTexture;
                        byte[] imageData = updatedKinectColorTexture.EncodeToJPG();
                        captures.Enqueue(imageData);


                    }

                }
                else
                {
                    
                    //Debug.Log("2D point Value:" + headPos2D.Value.ToString());
                    float depthPos2D = depthImage.GetPixel<ushort>((int)Math.Max(prevHeadPos2D.Y, 0), (int)Math.Max(prevHeadPos2D.X, 0));
                    depthPos2D = (depthPos2D / 1000);
                    Debug.Log("Depth point not found:" +depthPos2D.ToString());
                    //depthPos2D = 1;
                    
                    float width = refBox /depthPos2D;
                    float height = (refBox /depthPos2D);

                    Debug.Log("width without value:" + width.ToString());
                    Debug.Log("Height without value:" + height.ToString());
                    //Debug.Log("Depth" + depthPos2D.ToString());
                    if (depthPos2D != 0)
                    {
                        Texture2D updatedKinectColorTexture = ExtractRectangle(kinectColorTexture, (int)prevHeadPos2D.X, (int)width, (int)prevHeadPos2D.Y, (int)(height));
                        rawColorImg.texture = updatedKinectColorTexture;
                        byte[] imageData = updatedKinectColorTexture.EncodeToJPG();
                        captures.Enqueue(imageData);


                    }
                    else
                    {
                        Texture2D updatedKinectColorTexture = ExtractRectangle(kinectColorTexture, (int)prevHeadPos2D.X, (int)(refBox), (int)prevHeadPos2D.Y, (int)(refBox));
                        rawColorImg.texture = updatedKinectColorTexture;
                        byte[] imageData = updatedKinectColorTexture.EncodeToJPG();
                        captures.Enqueue(imageData);


                    }
                }


               
                //Debug.Log("Head Pos 2D:" + headPos2D.ToString());
                //headJoinPos = kinectDevice.transform.TransformPoint(headJoinPos);
                //headJoinPos = WorldZero.transform.InverseTransformPoint(headJoinPos);
                quatHead = new Quaternion(headoint.Quaternion.X, headoint.Quaternion.Y, headoint.Quaternion.Z, headoint.Quaternion.W);

                var u = quatHead.eulerAngles;
                quatHead = Quaternion.Euler(u.x, u.y + 180, u.z);//(u.x, u.y + 180, u.z)

                //x.transform.position = new Vector3(headJoinPos.x, -headJoinPos.y, headJoinPos.z);
                //x.transform.rotation = quatHead;

                Transform sourceTransform = kinectDevice.transform;
                Transform destinationTransform = WorldZero.transform;

                headPosUC = sourceTransform.TransformPoint(headJoinPos);
                headPosWC = destinationTransform.InverseTransformPoint(headPosUC);
                
                // need to fix the rotation in 3d global and other coordinates
                quatHeadWC = Quaternion.Inverse(destinationTransform.rotation) * (sourceTransform.rotation * quatHead);
                
                x.transform.position = headPosWC;
                x.transform.rotation = quatHeadWC;

              
                string[] values = gazeVectorString.Split(new char[0], StringSplitOptions.RemoveEmptyEntries);

                float i = 0.0f;
                float j = 0.0f;
                float k = 0.0f;
                
                if (values.Length > 0)
                {
                    //Debug.Log("String Value = " + gazeVectorString);
                    if (float.TryParse(values[0], out i) && values.Length == 3)
                    {
                        i = float.Parse(values[0]);
                        j = float.Parse(values[1]);
                        k = float.Parse(values[2]);
                        gazeVector = new Vector3(-i, -j, k);
                        //Debug.DrawRay(x.transform.position, gazeVector * 1000, Color.green, 0.5f);
                        gazeVector = kinectDevice.transform.TransformDirection(gazeVector);
                        //Debug.DrawRay(x.transform.position, gazeVector * 1000, Color.yellow, 0.5f);
                        gazeVectorWorld = WorldZero.transform.InverseTransformVector(gazeVector);
                        Debug.DrawRay(x.transform.position, gazeVector * 1000, Color.red, 0.5f);
                    }

                }

               // Debug.Log(gazeVectorString);

                //RobA.transform.rotation = Quaternion.Inverse(WorldZero.transform.rotation)*(RobA.transform.rotation * );
                //Debug.Log(kinectDevice.transform.position);
                errorA = Vector2.Angle(new Vector2(x.transform.position.x,x.transform.position.z) - new Vector2(RobA.transform.position.x,RobA.transform.position.z),
                                        new Vector2(gazeVector.x, gazeVector.z));
                errorB = Vector3.Angle(new Vector2(x.transform.position.x, x.transform.position.z) - new Vector2(RobB.transform.position.x, RobB.transform.position.z),
                                        new Vector2(gazeVector.x, gazeVector.z));
                robAerror.text = errorA.ToString();
                robBerror.text = errorB.ToString();
                //Debug.DrawRay(new Vector3(0, 0, 0), WorldZero.transform.InverseTransformPoint(RobA.transform.position) - headPosWC, Color.cyan);
                // Debug.DrawRay( new Vector3(0, 0, 0), gazeVector, Color.cyan);

                
            }
        }

    }


    private void sendData()
    {
        while (true)
        {
           
            if (captures.TryDequeue(out byte[] imageData))
            {
                byte[] dataLength = BitConverter.GetBytes(imageData.Length);
                stream.Write(dataLength, 0, dataLength.Length);
                int bytesSent = 0;
                int bytesLeft = imageData.Length;

                //Loop to send the image data
                while (bytesLeft > 0)
                {
                    int packetSize = Math.Min(1350000, bytesLeft);
                    //Debug.Log("Packet Size"+ packetSize.ToString());
                    stream.Write(imageData, bytesSent, packetSize);
                    bytesSent += packetSize;
                    bytesLeft -= packetSize;
                    //Debug.Log("Bytes Left"+ bytesLeft.ToString());

                }
            }
            else
            {
                Thread.Sleep(10);
            }
        }
    }
    
    private void recvData()
    {
        while (true)
        {
           // Debug.Log("rec Thread" + gazeVectorString);
            byte[] buffer = new byte[1024];
            int bytesRead = 0;
            int totalBytesRead = 0;
            string reply = "";
            while (true)
            {
                bytesRead = stream.Read(buffer, 0, buffer.Length);
                totalBytesRead += bytesRead;
                reply += Encoding.ASCII.GetString(buffer, 0, bytesRead);
                if (totalBytesRead == reply.Length)
                    break;
            }

            reply = reply.Replace("[", "");
            reply = reply.Replace("]", "");
            reply = reply.Replace("'", "");
            gazeVectorString = reply;
        }
        
    }

    
}
