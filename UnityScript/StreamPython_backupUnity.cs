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

public class StreamPython : MonoBehaviour
{
    Device kinect;
    Tracker bodyTracker;
    [SerializeField]GameObject x;
    [SerializeField] UnityEngine.UI.Text robAerror;
    [SerializeField] UnityEngine.UI.Text robBerror;
    Texture2D kinectColorTexture;
    [SerializeField] UnityEngine.UI.RawImage rawColorImg;


    [SerializeField] GameObject kinectDevice;
    [SerializeField] GameObject RobA;
    [SerializeField] GameObject RobB;
    [SerializeField] GameObject WorldZero;

    [SerializeField] Vector3 rotFixed = new Vector3(0,0,0);
    static Byte[] Buffer { get; set; }
    static Socket socket;
    NetworkStream stream;
    Vector3 gazeVector;
    private Quaternion quatHead;
    private Vector3 headJoinPos;
    private Vector3 headPosWC;
    private Quaternion quatHeadWC;
    private float errorA;
    private float errorB;
    private Calibration calibration;
    private Transformation transformation;

    private int frame = 0;
    

    void Awake()
    {
        InitKinect();
        Debug.Log("Waiting for Connection");
        TcpClient client = new TcpClient("192.168.3.68", 9000);//"192.168.3.68"
        Debug.Log("Connected");
        stream = client.GetStream();
        Debug.Log("Stream Acquired");
       
        //RobA.transform.position = WorldZero.transform.InverseTransformPoint(RobA.transform.position);
        //RobB.transform.position = WorldZero.transform.InverseTransformPoint(RobB.transform.position);
    }

    private void InitKinect()
    {
        Instantiate(x, new Vector3(0,0,0), new Quaternion(0,0,0,0));
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
        int height = kinect.GetCalibration().ColorCameraCalibration.ResolutionHeight;
        int width = kinect.GetCalibration().ColorCameraCalibration.ResolutionWidth;


        calibration = kinect.GetCalibration(configuration.DepthMode, configuration.ColorResolution);
        transformation = calibration.CreateTransformation();
        kinectColorTexture = new Texture2D(width, height);
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
        Gizmos.DrawLine(x.transform.position, RobA.transform.position);
        Gizmos.color = Color.black;
        Gizmos.DrawLine(x.transform.position, RobB.transform.position); ;
        //Gizmos.DrawLine(startPoint, endPoint);
        //Gizmos.DrawSphere(startPoint, lineWidth);
        //Gizmos.DrawSphere(endPoint, lineWidth);
    }
    void OnEditorClose()
    {
        kinect.StopCameras();
    }

    void Update()
    {
        
        var prevWorldPos = WorldZero.transform.position;
        //gameObjectPosInfo.text = x.transform.position.ToString();
        using (Capture capture = kinect.GetCapture())
        {

            bodyTracker.EnqueueCapture(capture);
            //Debug.Log(capture.Color.DeviceTimestamp);
            var frame = bodyTracker.PopResult();
            Image colorImg = capture.Color;
            //var pixel = colorImg.GetPixel((int)(0), (int)5);
            Color32[] pixels = colorImg.GetPixels<Color32>().ToArray();
            kinectColorTexture.SetPixels32(pixels);
            kinectColorTexture.Apply();
            rawColorImg.texture = kinectColorTexture;

            if (frame.NumberOfBodies>0)
            {
                var body = frame.GetBody(0);
                var headoint = body.Skeleton.GetJoint(JointId.Nose);

                headJoinPos = new Vector3(headoint.Position.X / 50, headoint.Position.Y / 50, headoint.Position.Z / 50);
                var headPos2D = calibration.TransformTo2D(headoint.Position, CalibrationDeviceType.Depth, CalibrationDeviceType.Color);
                //Debug.Log("Head Pos 2D:" + headPos2D.ToString());
                //headJoinPos = kinectDevice.transform.TransformPoint(headJoinPos);
                //headJoinPos = WorldZero.transform.InverseTransformPoint(headJoinPos);
                quatHead = new Quaternion(headoint.Quaternion.X, headoint.Quaternion.Y, headoint.Quaternion.Z, headoint.Quaternion.W);
             
                var u = quatHead.eulerAngles;
                quatHead = Quaternion.Euler(u.x, u.y+180, u.z);//(u.x, u.y + 180, u.z)
                
                //x.transform.position = new Vector3(headJoinPos.x, -headJoinPos.y, headJoinPos.z);
                //x.transform.rotation = quatHead;

                Transform sourceTransform = kinectDevice.transform;
                Transform destinationTransform = WorldZero.transform;

                headPosWC = destinationTransform.InverseTransformPoint(sourceTransform.TransformPoint(headJoinPos));
                quatHeadWC = Quaternion.Inverse(destinationTransform.rotation) * (sourceTransform.rotation * quatHead);
                x.transform.position = headPosWC;
                x.transform.rotation = quatHeadWC;

                var angle = quatHead.eulerAngles;
                //Debug.Log(headJoinPos - x.transform.position);
                //headPosInfo.text = headJoinPos.ToString() + angle.ToString();
                //Instantiate(x, headJoinPos, new Quaternion(0, 0, 0, 0));
                //gameObjectPosInfo.text = angle.ToString();
                string reply = Stream(capture, new Vector3(headPos2D.Value.X,headPos2D.Value.Y,0), angle);
                //Debug.Log("Reply:" + reply);
                //gameObjectPosInfo.text = reply;
                //Debug.Log(gameObjectPosInfo.text);
                string[] values = reply.Split(new char[0], StringSplitOptions.RemoveEmptyEntries);

                float i = 0.0f;
                float j = 0.0f;
                float k = 0.0f;
                if(values.Length > 0)
                {
                    if (float.TryParse(values[0], out i) && values.Length==3)
                    {
                        i = float.Parse(values[0]);
                        j = float.Parse(values[1]);
                        k = float.Parse(values[2]);
                        gazeVector = new Vector3(-i, -j, k);
                        //Debug.DrawRay(x.transform.position, gazeVector * 1000, Color.green, 0.5f);
                        gazeVector = kinectDevice.transform.TransformDirection(gazeVector);
                        //Debug.DrawRay(x.transform.position, gazeVector * 1000, Color.yellow, 0.5f);
                        gazeVector = WorldZero.transform.InverseTransformVector(gazeVector);
                        Debug.DrawRay(x.transform.position, gazeVector * 1000, Color.red, 0.5f);
                    }

                }
                
                Debug.Log(RobA.transform.position.ToString());
                
                //RobA.transform.rotation = Quaternion.Inverse(WorldZero.transform.rotation)*(RobA.transform.rotation * );
                //Debug.Log(kinectDevice.transform.position);
                errorA = Vector3.Angle(RobA.transform.position - headPosWC, gazeVector);
                errorB = Vector3.Angle(RobB.transform.position - headPosWC, gazeVector);
                
                //Debug.DrawRay(new Vector3(0, 0, 0), WorldZero.transform.InverseTransformPoint(RobA.transform.position) - headPosWC, Color.cyan);
               // Debug.DrawRay( new Vector3(0, 0, 0), gazeVector, Color.cyan);
                
                //Debug.DrawLine(WorldZero.transform.InverseTransformPoint(RobB.transform.position) - headPosWC, gazeVector, Color.green); 
                /*
                if (errorA < 5)
                {
                    robAerror.text = "Looking Here" + errorB.ToString();
                    robAerror.color = Color.red;
                }
                else 
                {
                    robAerror.color = Color.black;
                    robAerror.text = "Not Looking Here" + errorB.ToString();
                }
                    


                if (errorB < 5)
                {
                    robBerror.text = "Looking Here" + errorB.ToString();
                    robBerror.color = Color.red;
                }
                else
                {
                    robBerror.text = "Not Looking Here" + errorB.ToString();
                    robBerror.color = Color.black;

                }
                  */  
                //Debug.Log("The Gaze: "+ i.ToString()+j.ToString()+k.ToString());


            }
        }
            
    }

    private string Stream(Capture capture, Vector3 headJoinPos, Vector3 angle)
    {
        
        using (var color = capture.Color)
        {
           
            //using (MemoryStream ms = new MemoryStream())
            {

                byte[] imageData = kinectColorTexture.EncodeToJPG();

                //send data length and random integer first
                byte[] dataLength = BitConverter.GetBytes(imageData.Length);
                //Int32 a = 123;Convert the depth to bytes
                byte[] xBytes = BitConverter.GetBytes(headJoinPos.x);
                byte[] yBytes = BitConverter.GetBytes(headJoinPos.y);
                byte[] zBytes = BitConverter.GetBytes(headJoinPos.z);
                byte[] xRotBytes = BitConverter.GetBytes(angle.x);
                byte[] yRotBytes = BitConverter.GetBytes(angle.y);
                byte[] zRotBytes = BitConverter.GetBytes(angle.z);

                stream.Write(dataLength, 0, dataLength.Length);
                stream.Write(xBytes, 0, xBytes.Length);
                stream.Write(yBytes, 0, yBytes.Length);
                stream.Write(zBytes, 0, zBytes.Length);
                stream.Write(xRotBytes, 0, xRotBytes.Length);
                stream.Write(yRotBytes, 0, yRotBytes.Length);
                stream.Write(zRotBytes, 0, zRotBytes.Length);



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
                return reply;

            }

           
        }
    }
}
