﻿/// <summary>
/// This script is used to send car data in real-time to the server
/// </summary>
/// 
using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using SocketIO;
using UnityStandardAssets.Vehicles.Car;
using System;
using System.Security.AccessControl;
using System.IO;

public class CommandServer : MonoBehaviour
{
    public CarRemoteControl mainCar;
    public Camera mainCamera;
    //public bool saveData = false;
    private SocketIOComponent _socket;
    private CarController _carController;

    private AllDataCapture dataCapturer = new AllDataCapture();

    int imgWidth = 1242;
    int imgHeight = 375;

    Shader rgbShader;
    Shader depthShader;
    float[] pointCloud;

    int num = 0;

    // Use this for initialization
    void Start()
    {
        _socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
        _socket.On("open", OnOpen);
        _socket.On("steer", OnSteer);
        _socket.On("manual", onManual);
        _carController = mainCar.GetComponent<CarController>();

        dataCapturer.init(GetComponent<Renderer>(), mainCamera);
        
        rgbShader = Shader.Find("Standard");
        depthShader = Shader.Find("Custom/DepthGrayscale");
        
        pointCloud = new float[imgHeight * imgWidth * 4];
    }

    // Update is called once per frame according to FPS
    void Update() { }

    void OnOpen(SocketIOEvent obj)
    {
        Debug.Log("Connection Open");
        EmitTelemetry(obj);
    }

    void OnSteer(SocketIOEvent obj)
    {
        JSONObject jsonObject = obj.data;
        mainCar.SteeringAngle = float.Parse(jsonObject.GetField("steering_angle").str);
        mainCar.Acceleration = float.Parse(jsonObject.GetField("throttle").str);
        //CarRemoteControl.AddInput = float.Parse(jsonObject.GetField("add_input").str);
        EmitTelemetry(obj);
    }

    void onManual(SocketIOEvent obj)
    {
        EmitTelemetry(obj);
    }
    
    void EmitTelemetry(SocketIOEvent obj)
    {
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            //print("Sending data to the server...");
            // send only if it's not being manually driven
            if ((Input.GetKey(KeyCode.W)) || (Input.GetKey(KeyCode.S)))
            {
                _socket.Emit("telemetry", new JSONObject());
            }
            else
            {
                num++;
//                 //depth image capture
//                 dataCapturer.changeShader(depthShader);
//                 byte[] imageDepth = dataCapturer.getRenderResult();
//                 if (saveData)
//                     File.WriteAllBytes(dataPath + "//depth//" + num.ToString().PadLeft(6, '0') + ".jpg", imageDepth);
// 
//                 //point cloud capture
//                 byte[] pcByteArray = dataCapturer.getPointCloud();
//                 if (saveData)
//                     File.WriteAllBytes(dataPath + "//velodyne//" + num.ToString().PadLeft(6, '0') + ".bin", pcByteArray);


                //RGB image capture
                //dataCapturer.changeShader(rgbShader);
                byte[] image = dataCapturer.getRenderResult();
//                 if (saveData)
//                     File.WriteAllBytes(dataPath + "//image_2//" + num.ToString().PadLeft(6, '0') + ".jpg", image);

                RenderTexture.active = null;
                mainCamera.targetTexture = null;

                // Collect Data from the Car
                Dictionary<string, string> data = new Dictionary<string, string>();
                data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
                data["throttle"] = _carController.AccelInput.ToString("N4");
                data["speed"] = _carController.CurrentSpeed.ToString("N4");
                data["image"] = Convert.ToBase64String(image);
                //data["point_cloud"] = Convert.ToBase64String(pcByteArray);
                //data["image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));
                _socket.Emit("telemetry", new JSONObject(data));
            }
        });

    }
}