using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using System.Globalization; 
using System;

namespace EVP
{
public class CameraRender : MonoBehaviour
{
    public int textureWidth = 1024;
    public int textureHeight = 1024;
    public Camera renderCamera;
    public Image uiImage; // Referência para a UI Image
    public RawImage rawImage; // Alternativa usando RawImage
    private RenderTexture renderTexture;

    private const int PORT_SEND = 5000;    // Porta para enviar imagens
    private const int PORT_RECEIVE = 5001; // Porta para receber comandos
    private string serverIP = "127.0.0.1";
    
    private UdpClient sendClient;
    private UdpClient receiveClient;
    private Thread receiveThread;
    private bool isRunning = true;
    

    public float speed;
    public float steering;


    public VehicleController target;
	public float steerInterval = 2.0f;
	public float steerIntervalTolerance = 1.0f;
	public float steerChangeRate = 1.0f;
	[Range(0,1)]
	public float steerStraightRandom = 0.4f;

	[Space(5)]
	public float throttleInterval = 5.0f;
	public float throttleIntervalTolerance = 2.0f;
	public float throttleChangeRate = 3.0f;
	[Range(0,1)]
	public float throttleForwardRandom = 0.8f;

	float m_targetSteer = 0.0f;
	float m_nextSteerTime = 0.0f;
	float m_targetThrottle = 0.0f;
	float m_targetBrake = 0.0f;
	float m_nextThrottleTime = 0.0f;


    private bool clientConnected = false;
    private DateTime lastConnectionAttempt = DateTime.MinValue;
    private float connectionRetryInterval = 2.0f; // seconds

    
    void Start()
    {
     
        SetupCamera();
        SetupNetwork();
        StartReceiving();
    }

     void OnEnable()
        {
            if (target == null)
                target = GetComponent<VehicleController>();
        }

    
    void OnDisable()
    {
        if (renderCamera != null)
            renderCamera.targetTexture = null;
            
        if (renderTexture != null)
            renderTexture.Release();
    }
    
    // Método auxiliar para converter RenderTexture para Texture2D
    private Texture2D toTexture2D(RenderTexture rTex)
    {
        Texture2D tex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGB24, false);
        RenderTexture.active = rTex;
        tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        tex.Apply();
        return tex;
    }
    
    // Método para acessar a render texture de outros scripts
    public RenderTexture GetRenderTexture()
    {
        return renderTexture;
    }

    
    void SetupCamera()
    {

        renderTexture = new RenderTexture(textureWidth, textureHeight, 24);
        renderTexture.antiAliasing = 2;
        renderTexture.filterMode = FilterMode.Bilinear;
        
      
        if (renderCamera == null)
            renderCamera = GetComponent<Camera>();
            
        renderCamera.targetTexture = renderTexture;
        
        //  UI Image para mostrar a render texture
        if (uiImage != null)
        {
            Sprite sprite = Sprite.Create(
                toTexture2D(renderTexture),
                new Rect(0, 0, renderTexture.width, renderTexture.height),
                new Vector2(0.5f, 0.5f)
            );
            uiImage.sprite = sprite;
        }
        
        //  recomendado para render textures)
        if (rawImage != null)
        {
            rawImage.texture = renderTexture;
        }

    
            
        StartCoroutine(SendCameraFrames());
    }
    
    void SetupNetwork()
    {
        sendClient = new UdpClient();
        receiveClient = new UdpClient(PORT_RECEIVE);
    }
    
    void StartReceiving()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.Start();
    }
    
/*
    IEnumerator SendCameraFrames()
    {
        while (true)
        {
            // Capturar frame da câmera
            Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height);
            RenderTexture.active = renderTexture;
            tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            tex.Apply();
            
            // Converter para JPG para reduzir tamanho
            byte[] bytes = tex.EncodeToJPG(75);
            
            // Enviar para Python
            try
            {
                sendClient.Send(bytes, bytes.Length, serverIP, PORT_SEND);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Erro ao enviar frame: {e.Message}");
            }
            
            Destroy(tex);
            yield return new WaitForSeconds(0.033f); // ~30 FPS
        }
    }
*/


        IEnumerator SendCameraFrames()
        {
            while (true)
            {
                // Check if we should try to connect or send frames
                if (!clientConnected)
                {
                    // Don't spam connection attempts
                    if ((DateTime.Now - lastConnectionAttempt).TotalSeconds >= connectionRetryInterval)
                    {
                        // Try to ping the Python client
                        try
                        {
                            // Send a small ping packet
                            byte[] pingData = Encoding.ASCII.GetBytes("ping");
                            sendClient.Send(pingData, pingData.Length, serverIP, PORT_SEND);
                            clientConnected = true;
                            Debug.Log("Client connection established");
                        }
                        catch (System.Exception)
                        {
                            // Silent fail - we'll try again later
                            clientConnected = false;
                            lastConnectionAttempt = DateTime.Now;
                            yield return new WaitForSeconds(connectionRetryInterval);
                            continue;
                        }
                    }
                    else
                    {
                        // Wait before trying to connect again
                        yield return new WaitForSeconds(0.5f);
                        continue;
                    }
                }

                try
                {
                    // Only capture and send frames if connected
                    Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height);
                    RenderTexture.active = renderTexture;
                    tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
                    tex.Apply();
                    
                    // Compress to JPG to reduce size (consider different quality settings based on needs)
                    byte[] bytes = tex.EncodeToJPG(75);
                    
                    // Send to Python
                    sendClient.Send(bytes, bytes.Length, serverIP, PORT_SEND);
                    
                    Destroy(tex);
                }
                catch (System.Exception e)
                {
                    Debug.LogWarning($"Connection lost: {e.Message}");
                    clientConnected = false;
                    lastConnectionAttempt = DateTime.Now;
                }
                
                // Frame rate limiting
                yield return new WaitForSeconds(0.033f); // ~30 FPS
            }
        }

/*
    private float NetworkBytesToFloat(byte[] bytes, int startIndex)
    {
        // Manually convert network byte order (big-endian) to float
        byte[] floatBytes = new byte[4];
        floatBytes[0] = bytes[startIndex + 3];
        floatBytes[1] = bytes[startIndex + 2];
        floatBytes[2] = bytes[startIndex + 1];
        floatBytes[3] = bytes[startIndex + 0];
        return BitConverter.ToSingle(floatBytes, 0);
    }

    
void ReceiveData()
{
    IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, PORT_RECEIVE);
    
    while (isRunning)
    {
        try
        {
            byte[] receiveBytes = receiveClient.Receive(ref remoteEndPoint);
            if (receiveBytes.Length == 8) // 2 floats, 4 bytes each
            {
                speed = BitConverter.ToSingle(receiveBytes, 0);
                steering = BitConverter.ToSingle(receiveBytes, 4);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Erro ao receber dados: {e.Message}");
        }
    }
}
*/

        void ReceiveData()
        {
            IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, PORT_RECEIVE);
            
            while (isRunning)
            {
                try
                {
                    byte[] receiveBytes = receiveClient.Receive(ref remoteEndPoint);
                    
                    // Any successful receive means the client is connected
                    clientConnected = true;
                    
                    if (receiveBytes.Length == 8) // 2 floats, 4 bytes each
                    {
                        // Your existing byte conversion code
                        if (BitConverter.IsLittleEndian)
                        {
                            Array.Reverse(receiveBytes, 0, 4);
                            Array.Reverse(receiveBytes, 4, 4);
                        }
                        
                        speed = BitConverter.ToSingle(receiveBytes, 0);
                        steering = BitConverter.ToSingle(receiveBytes, 4);
                    }
                    else if (receiveBytes.Length == 4 && Encoding.ASCII.GetString(receiveBytes) == "ping")
                    {
                        // Respond to ping
                        byte[] pongData = Encoding.ASCII.GetBytes("pong");
                        sendClient.Send(pongData, pongData.Length, remoteEndPoint.Address.ToString(), PORT_SEND);
                    }
                }
                catch (System.Exception e)
                {
                    Debug.LogWarning($"Receive error: {e.Message}");
                    clientConnected = false;
                    // Add a small sleep to prevent CPU spiking in case of continuous errors
                    Thread.Sleep(100);
                }
            }
        }

    
    void Update()
    {
      if (target == null) return;

	      
    }

 void FixedUpdate()
        {
             if (target == null) return;

       
	

		target.steerInput = steering;
		target.throttleInput =speed;// Mathf.Clamp(speed, -1.0f, 1.0f);;

     
       
        }

    
    void OnDestroy()
    {
        isRunning = false;
        if (receiveThread != null)
            receiveThread.Abort();
            
        if (sendClient != null)
            sendClient.Close();
            
        if (receiveClient != null)
            receiveClient.Close();
            
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
    }
}
}
